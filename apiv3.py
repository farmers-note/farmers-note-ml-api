from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
from collections import OrderedDict
import mysql.connector
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

IMG_SIZE = 224
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

WEBAPP_DIR = "build"

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_webapp(path):
    """
    루트 URL(/) 및 웹앱의 모든 경로를 빌드 폴더의 index.html로 라우팅합니다.
    """
    if path != "" and os.path.exists(os.path.join(WEBAPP_DIR, path)):
        return send_from_directory(WEBAPP_DIR, path)
    
    elif path.startswith("api") or path.startswith("records"):
        pass
        
    return send_from_directory(WEBAPP_DIR, 'index.html')

# 최대 캐시: 최근 사용 모델 2개만 메모리에 유지
MAX_CACHED_MODELS = 2

# 모델 메타 정보 (파일 경로 + 클래스)
MODEL_INFO = {
    "corn": ("./model/corn_disease_model_merged.onnx",
             ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']),
    "crop": ("./model/crop_classification_merged.onnx",
             ["Cherry", "Coffee-plant", "Cucumber", "Fox_nut(Makhana)", "Lemon", "Olive-tree", "Pearl_millet(bajra)", "Tobacco-plant", "almond", "banana", "cardamom", "chilli", "clove", "coconut", "cotton", "gram", "jowar", "jute", "maize", "mustard-oil", "papaya", "pineapple", "rice", "soyabean", "sugarcane", "sunflower", "tea", "tomato", "vigna-radiati(Mung)", "wheat"]),
    "cucumber": ("cucumber_disease_model_merged.onnx",
                 ['Anthracnose', 'Bacterial Wilt', 'Belly Rot', 'Downy Mildew', 'Fresh Cucumber', 'Fresh Leaf', 'Gummy Stem Blight', 'Pythium Fruit Rot']),
    "potato": ("./modelpotato_disease_model_merged.onnx",
               ['Early Blight', 'Fungal Diseases', 'Healthy', 'Late Blight', 'Plant Pests', 'Potato Cyst Nematode', 'Potato Virus']),
    "rice": ("./model/rice_disease_model_merged.onnx",
             ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'neck_blast', 'rice_hispa', 'sheath_blight', 'tungro']),
    "strawberry": ("./strawberry_disease_model_merged.onnx",
                   ['Strawberry___Leaf_scorch', 'Strawberry___healthy', 'alternaria_leaf_blight', 'angular_leafspot', 'anthocyanosis', 'anthracnose', 'anthracnose_fruit_rot', 'ants', 'aphid', 'aphid_effects', 'ascochyta_blight', 'bacterial_spot', 'black_chaff', 'black_rot', 'black_spots', 'blossom_blight', 'blossom_end_rot', 'botrytis_cinerea', 'burn', 'calciumdeficiency', 'canker', 'caterpillars', 'cherry_leaf_spot', 'coccomyces_of_pome_fruits', 'colorado_beetle', 'colorado_beetle_effects', 'corn_downy_mildew', 'cyclamen_mite', 'downy_mildew', 'dry_rot', 'edema', 'esca', 'eyespot', 'frost_cracks', 'galls', 'gray_mold', 'grey_mold', 'gryllotalpa', 'gryllotalpa_effects', 'healthy', 'late_blight', 'leaf_deformation', 'leaf_miners', 'leaf_spot', 'leaves_scorch', 'lichen', 'loss_of_foliage_turgor', 'marginal_leaf_necrosis', 'mealybug', 'mechanical_damage', 'monilia', 'mosaic_virus', 'northern_leaf_blight', 'nutrient_deficiency', 'pear_blister_mite', 'pest_damage', 'polypore', 'powdery_mildew', 'powdery_mildew_fruit', 'powdery_mildew_leaf', 'rust', 'scab', 'scale', 'shot_hole', 'shute', 'slugs', 'slugs_caterpillars_effects', 'sooty_mold', 'spider_mite', 'thrips', 'tubercular_necrosis', 'verticillium_wilt', 'whitefly', 'wilting', 'wireworm', 'wireworm_effects', 'yellow_leaves']),
    "tomato": ("./model/tomato_disease_model_merged.onnx",
               ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']),
}

# LRU 캐시 저장소
loaded_models = OrderedDict()

def load_model_once(model_key):
    # 이미 캐시에 있으면 최신으로 이동
    if model_key in loaded_models:
        loaded_models.move_to_end(model_key)
        return loaded_models[model_key]

    # 캐시 꽉 차면 가장 오래된 모델 제거
    if len(loaded_models) >= MAX_CACHED_MODELS:
        old_key, (old_session, _, _) = loaded_models.popitem(last=False)
        print(f"🧹 Unload model from memory: {old_key}")
        del old_session

    print(f"📌 Loading model into memory: {model_key}")
    model_path, classes = MODEL_INFO[model_key]
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    loaded_models[model_key] = (session, input_name, classes)
    return loaded_models[model_key]


def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/predict/<model_key>", methods=["POST"])
def predict(model_key):
    if model_key not in MODEL_INFO:
        return jsonify({"error": "unknown model key"}), 400

    if "file" not in request.files:
        return jsonify({"error": "file field required"}), 400

    img = Image.open(io.BytesIO(request.files["file"].read()))
    tensor = preprocess(img)

    session, input_name, classes = load_model_once(model_key)
    logits = session.run(None, {input_name: tensor})[0][0]

    idx = int(np.argmax(logits))
    label = classes[idx]

    return jsonify({
        "model": model_key,
        "prediction_index": idx,
        "prediction_class": label
    })

DB_CONFIG = {
    'host': 'farmersnote-mysql',
    'user': 'farmersnote',
    'password': 'farmersnote',
    'database': 'farmersnote'
}


def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None

def init_db():
    """테이블이 존재하지 않으면 생성합니다."""
    conn = get_db_connection()
    if conn is None:
        return
    
    cursor = conn.cursor()
    # DATE 타입은 'YYYY-MM-DD' 형식의 문자열로 삽입 가능
    table_create_query = """
    CREATE TABLE IF NOT EXISTS field_records (
        id INT AUTO_INCREMENT PRIMARY KEY,
        field_name VARCHAR(100) NOT NULL,
        crop_type VARCHAR(50) NOT NULL,
        size_pyeong INT NOT NULL,
        disease_status VARCHAR(100) NOT NULL,
        record_date DATE NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    try:
        cursor.execute(table_create_query)
        conn.commit()
        print("✅ field_records 테이블 준비 완료.")
    except mysql.connector.Error as err:
        print(f"Error creating table: {err}")
    finally:
        cursor.close()
        conn.close()

init_db();

@app.route("/records", methods=["POST"])
def add_field_record():
    data = request.get_json()
    
    required_fields = ['fieldName', 'cropType', 'sizePyeong', 'diseaseStatus', 'recordDate']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing one or more required fields"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO field_records 
    (field_name, crop_type, size_pyeong, disease_status, record_date) 
    VALUES (%s, %s, %s, %s, %s)
    """
    
    try:
        # 데이터 유효성 검사 (평수: 정수, 날짜: YYYY-MM-DD 형식)
        size_pyeong = int(data['sizePyeong'])
        datetime.strptime(data['recordDate'], '%Y-%m-%d')
        
        record_data = (
            data['fieldName'],
            data['cropType'],
            size_pyeong,
            data['diseaseStatus'],
            data['recordDate']
        )
        
        cursor.execute(insert_query, record_data)
        conn.commit()
        
        # 삽입된 레코드의 ID를 가져옴
        new_id = cursor.lastrowid

        return jsonify({
            "message": "Record added successfully",
            "id": new_id
        }), 201

    except ValueError:
        return jsonify({"error": "Invalid data type for sizePyeong or recordDate format (must be YYYY-MM-DD)"}), 400
    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"DB insert error: {err}"}), 500
    finally:
        cursor.close()
        conn.close()

@app.route("/records", methods=["GET"])
def get_field_records():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = conn.cursor(dictionary=True) # dictionary=True로 설정하여 딕셔너리 형태로 결과 반환
    
    # record_date를 기준으로 내림차순 정렬 (최신 기록이 먼저 오도록)
    select_query = """
    SELECT id, field_name, crop_type, size_pyeong, disease_status, record_date, created_at 
    FROM field_records 
    ORDER BY record_date DESC, created_at DESC
    """
    
    try:
        cursor.execute(select_query)
        records = cursor.fetchall()
        
        # 날짜/시간 객체를 JSON 직렬화를 위해 문자열로 변환
        records_list = []
        for record in records:
            record['record_date'] = record['record_date'].isoformat()
            record['created_at'] = record['created_at'].isoformat()
            records_list.append(record)

        return jsonify({
            "total_count": len(records_list),
            "records": records_list
        })
    except mysql.connector.Error as err:
        return jsonify({"error": f"DB select error: {err}"}), 500
    finally:
        cursor.close()
        conn.close()

@app.route("/records/<int:record_id>", methods=["DELETE"])
def delete_field_record(record_id):
    """
    특정 ID를 가진 밭 기록을 삭제합니다.
    URL 예: DELETE /records/123
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = conn.cursor()
    
    # 해당 ID의 레코드를 삭제하는 Raw SQL 쿼리
    delete_query = "DELETE FROM field_records WHERE id = %s"
    
    try:
        cursor.execute(delete_query, (record_id,))
        conn.commit()
        
        # 실제로 삭제된 행의 개수를 확인
        rows_affected = cursor.rowcount
        
        if rows_affected == 0:
            return jsonify({
                "error": f"Record with ID {record_id} not found."
            }), 404
        else:
            return jsonify({
                "message": f"Record ID {record_id} deleted successfully."
            }), 200 # 성공적으로 삭제 시 200 OK 또는 204 No Content 사용 가능

    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"error": f"DB delete error: {err}"}), 500
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
