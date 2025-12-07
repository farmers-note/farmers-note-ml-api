from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
from collections import OrderedDict

app = Flask(__name__)

IMG_SIZE = 224
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 최대 캐시: 최근 사용 모델 2개만 메모리에 유지
MAX_CACHED_MODELS = 2

# 모델 메타 정보 (파일 경로 + 클래스)
MODEL_INFO = {
    "corn": ("corn_disease_model_merged.onnx",
             ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']),
    "crop": ("crop_classification_merged.onnx",
             ["Cherry", "Coffee-plant", "Cucumber", "Fox_nut(Makhana)", "Lemon", "Olive-tree", "Pearl_millet(bajra)", "Tobacco-plant", "almond", "banana", "cardamom", "chilli", "clove", "coconut", "cotton", "gram", "jowar", "jute", "maize", "mustard-oil", "papaya", "pineapple", "rice", "soyabean", "sugarcane", "sunflower", "tea", "tomato", "vigna-radiati(Mung)", "wheat"]),
    "cucumber": ("cucumber_disease_model_merged.onnx",
                 ['Anthracnose', 'Bacterial Wilt', 'Belly Rot', 'Downy Mildew', 'Fresh Cucumber', 'Fresh Leaf', 'Gummy Stem Blight', 'Pythium Fruit Rot']),
    "potato": ("potato_disease_model_merged.onnx",
               ['Early Blight', 'Fungal Diseases', 'Healthy', 'Late Blight', 'Plant Pests', 'Potato Cyst Nematode', 'Potato Virus']),
    "rice": ("rice_disease_model_merged.onnx",
             ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'neck_blast', 'rice_hispa', 'sheath_blight', 'tungro']),
    "strawberry": ("strawberry_disease_model_merged.onnx",
                   ['Strawberry___Leaf_scorch', 'Strawberry___healthy', 'alternaria_leaf_blight', 'angular_leafspot', 'anthocyanosis', 'anthracnose', 'anthracnose_fruit_rot', 'ants', 'aphid', 'aphid_effects', 'ascochyta_blight', 'bacterial_spot', 'black_chaff', 'black_rot', 'black_spots', 'blossom_blight', 'blossom_end_rot', 'botrytis_cinerea', 'burn', 'calciumdeficiency', 'canker', 'caterpillars', 'cherry_leaf_spot', 'coccomyces_of_pome_fruits', 'colorado_beetle', 'colorado_beetle_effects', 'corn_downy_mildew', 'cyclamen_mite', 'downy_mildew', 'dry_rot', 'edema', 'esca', 'eyespot', 'frost_cracks', 'galls', 'gray_mold', 'grey_mold', 'gryllotalpa', 'gryllotalpa_effects', 'healthy', 'late_blight', 'leaf_deformation', 'leaf_miners', 'leaf_spot', 'leaves_scorch', 'lichen', 'loss_of_foliage_turgor', 'marginal_leaf_necrosis', 'mealybug', 'mechanical_damage', 'monilia', 'mosaic_virus', 'northern_leaf_blight', 'nutrient_deficiency', 'pear_blister_mite', 'pest_damage', 'polypore', 'powdery_mildew', 'powdery_mildew_fruit', 'powdery_mildew_leaf', 'rust', 'scab', 'scale', 'shot_hole', 'shute', 'slugs', 'slugs_caterpillars_effects', 'sooty_mold', 'spider_mite', 'thrips', 'tubercular_necrosis', 'verticillium_wilt', 'whitefly', 'wilting', 'wireworm', 'wireworm_effects', 'yellow_leaves']),
    "tomato": ("tomato_disease_model_merged.onnx",
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
