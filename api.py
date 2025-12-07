from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

app = Flask(__name__)

# ONNX 모델 로드
model_path = "corn_disease_model_merged.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# EfficientNet ImageNet 정규화
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 224

# 클래스 이름 (학습에서 사용한 그대로)
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']


def preprocess(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize + CenterCrop 대체
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "file field required"}), 400
    
    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    tensor = preprocess(img)

    outputs = session.run(None, {input_name: tensor})
    logits = outputs[0][0]
    
    pred_index = int(np.argmax(logits))
    confidence = float( np.exp(logits[pred_index]) / np.sum(np.exp(logits)) )

    return jsonify({
        "prediction_index": pred_index,
        "prediction_class": class_names[pred_index],
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
