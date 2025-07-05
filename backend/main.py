from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import cloudinary
import cloudinary.uploader
import os
import gdown
import checkMango

app = Flask(__name__)
CORS(app)

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = (224, 224)
USE_FILTER = True  # True = เช็คว่าเป็นใบมะม่วงก่อนทำนาย, False = ทำนายเลยไม่กรอง

# ค่า confidence สำหรับเช็คใบมะม่วงและโรค
MANGO_LEAF_THRESHOLD = 0.70
DISEASE_CONFIDENCE_THRESHOLD = 0.80

model_classes = ['Anthracnose', 'Healthy', 'Sooty-mold', 'raised-spot']
class_map = {
    'Anthracnose': 'โรคแอนแทรคโนส',
    'Healthy': 'ใบปกติ',
    'Sooty-mold': 'โรคราดำ',
    'raised-spot': 'โรคใบจุดนูน',
}

# -------------------------------
# Cloudinary config (optional)
# -------------------------------
cloudinary.config(
    cloud_name='dsf25dlca',
    api_key='978124749794588',
    api_secret='s_KmqxdLxYeW8H-dCbLkWFx_ZTQ',
)

# -------------------------------
# โหลดโมเดลจาก Google Drive
# -------------------------------
model_path = "Model/model_efficientnetv2s_224_R1.keras"
model_file_id = "1cf-SSC8SdcgbJYhqn_-fu7hDhmgUcCST"  # ใส่ ID จริง
model_url = f"https://drive.google.com/uc?id={model_file_id}"

if not os.path.exists(model_path):
    print("📥 Downloading model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    gdown.download(model_url, model_path, quiet=False)
else:
    print("✅ Model already exists.")

def verify_model_file(model_path):
    """ตรวจสอบความถูกต้องของไฟล์โมเดล"""
    if not os.path.exists(model_path):
        return False, f"Model file does not exist: {model_path}"
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        return False, "Model file is empty"
    
    if file_size < 1024:  # ไฟล์เล็กกว่า 1KB น่าจะเป็นไฟล์ error
        return False, f"Model file too small ({file_size} bytes)"
    
    # ลองอ่านไฟล์ดูว่าเป็น binary file ที่ถูกต้องหรือไม่
    try:
        with open(model_path, 'rb') as f:
            header = f.read(8)
            if len(header) < 8:
                return False, "Invalid file header"
    except Exception as e:
        return False, f"Cannot read file: {e}"
    
    return True, "File appears valid"

# -------------------------------
# โหลดโมเดลหลัก
# -------------------------------
print(f"Checking model file: {model_path}")
is_valid, message = verify_model_file(model_path)
if not is_valid:
    print(f"❌ Model file issue: {message}")
    print(f"📝 Please ensure your model file is located at: {os.path.abspath(model_path)}")
    print("   Supported formats: .keras, .h5")
    raise RuntimeError(f"Model file not found or invalid: {message}")

try:
    print("Loading model...")
    model = load_model(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise RuntimeError(f"Failed to load model from {model_path}: {e}")

# -------------------------------
# โหลด embedding model และ reference embeddings
# -------------------------------
if USE_FILTER:
    # โหลด embedding model
    try:
        checkMango.embedding_model = EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg")
        print("✅ EfficientNetV2S embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        raise RuntimeError(f"Failed to load embedding model: {e}")

    # โหลด reference embeddings จาก Google Drive
    embedding_path = "Model/mango_reference_embeddings.npy"
    embedding_file_id = "1mBCsEXT7yF8xJ8K72SLHyC134Qt2Zkgo"  # เปลี่ยนเป็น ID จริง
    embedding_url = f"https://drive.google.com/uc?id={embedding_file_id}"

    try:
        if not os.path.exists(embedding_path):
            print("📥 Downloading mango_reference_embeddings.npy from Google Drive...")
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            gdown.download(embedding_url, embedding_path, quiet=False)
        
        checkMango.mango_embeddings = np.load(embedding_path)
        print(f"✅ Loaded {embedding_path} with shape {checkMango.mango_embeddings.shape}")
    except Exception as e:
        print(f"❌ Error loading {embedding_path}: {e}")
        checkMango.mango_embeddings = np.array([])
else:
    print("🔄 Mango leaf filtering is disabled (USE_FILTER = False)")
    checkMango.mango_embeddings = np.array([])

# -------------------------------
# ฟังก์ชันช่วยเตรียมภาพ
# -------------------------------
def load_and_prep_image(image_file):
    try:
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img)
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise Exception(f"Error processing image: {e}")

def validate_image_file(image_file):
    if not image_file:
        raise ValueError("No image file provided")

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    filename = image_file.filename.lower() if image_file.filename else ""
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise ValueError("Invalid image format. Supported formats: PNG, JPG, JPEG, GIF, BMP, WEBP")

    image_file.seek(0, 2)
    file_size = image_file.tell()
    image_file.seek(0)

    if file_size > 10 * 1024 * 1024:
        raise ValueError("File size too large. Maximum size is 10MB")

# -------------------------------
# API Routes
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image = request.files['image']
        validate_image_file(image)

        # ตรวจสอบว่าเป็นใบมะม่วงหรือไม่ (ถ้า USE_FILTER = True)
        similarity = 0.0
        if USE_FILTER and len(checkMango.mango_embeddings) > 0:
            try:
                image.seek(0)
                is_leaf, similarity = checkMango.is_mango_leaf_from_embedding(image, checkMango.mango_embeddings)
                if similarity < MANGO_LEAF_THRESHOLD:
                    return jsonify({
                        "prediction": "ไม่ใช่ภาพใบมะม่วง",
                        "confidence": float(similarity),
                        "raw_class": None,
                        "accuracy": 0,
                        "mango_leaf_confidence": float(similarity),
                        "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
                        "status": "rejected_not_mango_leaf"
                    })
            except Exception as e:
                print(f"Error in mango leaf detection: {e}")
                similarity = 0.0

        # ทำนายโรค
        image.seek(0)
        img_array = load_and_prep_image(image)
        prediction = model.predict(img_array, verbose=0)
        class_id = int(np.argmax(prediction))
        class_eng = model_classes[class_id]
        class_th = class_map[class_eng]
        confidence = float(prediction[0][class_id])

        # ตรวจสอบ confidence ของการทำนายโรค
        if confidence < DISEASE_CONFIDENCE_THRESHOLD:
            return jsonify({
                "prediction": "ไม่พบโรคที่ตรงกับข้อมูล",
                "confidence": confidence,
                "raw_class": class_eng,
                "accuracy": 0,
                "disease_threshold": DISEASE_CONFIDENCE_THRESHOLD,
                "status": "low_confidence"
            })

        # ส่งผลลัพธ์
        response_data = {
            "prediction": class_th,
            "confidence": confidence,
            "raw_class": class_eng,
            "accuracy": 1,
            "disease_threshold": DISEASE_CONFIDENCE_THRESHOLD,
            "status": "success"
        }

        # เพิ่มข้อมูล mango leaf confidence ถ้ามีการใช้ filter
        if USE_FILTER and len(checkMango.mango_embeddings) > 0:
            response_data["mango_leaf_confidence"] = float(similarity)
            response_data["mango_leaf_threshold"] = MANGO_LEAF_THRESHOLD

        return jsonify(response_data)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image = request.files['image']
        validate_image_file(image)

        upload_result = cloudinary.uploader.upload(image, folder="mango_diseases")
        return jsonify({
            "imageUrl": upload_result['secure_url'],
            "public_id": upload_result['public_id']
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/delete", methods=["POST"])
def delete_image():
    try:
        public_id = request.form.get('public_id') or request.json.get('public_id')
        if not public_id:
            return jsonify({"error": "No public_id provided"}), 400

        cloudinary.uploader.destroy(public_id)
        return jsonify({"result": "Image deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Delete failed: {str(e)}"}), 500

@app.route('/config', methods=['GET'])
def get_config():
    return jsonify({
        "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
        "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
        "use_filter": USE_FILTER,
        "img_size": IMG_SIZE,
        "model_classes": model_classes,
        "has_mango_embeddings": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "model_path": model_path,
        "embedding_path": "Model/mango_reference_embeddings.npy" if USE_FILTER else None
    })

@app.route('/config', methods=['POST'])
def update_config():
    global MANGO_LEAF_THRESHOLD, DISEASE_CONFIDENCE_THRESHOLD, USE_FILTER
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No config data provided"}), 400

        if 'mango_leaf_threshold' in data:
            MANGO_LEAF_THRESHOLD = float(data['mango_leaf_threshold'])
        if 'disease_confidence_threshold' in data:
            DISEASE_CONFIDENCE_THRESHOLD = float(data['disease_confidence_threshold'])
        if 'use_filter' in data:
            USE_FILTER = bool(data['use_filter'])

        return jsonify({"message": "Config updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to update config: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """ตรวจสอบสถานะของระบบ"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "embedding_model_loaded": hasattr(checkMango, 'embedding_model') and checkMango.embedding_model is not None,
        "mango_embeddings_loaded": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "use_filter": USE_FILTER
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🥭 Mango Disease Detection API")
    print("="*50)
    print(f"📂 Model file: {os.path.abspath(model_path)}")
    print(f"📂 Embedding file: {os.path.abspath('Model/mango_reference_embeddings.npy') if USE_FILTER else 'Not used'}")
    print(f"🔍 Mango leaf filtering: {'Enabled' if USE_FILTER else 'Disabled'}")
    print(f"🎯 Mango leaf threshold: {MANGO_LEAF_THRESHOLD}")
    print(f"🎯 Disease confidence threshold: {DISEASE_CONFIDENCE_THRESHOLD}")
    print("="*50)
    
    # สำหรับ Render deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)