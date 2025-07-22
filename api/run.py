# api/index.py (ไฟล์นี้จะอยู่ในโฟลเดอร์ api/ ของโปรเจกต์)

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
import checkMango 

# สร้าง Flask App
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
# Cloudinary config
# -------------------------------
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', 'dsf25dlca'),
    api_key=os.environ.get('CLOUDINARY_API_KEY', '978124749794588'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', 's_KmqxdLxYeW8H-dCbLkWFx_ZTQ'),
)

# -------------------------------
# Model Path Configuration - ระบุ path ตรงๆ
# -------------------------------

# ระบุ path ของโมเดลตรงๆ เลย
MODEL_PATH = r'C:\Users\Asus\mango-app\backend\models\EfficientNetV2S\model_efficientnetv2s_224_R1.keras'  # เปลี่ยน path นี้ตามที่ต้องการ
EMBEDDING_PATH = r'C:\Users\Asus\mango-app\api\models\mango_reference_embeddings.npy'  # เปลี่ยน path นี้ตามที่ต้องการ

# หรือจะใช้ environment variable ก็ได้
model_path = os.environ.get('MODEL_PATH', MODEL_PATH)
embedding_path = os.environ.get('EMBEDDING_PATH', EMBEDDING_PATH)

def verify_model_file(model_path):
    """ตรวจสอบความถูกต้องของไฟล์โมเดล"""
    if not os.path.exists(model_path):
        return False, f"Model file does not exist: {model_path}"
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        return False, "Model file is empty"
    
    if file_size < 1024:  # ไฟล์เล็กกว่า 1KB น่าจะเป็นไฟล์ error
        return False, f"Model file too small ({file_size} bytes)"
    
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
print(f"🔍 Checking model file: {model_path}")
is_valid, message = verify_model_file(model_path)
if not is_valid:
    print(f"❌ Model file issue: {message}")
    print(f"📝 กรุณาเปลี่ยน MODEL_PATH ใน code หรือตั้ง environment variable")
    raise RuntimeError(f"Model file not found or invalid: {message}")

try:
    print("📚 Loading main model...")
    model = load_model(model_path)
    print(f"✅ Main model loaded successfully from {model_path}")
    print(f"   📊 Model input shape: {model.input_shape}")
    print(f"   📊 Model output shape: {model.output_shape}")
    print(f"   📁 Model file size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
except Exception as e:
    print(f"❌ Error loading main model: {e}")
    raise RuntimeError(f"Failed to load main model from {model_path}: {e}")

# -------------------------------
# โหลด embedding model และ reference embeddings
# -------------------------------
if USE_FILTER:
    # โหลด embedding model
    try:
        print("📚 Loading embedding model...")
        checkMango.embedding_model = EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg")
        print("✅ EfficientNetV2S embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        raise RuntimeError(f"Failed to load embedding model: {e}")

    # โหลด reference embeddings
    if embedding_path:
        try:
            print(f"📚 Loading embeddings from: {embedding_path}")
            checkMango.mango_embeddings = np.load(embedding_path)
            print(f"✅ Loaded embeddings with shape {checkMango.mango_embeddings.shape}")
            print(f"   📁 Embeddings file size: {os.path.getsize(embedding_path) / 1024:.1f} KB")
        except Exception as e:
            print(f"❌ Error loading embeddings from {embedding_path}: {e}")
            print("⚠️  Running without mango leaf filtering")
            checkMango.mango_embeddings = np.array([])
    else:
        print("⚠️  EMBEDDING_PATH ไม่ได้กำหนด, running without mango leaf filtering")
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
        if USE_FILTER and hasattr(checkMango, 'mango_embeddings') and len(checkMango.mango_embeddings) > 0:
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
        if USE_FILTER and hasattr(checkMango, 'mango_embeddings') and len(checkMango.mango_embeddings) > 0:
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
        "embedding_path": embedding_path,
        "model_size_mb": os.path.getsize(model_path) / (1024*1024) if model_path and os.path.exists(model_path) else 0,
        "embeddings_size_kb": os.path.getsize(embedding_path) / 1024 if embedding_path and os.path.exists(embedding_path) else 0
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
        "model_loaded": 'model' in globals() and model is not None,
        "model_path": model_path,
        "embedding_model_loaded": hasattr(checkMango, 'embedding_model') and checkMango.embedding_model is not None,
        "mango_embeddings_loaded": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "embedding_path": embedding_path,
        "use_filter": USE_FILTER,
        "model_size_mb": os.path.getsize(model_path) / (1024*1024) if model_path and os.path.exists(model_path) else 0,
        "embeddings_size_kb": os.path.getsize(embedding_path) / 1024 if embedding_path and os.path.exists(embedding_path) else 0
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """แสดงข้อมูลโมเดลที่โหลดอยู่"""
    info = {
        "model_path": model_path,
        "model_exists": os.path.exists(model_path) if model_path else False,
        "model_size_mb": os.path.getsize(model_path) / (1024*1024) if model_path and os.path.exists(model_path) else 0,
        "embedding_path": embedding_path,
        "embeddings_exists": os.path.exists(embedding_path) if embedding_path else False,
        "embeddings_size_kb": os.path.getsize(embedding_path) / 1024 if embedding_path and os.path.exists(embedding_path) else 0,
        "model_loaded": 'model' in globals() and model is not None,
        "model_input_shape": model.input_shape if 'model' in globals() and model is not None else None,
        "model_output_shape": model.output_shape if 'model' in globals() and model is not None else None,
        "use_filter": USE_FILTER,
        "has_embeddings": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "embeddings_shape": checkMango.mango_embeddings.shape if hasattr(checkMango, 'mango_embeddings') and len(checkMango.mango_embeddings) > 0 else None
    }
    return jsonify(info)

# -------------------------------
# รันแอพพลิเคชัน
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"\n🚀 Starting Flask app on http://{host}:{port}")
    print(f"🔥 Debug mode: {os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'}")
    print(f"🌍 Environment: {os.environ.get('FLASK_ENV', 'production')}")
    print(f"📊 Model: {os.path.basename(model_path) if model_path else 'Not found'}")
    print(f"🔍 Filter: {'Enabled' if USE_FILTER else 'Disabled'}")
    
    app.run(
        host=host,
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )