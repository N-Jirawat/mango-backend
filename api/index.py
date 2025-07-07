# api/index.py (ไฟล์นี้จะอยู่ในโฟลเดอร์ api/ ของโปรเจกต์)

from flask import Flask, request, jsonify # send_from_directory ไม่จำเป็นแล้ว
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
# import gdown # ไม่จำเป็นแล้ว ถ้าโมเดลอยู่ใน repo
import checkMango # ตรวจสอบให้แน่ใจว่า checkMango.py อยู่ในโฟลเดอร์ api/ เดียวกัน

# สร้าง Flask App โดยไม่ต้องระบุ static_folder หรือ static_url_path
# เนื่องจาก Vercel จะ Serve Frontend แยกต่างหาก
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
# แนะนำให้เก็บ API keys ใน Environment Variables บน Vercel
# เช่น VERCEL_CLOUD_NAME, VERCEL_API_KEY, VERCEL_API_SECRET
# เพื่อความปลอดภัยและง่ายต่อการจัดการ
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', 'dsf25dlca'),
    api_key=os.environ.get('CLOUDINARY_API_KEY', '978124749794588'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', 's_KmqxdLxYeW8H-dCbLkWFx_ZTQ'),
)
# หากคุณใช้ Environment Variables บน Vercel ให้ไปเพิ่มค่าเหล่านี้ใน Project Settings -> Environment Variables

# -------------------------------
# โหลดโมเดลจาก Local Path
# -------------------------------
# ใช้ os.path.join และ os.path.dirname(__file__) เพื่อให้ Path ถูกต้องเสมอ
# ไม่ว่าไฟล์จะถูกรันที่ไหน
model_base_dir = os.path.join(os.path.dirname(__file__), "Model")
model_path = os.path.join(model_base_dir, "model_efficientnetv2s_224_R1.keras")

# gdown และ model_file_id ไม่จำเป็นแล้ว เพราะโมเดลจะอยู่ใน repo
# model_file_id = "1cf-SSC8SdcgbJYhqn_-fu7hDhmgUcCST"
# model_url = f"https://drive.google.com/uc?id={model_file_id}"

# Vercel จะจัดการการดาวน์โหลดโมเดลอัตโนมัติจาก Git LFS (ถ้าใช้)
# หรือถ้าเป็นไฟล์ขนาดเล็ก ก็จะอยู่ใน repo โดยตรง
# ไม่ต้องเช็ค os.path.exists หรือ gdown.download ที่นี่แล้ว
# เพราะมันควรจะอยู่แล้วตอน Build Time

def verify_model_file(model_path):
    """ตรวจสอบความถูกต้องของไฟล์โมเดล (รันตอน Build Time)"""
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
# โหลดโมเดลหลัก (จะถูกโหลดเพียงครั้งเดียวตอน Cold Start)
# -------------------------------
print(f"Checking model file: {model_path}")
is_valid, message = verify_model_file(model_path)
if not is_valid:
    print(f"❌ Model file issue: {message}")
    print(f"📝 Please ensure your model file is located at: {os.path.abspath(model_path)}")
    print("   Supported formats: .keras, .h5")
    raise RuntimeError(f"Model file not found or invalid: {message}")

try:
    print("Loading main model...")
    model = load_model(model_path)
    print(f"✅ Main model loaded successfully from {model_path}")
    print(f"   Model input shape: {model.input_shape}")
    print(f"   Model output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading main model: {e}")
    raise RuntimeError(f"Failed to load main model from {model_path}: {e}")

# -------------------------------
# โหลด embedding model และ reference embeddings (จะถูกโหลดเพียงครั้งเดียวตอน Cold Start)
# -------------------------------
if USE_FILTER:
    # โหลด embedding model
    try:
        checkMango.embedding_model = EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg")
        print("✅ EfficientNetV2S embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        raise RuntimeError(f"Failed to load embedding model: {e}")

    # โหลด reference embeddings จาก Local Path
    embedding_path = os.path.join(model_base_dir, "mango_reference_embeddings.npy")
    # embedding_file_id และ embedding_url ไม่จำเป็นแล้ว

    try:
        # ไม่ต้องเช็ค os.path.exists หรือ gdown.download ที่นี่แล้ว
        checkMango.mango_embeddings = np.load(embedding_path)
        print(f"✅ Loaded {embedding_path} with shape {checkMango.mango_embeddings.shape}")
    except Exception as e:
        print(f"❌ Error loading {embedding_path}: {e}")
        # ถ้าโหลดไม่ได้ ควรยกเว้น error หรือให้มันยังรันได้แต่ไม่มี filter
        # ในที่นี้เลือก raise error เพื่อให้แน่ใจว่าระบบทำงานถูกต้อง
        raise RuntimeError(f"Failed to load mango embeddings from {embedding_path}: {e}")
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
                similarity = 0.0 # ถ้าเกิด error ในการตรวจ ให้ค่าเป็น 0 ไปก่อน หรือจะ return error เลยก็ได้

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
        "embedding_path": os.path.join(model_base_dir, "mango_reference_embeddings.npy") if USE_FILTER else None # ปรับ path
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
        "model_loaded": 'model' in globals() and model is not None, # ตรวจสอบตัวแปร model
        "embedding_model_loaded": hasattr(checkMango, 'embedding_model') and checkMango.embedding_model is not None,
        "mango_embeddings_loaded": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "use_filter": USE_FILTER
    })

