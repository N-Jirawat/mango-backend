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
from . import checkMango
from google.cloud import storage
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, get_jwt
from datetime import timedelta, datetime
from auth import auth_bp

# -------------------------------
# สร้าง Flask App
# -------------------------------
app = Flask(__name__)

# Secret key สำหรับ JWT
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "super-secret-key")
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=15)  # Token หมดอายุใน 30 นาที
app.config['JWT_BLACKLIST_ENABLED'] = True
app.config['JWT_BLACKLIST_TOKEN_CHECKS'] = ['access']

jwt = JWTManager(app)

# เก็บ blacklisted tokens (ใน production ควรใช้ Redis หรือ Database)
blacklisted_tokens = set()

# JWT Token Blacklist
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    return jwt_payload['jti'] in blacklisted_tokens

# Register auth blueprint
app.register_blueprint(auth_bp, url_prefix="/auth")

# -------------------------------
# CORS config (อนุญาต Frontend เรียก API)
# -------------------------------
CORS(app, origins="https://mangoleafanalyzer.onrender.com")

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = (224, 224)
USE_FILTER = True
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
# Google Cloud Storage config
# -------------------------------
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'mango-app-models-bucket')
EMBEDDINGS_GCS_PATH = "mango_reference_embeddings.npy"
MODEL_GCS_PATH = "model_efficientnetv2s_224_R2.keras"

LOCAL_MODEL_DIR = "/tmp/models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model_efficientnetv2s_224_R2.keras")
LOCAL_EMBEDDING_PATH = os.path.join(LOCAL_MODEL_DIR, "mango_reference_embeddings.npy")

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """ดาวน์โหลด Blob จาก GCS Bucket ไปยังไฟล์ในเครื่อง"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"✅ ดาวน์โหลด '{source_blob_name}' ไปยัง '{destination_file_name}' สำเร็จ")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดาวน์โหลด '{source_blob_name}' จาก GCS: {e}")
        raise

def verify_file_exists_and_not_empty(file_path):
    """ตรวจสอบว่าไฟล์มีอยู่และไม่ว่างเปล่า"""
    if not os.path.exists(file_path):
        return False, f"ไฟล์ไม่มีอยู่: {file_path}"
    if os.path.getsize(file_path) == 0:
        return False, f"ไฟล์ว่างเปล่า: {file_path}"
    return True, "ไฟล์ดูเหมือนถูกต้อง"

# -------------------------------
# โหลดโมเดลหลักและ Embedding
# -------------------------------
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ดาวน์โหลดและโหลดโมเดลหลัก
print(f"กำลังดาวน์โหลดโมเดลหลักจาก GCS: {MODEL_GCS_PATH}")
try:
    download_from_gcs(GCS_BUCKET_NAME, MODEL_GCS_PATH, LOCAL_MODEL_PATH)
    is_valid_model, model_message = verify_file_exists_and_not_empty(LOCAL_MODEL_PATH)
    if not is_valid_model:
        raise RuntimeError(f"ไฟล์โมเดลหลักไม่ถูกต้องหลังดาวน์โหลด: {model_message}")
    
    print("กำลังโหลดโมเดลหลัก...")
    model = load_model(LOCAL_MODEL_PATH)
    print(f"✅ โหลดโมเดลหลักสำเร็จจาก {LOCAL_MODEL_PATH}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดลหลัก: {e}")
    raise RuntimeError(f"ไม่สามารถโหลดโมเดลหลักจาก GCS ได้: {e}")

# โหลด embedding model และ reference embeddings
if USE_FILTER:
    try:
        checkMango.embedding_model = EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg")
        print("✅ โหลด EfficientNetV2S embedding model สำเร็จ")
    except Exception as e:
        print(f"❌ ไม่สามารถโหลด embedding model ได้: {e}")
        raise RuntimeError(f"ไม่สามารถโหลด embedding model ได้: {e}")

    print(f"กำลังดาวน์โหลดไฟล์ Embedding จาก GCS: {EMBEDDINGS_GCS_PATH}")
    try:
        download_from_gcs(GCS_BUCKET_NAME, EMBEDDINGS_GCS_PATH, LOCAL_EMBEDDING_PATH)
        is_valid_embedding, embedding_message = verify_file_exists_and_not_empty(LOCAL_EMBEDDING_PATH)
        if not is_valid_embedding:
            raise RuntimeError(f"ไฟล์ Embedding ไม่ถูกต้องหลังดาวน์โหลด: {embedding_message}")
        
        checkMango.mango_embeddings = np.load(LOCAL_EMBEDDING_PATH)
        print(f"✅ โหลด {LOCAL_EMBEDDING_PATH} สำเร็จด้วยรูปร่าง {checkMango.mango_embeddings.shape}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลดไฟล์ Embedding: {e}")
        raise RuntimeError(f"ไม่สามารถโหลด mango embeddings จาก {EMBEDDINGS_GCS_PATH} ได้: {e}")
else:
    print("🔄 การกรองใบมะม่วงถูกปิดใช้งาน (USE_FILTER = False)")
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
        raise Exception(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {e}")

def validate_image_file(image_file):
    if not image_file:
        raise ValueError("ไม่ได้ระบุไฟล์ภาพ")

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    filename = image_file.filename.lower() if image_file.filename else ""
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise ValueError("รูปแบบภาพไม่ถูกต้อง รูปแบบที่รองรับ: PNG, JPG, JPEG, GIF, BMP, WEBP")

    image_file.seek(0, 2)
    file_size = image_file.tell()
    image_file.seek(0)

    if file_size > 10 * 1024 * 1024:
        raise ValueError("ขนาดไฟล์ใหญ่เกินไป ขนาดสูงสุดคือ 10MB")

# -------------------------------
# JWT Error Handlers
# -------------------------------
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({
        "error": "Token หมดอายุแล้ว กรุณา login ใหม่",
        "code": "TOKEN_EXPIRED"
    }), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({
        "error": "Token ไม่ถูกต้อง",
        "code": "INVALID_TOKEN"
    }), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({
        "error": "ไม่มี Authorization token", 
        "hint": "ใส่ 'Authorization: Bearer <token>' ใน header",
        "code": "MISSING_TOKEN"
    }), 401

@jwt.revoked_token_loader
def revoked_token_callback(jwt_header, jwt_payload):
    return jsonify({
        "error": "Token ถูก logout แล้ว กรุณา login ใหม่",
        "code": "TOKEN_REVOKED"
    }), 401

# -------------------------------
# API Routes (ป้องกันด้วย JWT)
# -------------------------------
@app.route('/predict', methods=['POST'])
@jwt_required()  # ✅ ป้องกัน endpoint สำคัญ
def predict_image():
    try:
        # ดึงข้อมูลผู้ใช้จาก token
        current_user = get_jwt_identity()
        print(f"🔒 การทำนายโดยผู้ใช้: {current_user}")
        
        if 'image' not in request.files:
            return jsonify({"error": "ไม่ได้ระบุไฟล์ภาพ"}), 400
        
        image = request.files['image']
        validate_image_file(image)

        # ตรวจสอบว่าเป็นใบมะม่วงหรือไม่
        similarity = 0.0
        if USE_FILTER and hasattr(checkMango, 'mango_embeddings') and len(checkMango.mango_embeddings) > 0:
            try:
                image.seek(0)
                is_leaf, similarity = checkMango.is_mango_leaf_from_embedding(image, checkMango.mango_embeddings)
                if similarity < MANGO_LEAF_THRESHOLD:
                    return jsonify({
                        "prediction": "ไม่พบโรคที่ตรงกับข้อมูลในระบบ",
                        "confidence": float(similarity),
                        "raw_class": None,
                        "accuracy": 0,
                        "mango_leaf_confidence": float(similarity),
                        "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
                        "status": "rejected_not_mango_leaf",
                        "predicted_by": current_user
                    })
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการตรวจจับใบมะม่วง: {e}")
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
                "prediction": "ไม่พบโรคที่ตรงกับข้อมูลในระบบ",
                "confidence": confidence,
                "raw_class": class_eng,
                "accuracy": 0,
                "disease_at_threshold": DISEASE_CONFIDENCE_THRESHOLD,
                "status": "low_confidence",
                "predicted_by": current_user
            })

        # ส่งผลลัพธ์
        response_data = {
            "prediction": class_th,
            "confidence": confidence,
            "raw_class": class_eng,
            "accuracy": 1,
            "disease_at_threshold": DISEASE_CONFIDENCE_THRESHOLD,
            "status": "success",
            "predicted_by": current_user,
            "timestamp": datetime.utcnow().isoformat()
        }

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
@jwt_required()  # ✅ ป้องกัน upload
def upload_image():
    try:
        current_user = get_jwt_identity()
        
        if 'image' not in request.files:
            return jsonify({"error": "ไม่ได้ระบุไฟล์ภาพ"}), 400

        image = request.files['image']
        validate_image_file(image)

        upload_result = cloudinary.uploader.upload(
            image, 
            folder="mango_diseases",
            tags=[f"user_{current_user}"]  # เพิ่ม tag ระบุผู้ใช้
        )
        return jsonify({
            "imageUrl": upload_result['secure_url'],
            "public_id": upload_result['public_id'],
            "uploaded_by": current_user
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"การอัปโหลดล้มเหลว: {str(e)}"}), 500

@app.route("/delete", methods=["POST"])
@jwt_required()  # ✅ ป้องกัน delete
def delete_image():
    try:
        current_user = get_jwt_identity()
        
        public_id = request.form.get('public_id') or request.json.get('public_id')
        if not public_id:
            return jsonify({"error": "ไม่ได้ระบุ public_id"}), 400

        cloudinary.uploader.destroy(public_id)
        return jsonify({
            "result": "ลบภาพสำเร็จ",
            "deleted_by": current_user
        }), 200
    except Exception as e:
        return jsonify({"error": f"การลบล้มเหลว: {str(e)}"}), 500

# ✅ Logout endpoint (เพิ่มใหม่)
@app.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    try:
        current_user = get_jwt_identity()
        jti = get_jwt()['jti']  # JWT ID
        blacklisted_tokens.add(jti)
        
        return jsonify({
            "message": f"ออกจากระบบสำเร็จ สำหรับผู้ใช้ {current_user}",
            "logged_out_user": current_user
        }), 200
    except Exception as e:
        return jsonify({"error": f"เกิดข้อผิดพลาดในการออกจากระบบ: {str(e)}"}), 500

# ✅ ตรวจสอบสถานะ token
@app.route('/auth/verify', methods=['GET'])
@jwt_required()
def verify_token():
    try:
        current_user = get_jwt_identity()
        token_data = get_jwt()
        
        return jsonify({
            "valid": True,
            "user": current_user,
            "expires_at": token_data['exp'],
            "issued_at": token_data['iat']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Configuration endpoints (ป้องกันสำหรับ admin เท่านั้น)
@app.route('/config', methods=['GET'])
@jwt_required()
def get_config():
    current_user = get_jwt_identity()
    # เฉพาะ admin เท่านั้นที่ดู config ได้
    if current_user != "admin":
        return jsonify({"error": "ไม่มีสิทธิ์เข้าถึง"}), 403
        
    return jsonify({
        "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
        "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
        "use_filter": USE_FILTER,
        "img_size": IMG_SIZE,
        "model_classes": model_classes,
        "has_mango_embeddings": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "model_path": LOCAL_MODEL_PATH,
        "embedding_path": LOCAL_EMBEDDING_PATH if USE_FILTER else None,
        "accessed_by": current_user
    })

@app.route('/config', methods=['POST'])
@jwt_required()
def update_config():
    global MANGO_LEAF_THRESHOLD, DISEASE_CONFIDENCE_THRESHOLD, USE_FILTER
    
    current_user = get_jwt_identity()
    # เฉพาะ admin เท่านั้นที่แก้ไข config ได้
    if current_user != "admin":
        return jsonify({"error": "ไม่มีสิทธิ์เข้าถึง"}), 403
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "ไม่ได้ระบุข้อมูลการตั้งค่า"}), 400

        if 'mango_leaf_threshold' in data:
            MANGO_LEAF_THRESHOLD = float(data['mango_leaf_threshold'])
        if 'disease_confidence_threshold' in data:
            DISEASE_CONFIDENCE_THRESHOLD = float(data['disease_confidence_threshold'])
        if 'use_filter' in data:
            USE_FILTER = bool(data['use_filter'])

        return jsonify({
            "message": "อัปเดตการตั้งค่าสำเร็จ",
            "updated_by": current_user
        }), 200
    except Exception as e:
        return jsonify({"error": f"ไม่สามารถอัปเดตการตั้งค่าได้: {str(e)}"}), 500

# Health check (ไม่ต้องป้องกัน)
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": 'model' in globals() and model is not None,
        "embedding_model_loaded": hasattr(checkMango, 'embedding_model') and checkMango.embedding_model is not None,
        "mango_embeddings_loaded": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "use_filter": USE_FILTER,
        "active_sessions": len(blacklisted_tokens),
        "timestamp": datetime.utcnow().isoformat()
    })

# -------------------------------
# Auto-cleanup สำหรับ expired tokens (ทำงานใน background)
# -------------------------------
import threading
import time

def cleanup_expired_tokens():
    """ลบ blacklisted tokens ที่หมดอายุแล้วออกจาก memory"""
    while True:
        try:
            time.sleep(1800)  # ตรวจสอบทุก 30 นาที
            # ใน production จริงควรใช้ Redis หรือ Database
            # และมีกลไกลบ token ที่หมดอายุแล้วออกอัตโนมัติ
            print(f"🧹 Cleanup: {len(blacklisted_tokens)} blacklisted tokens in memory")
        except Exception as e:
            print(f"Error in token cleanup: {e}")

# เริ่ม background task (เฉพาะใน production)
if not app.debug:
    cleanup_thread = threading.Thread(target=cleanup_expired_tokens, daemon=True)
    cleanup_thread.start()

# -------------------------------
# สำหรับการรันใน Local Development
# -------------------------------
if __name__ == '__main__':
    print("\n--- กำลังเริ่ม Flask App พร้อม JWT Authentication ---")
    print("🔒 Endpoints ที่ป้องกันด้วย JWT:")
    print("  - POST /predict (ต้อง login)")
    print("  - POST /upload (ต้อง login)")
    print("  - POST /delete (ต้อง login)")
    print("  - GET/POST /config (admin เท่านั้น)")
    print("\n🔓 Public Endpoints:")
    print("  - POST /auth/login")
    print("  - GET /health")
    print("\n💡 การใช้งาน:")
    print("1. Login ที่ /auth/login ก่อน")
    print("2. เอา access_token ไปใส่ใน Header: Authorization: Bearer <token>")
    print("3. Token หมดอายุใน 30 นาที")
    print("4. POST /logout เพื่อออกจากระบบ")
    
    app.run(debug=True)