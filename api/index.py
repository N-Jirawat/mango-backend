from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import cloudinary
import cloudinary.uploader
import os
import secrets
from google.cloud import storage
import logging
import firebase_admin
from firebase_admin import credentials, auth, firestore
from functools import wraps
from datetime import datetime, timedelta
from . import checkMango

# -------------------------------
# Flask & CORS
# -------------------------------
app = Flask(__name__)
CORS(app, origins=os.environ.get("https://mangoleafanalyzer.onrender.com", "*"))

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------
# Firebase Setup
# -------------------------------
# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    # ใช้ service account key หรือ default credentials
    cred = credentials.ApplicationDefault()  # หรือ credentials.Certificate("path/to/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------------------
# User Activity Tracking with Firestore
# -------------------------------
INACTIVITY_TIMEOUT = 15 * 60  # 15 นาที

def update_user_activity_firebase(uid):
    """อัปเดตเวลาการใช้งานล่าสุดของผู้ใช้ใน Firestore"""
    try:
        user_ref = db.collection('user_activities').document(uid)
        user_ref.set({
            'last_activity': firestore.SERVER_TIMESTAMP,
            'uid': uid
        }, merge=True)
    except Exception as e:
        logging.error(f"Error updating user activity: {e}")

def is_user_active_firebase(uid):
    """ตรวจสอบว่าผู้ใช้ยังอยู่ในช่วงเวลาที่กำหนดหรือไม่"""
    try:
        user_ref = db.collection('user_activities').document(uid)
        doc = user_ref.get()
        
        if not doc.exists:
            return False
        
        data = doc.to_dict()
        last_activity = data.get('last_activity')
        
        if not last_activity:
            return False
        
        # แปลง Firestore timestamp เป็น datetime
        current_time = datetime.now()
        time_diff = (current_time - last_activity.replace(tzinfo=None)).total_seconds()
        
        return time_diff < INACTIVITY_TIMEOUT
    except Exception as e:
        logging.error(f"Error checking user activity: {e}")
        return False

def cleanup_inactive_users_firebase():
    """ลบข้อมูล activity ของ user ที่ไม่ active แล้ว"""
    try:
        cutoff_time = datetime.now() - timedelta(seconds=INACTIVITY_TIMEOUT)
        
        # Query inactive users
        inactive_query = db.collection('user_activities').where(
            'last_activity', '<', cutoff_time
        )
        
        inactive_docs = inactive_query.stream()
        batch = db.batch()
        
        for doc in inactive_docs:
            batch.delete(doc.reference)
            logging.info(f"User {doc.id} auto-logged out due to inactivity")
        
        batch.commit()
    except Exception as e:
        logging.error(f"Error cleaning up inactive users: {e}")

# -------------------------------
# Firebase Auth Decorator
# -------------------------------
def firebase_auth_required(admin_only=False):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # ดึง token จาก header
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({"error": "Missing or invalid authorization header"}), 401
                
                token = auth_header.split('Bearer ')[1]
                
                # Verify Firebase token
                decoded_token = auth.verify_id_token(token)
                uid = decoded_token['uid']
                
                # ตรวจสอบ user activity
                if not is_user_active_firebase(uid):
                    return jsonify({
                        "error": "Session expired due to inactivity",
                        "code": "INACTIVE_SESSION",
                        "message": "กรุณาเข้าสู่ระบบใหม่"
                    }), 401
                
                # อัปเดต activity
                update_user_activity_firebase(uid)
                
                # ตรวจสอบ admin role (ถ้าต้องการ)
                if admin_only:
                    user_claims = decoded_token.get('custom_claims', {})
                    if user_claims.get('role') != 'admin':
                        return jsonify({"error": "Admin access required"}), 403
                
                # เพิ่ม user info ใน request context
                request.current_user = {
                    'uid': uid,
                    'email': decoded_token.get('email'),
                    'claims': decoded_token.get('custom_claims', {})
                }
                
                return f(*args, **kwargs)
                
            except auth.ExpiredIdTokenError:
                return jsonify({"error": "Token expired", "code": "TOKEN_EXPIRED"}), 401
            except auth.InvalidIdTokenError:
                return jsonify({"error": "Invalid token", "code": "INVALID_TOKEN"}), 401
            except Exception as e:
                logging.error(f"Auth error: {e}")
                return jsonify({"error": "Authentication failed"}), 401
        
        return decorated_function
    return decorator

@app.before_request
def log_request_info():
    user = getattr(request, 'current_user', {}).get('email', 'anonymous')
    logging.info(f"{user} called {request.path}")
    
    # ทำความสะอาด inactive users ทุกๆ request
    cleanup_inactive_users_firebase()

# -------------------------------
# Rate Limiter
# -------------------------------
limiter = Limiter(app, key_func=get_remote_address)

# -------------------------------
# Models & Embeddings (เหมือนเดิม)
# -------------------------------
IMG_SIZE = (224, 224)
USE_FILTER = True
MANGO_LEAF_THRESHOLD = 0.7
DISEASE_CONFIDENCE_THRESHOLD = 0.8
model_classes = ['Anthracnose', 'Healthy', 'Sooty-mold', 'raised-spot']
class_map = {
    'Anthracnose': 'โรคแอนแทรคโนส',
    'Healthy': 'ใบปกติ',
    'Sooty-mold': 'โรคราดำ',
    'raised-spot': 'โรคใบจุดนูน',
}

GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'mango-app-models-bucket')
EMBEDDINGS_GCS_PATH = "mango_reference_embeddings.npy"
MODEL_GCS_PATH = "model_efficientnetv2s_224_R2.keras"
LOCAL_MODEL_DIR = "/tmp/models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model_efficientnetv2s_224_R2.keras")
LOCAL_EMBEDDING_PATH = os.path.join(LOCAL_MODEL_DIR, "mango_reference_embeddings.npy")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

# โหลด model
download_from_gcs(GCS_BUCKET_NAME, MODEL_GCS_PATH, LOCAL_MODEL_PATH)
model = load_model(LOCAL_MODEL_PATH)

if USE_FILTER:
    checkMango.embedding_model = EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg")
    download_from_gcs(GCS_BUCKET_NAME, EMBEDDINGS_GCS_PATH, LOCAL_EMBEDDING_PATH)
    checkMango.mango_embeddings = np.load(LOCAL_EMBEDDING_PATH)
else:
    checkMango.mango_embeddings = np.array([])

# -------------------------------
# Helper functions
# -------------------------------
def load_and_prep_image(image_file):
    image_file.seek(0)
    img = Image.open(image_file).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def validate_image_file(image_file):
    if not image_file:
        raise ValueError("ไม่ได้ระบุไฟล์ภาพ")
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    filename = image_file.filename.lower() if image_file.filename else ""
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise ValueError("รูปแบบภาพไม่ถูกต้อง")
    image_file.seek(0, 2)
    if image_file.tell() > 10 * 1024 * 1024:
        raise ValueError("ขนาดไฟล์ใหญ่เกิน 10MB")
    image_file.seek(0)

# -------------------------------
# Auth Routes
# -------------------------------
@app.route("/logout", methods=["POST"])
@firebase_auth_required()
def logout():
    """Manual logout - ลบ activity record"""
    uid = request.current_user['uid']
    try:
        # ลบ activity record จาก Firestore
        db.collection('user_activities').document(uid).delete()
        return jsonify({"message": "ออกจากระบบสำเร็จ"}), 200
    except Exception as e:
        logging.error(f"Logout error: {e}")
        return jsonify({"error": "เกิดข้อผิดพลาดในการออกจากระบบ"}), 500

@app.route("/check-activity", methods=["GET"])
@firebase_auth_required()
def check_activity():
    """ตรวจสอบสถานะ session และเวลาที่เหลือ"""
    uid = request.current_user['uid']
    
    try:
        user_ref = db.collection('user_activities').document(uid)
        doc = user_ref.get()
        
        if not doc.exists:
            return jsonify({
                "active": False,
                "message": "Session ไม่พบ กรุณาเข้าสู่ระบบใหม่"
            }), 401
        
        data = doc.to_dict()
        last_activity = data.get('last_activity')
        
        if not last_activity:
            return jsonify({
                "active": False,
                "message": "Session ข้อมูลไม่ถูกต้อง"
            }), 401
        
        current_time = datetime.now()
        time_diff = (current_time - last_activity.replace(tzinfo=None)).total_seconds()
        time_remaining = INACTIVITY_TIMEOUT - time_diff
        
        if time_remaining <= 0:
            return jsonify({
                "active": False,
                "message": "Session หมดอายุ กรุณาเข้าสู่ระบบใหม่"
            }), 401
        
        return jsonify({
            "active": True,
            "time_remaining": int(time_remaining),
            "time_remaining_minutes": int(time_remaining // 60),
            "last_activity": last_activity.isoformat(),
            "timeout_minutes": INACTIVITY_TIMEOUT // 60
        })
        
    except Exception as e:
        logging.error(f"Check activity error: {e}")
        return jsonify({"error": "ไม่สามารถตรวจสอบ session ได้"}), 500

# -------------------------------
# Predict (User / Admin)
# -------------------------------
@app.route('/predict', methods=['POST'])
@firebase_auth_required()
@limiter.limit("10/minute")
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "ไม่ได้ระบุไฟล์ภาพ"}), 400
        image = request.files['image']
        validate_image_file(image)

        # ใช้ filter ตรวจใบมะม่วง
        similarity = 0.0
        if USE_FILTER and len(checkMango.mango_embeddings) > 0:
            is_leaf, similarity = checkMango.is_mango_leaf_from_embedding(image, checkMango.mango_embeddings)
            if similarity < MANGO_LEAF_THRESHOLD:
                return jsonify({
                    "prediction": "ไม่พบโรคที่ตรงกับข้อมูลในระบบ",
                    "confidence": float(similarity),
                    "raw_class": None,
                    "accuracy": 0,
                    "mango_leaf_confidence": float(similarity),
                    "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
                    "status": "rejected_not_mango_leaf"
                })

        image.seek(0)
        img_array = load_and_prep_image(image)
        prediction = model.predict(img_array, verbose=0)
        class_id = int(np.argmax(prediction))
        class_eng = model_classes[class_id]
        class_th = class_map[class_eng]
        confidence = float(prediction[0][class_id])

        if confidence < DISEASE_CONFIDENCE_THRESHOLD:
            return jsonify({
                "prediction": "ไม่พบโรคที่ตรงกับข้อมูลในระบบ",
                "confidence": confidence,
                "raw_class": class_eng,
                "accuracy": 0,
                "disease_at_threshold": DISEASE_CONFIDENCE_THRESHOLD,
                "status": "low_confidence"
            })

        response_data = {
            "prediction": class_th,
            "confidence": confidence,
            "raw_class": class_eng,
            "accuracy": 1,
            "disease_at_threshold": DISEASE_CONFIDENCE_THRESHOLD,
            "status": "success"
        }
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

# -------------------------------
# Upload Image (Admin)
# -------------------------------
@app.route("/upload", methods=["POST"])
@firebase_auth_required(admin_only=True)
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "ไม่ได้ระบุไฟล์ภาพ"}), 400
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
        return jsonify({"error": f"การอัปโหลดล้มเหลว: {str(e)}"}), 500

# -------------------------------
# Delete Image (Admin)
# -------------------------------
@app.route("/delete", methods=["POST"])
@firebase_auth_required(admin_only=True)
def delete_image():
    try:
        public_id = request.form.get('public_id') or request.json.get('public_id')
        if not public_id:
            return jsonify({"error": "ไม่ได้ระบุ public_id"}), 400
        cloudinary.uploader.destroy(public_id)
        return jsonify({"result": "ลบภาพสำเร็จ"}), 200
    except Exception as e:
        return jsonify({"error": f"การลบล้มเหลว: {str(e)}"}), 500

# -------------------------------
# Config (Admin)
# -------------------------------
@app.route('/config', methods=['GET'])
@firebase_auth_required(admin_only=True)
def get_config():
    return jsonify({
        "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
        "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
        "use_filter": USE_FILTER,
        "img_size": IMG_SIZE,
        "model_classes": model_classes,
        "has_mango_embeddings": len(checkMango.mango_embeddings) > 0,
        "model_path": LOCAL_MODEL_PATH,
        "embedding_path": LOCAL_EMBEDDING_PATH if USE_FILTER else None,
        "inactivity_timeout_minutes": INACTIVITY_TIMEOUT // 60
    })

@app.route('/config', methods=['POST'])
@firebase_auth_required(admin_only=True)
def update_config():
    global MANGO_LEAF_THRESHOLD, DISEASE_CONFIDENCE_THRESHOLD, USE_FILTER, INACTIVITY_TIMEOUT
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
        if 'inactivity_timeout_minutes' in data:
            INACTIVITY_TIMEOUT = int(data['inactivity_timeout_minutes']) * 60
        return jsonify({"message": "อัปเดตการตั้งค่าสำเร็จ"}), 200
    except Exception as e:
        return jsonify({"error": f"ไม่สามารถอัปเดตการตั้งค่าได้: {str(e)}"}), 500

# -------------------------------
# Admin: View Active Users
# -------------------------------
@app.route('/admin/active-users', methods=['GET'])
@firebase_auth_required(admin_only=True)
def get_active_users():
    try:
        cutoff_time = datetime.now() - timedelta(seconds=INACTIVITY_TIMEOUT)
        
        # Query active users
        active_query = db.collection('user_activities').where(
            'last_activity', '>=', cutoff_time
        )
        
        active_docs = active_query.stream()
        active_users_info = []
        
        for doc in active_docs:
            data = doc.to_dict()
            last_activity = data.get('last_activity')
            
            if last_activity:
                current_time = datetime.now()
                time_since_activity = (current_time - last_activity.replace(tzinfo=None)).total_seconds()
                time_remaining = INACTIVITY_TIMEOUT - time_since_activity
                
                if time_remaining > 0:
                    active_users_info.append({
                        "uid": doc.id,
                        "last_activity": last_activity.isoformat(),
                        "time_remaining_seconds": int(time_remaining),
                        "time_remaining_minutes": int(time_remaining // 60)
                    })
        
        return jsonify({
            "active_users": active_users_info,
            "total_active": len(active_users_info),
            "timeout_setting_minutes": INACTIVITY_TIMEOUT // 60
        })
        
    except Exception as e:
        logging.error(f"Error getting active users: {e}")
        return jsonify({"error": "ไม่สามารถดึงข้อมูล active users ได้"}), 500

# -------------------------------
# Health Check
# -------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # ตรวจสอบ Firebase connection
        firebase_status = "connected"
        try:
            db.collection('health_check').limit(1).get()
        except:
            firebase_status = "disconnected"
        
        return jsonify({
            "status": "healthy",
            "model_loaded": 'model' in globals() and model is not None,
            "embedding_model_loaded": hasattr(checkMango, 'embedding_model') and checkMango.embedding_model is not None,
            "mango_embeddings_loaded": len(checkMango.mango_embeddings) > 0,
            "use_filter": USE_FILTER,
            "inactivity_timeout_minutes": INACTIVITY_TIMEOUT // 60,
            "firebase_status": firebase_status
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# -------------------------------
# Run Local
# -------------------------------
if __name__ == '__main__':
    print("\n--- Starting Flask App with Firebase ---")
    print(f"Auto logout after {INACTIVITY_TIMEOUT//60} minutes of inactivity")
    print("Using Firebase Auth & Firestore for session management")
    app.run(debug=True)