# =================================================================
# Flask API สำหรับการวิเคราะห์โรคในใบมะม่วง (เวอร์ชันปรับปรุงแล้ว)
# ใช้ Machine Learning (EfficientNetV2S) ในการตรวจจับและจำแนกโรค
# =================================================================

# =================== การ Import Libraries ===================
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth, firestore
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import cloudinary
import cloudinary.uploader
import os, json, time, signal
import checkMango
from google.cloud import storage
from datetime import datetime
from contextlib import contextmanager
import traceback
import psutil  # สำหรับตรวจสอบ system resources

# =================== การตั้งค่า Flask Application ===================
app = Flask(__name__)

# =================== การตั้งค่า CORS ===================
# กำหนดให้รองรับหลาย origins และเพิ่มความยืดหยุ่น
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://mangoleafanalyzer.onrender.com",
            "http://localhost:3000",  # สำหรับ development
            "http://127.0.0.1:3000"   # สำหรับ local testing
        ],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# =================== การตั้งค่าพื้นฐานของระบบ ===================
IMG_SIZE = (224, 224)
USE_FILTER = True

# ค่า Threshold สำหรับการตัดสินใจ
MANGO_LEAF_THRESHOLD = 0.70
DISEASE_CONFIDENCE_THRESHOLD = 0.80

# Timeout settings
MODEL_LOAD_TIMEOUT = 300  # 5 minutes
PREDICTION_TIMEOUT = 30   # 30 seconds

# =================== การจำแนกโรคและการแปลภาษา ===================
model_classes = ['Anthracnose', 'Healthy', 'Sooty-mold', 'raised-spot']
class_map = {
    'Anthracnose': 'โรคแอนแทรคโนส',
    'Healthy': 'ใบปกติ',
    'Sooty-mold': 'โรคราดำ',
    'raised-spot': 'โรคใบจุดนูน',
}

# =================== การตั้งค่า Cloudinary ===================
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
)

# =================== การตั้งค่า Google Cloud Storage ===================
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'mango-app-models-bucket')
EMBEDDINGS_GCS_PATH = "mango_reference_embeddings.npy"
MODEL_GCS_PATH = "model_efficientnetv2s_224_R3.keras"

LOCAL_MODEL_DIR = "/tmp/models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model_efficientnetv2s_224_R3.keras")
LOCAL_EMBEDDING_PATH = os.path.join(LOCAL_MODEL_DIR, "mango_reference_embeddings.npy")

# =================== Global Variables สำหรับจัดการสถานะ ===================
model = None
model_load_error = None
embedding_load_error = None
app_start_time = datetime.now()

# =================== Utility Functions ===================

@contextmanager
def timeout(duration):
    """Context manager สำหรับจำกัดเวลาการทำงาน"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    # ใช้ได้เฉพาะใน Unix-like systems
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration)
    
    try:
        yield
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

def get_system_info():
    """ดึงข้อมูลระบบสำหรับ monitoring"""
    try:
        return {
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "disk_usage": psutil.disk_usage('/').percent if os.path.exists('/') else None
        }
    except:
        return {"memory_percent": "N/A", "cpu_percent": "N/A", "disk_usage": "N/A"}

def download_from_gcs(bucket_name, source_blob_name, destination_file_name, max_retries=3):
    """ดาวน์โหลด Blob จาก GCS พร้อม retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"🔄 ความพยายามที่ {attempt + 1}/{max_retries}: ดาวน์โหลด {source_blob_name}")
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            
            # ตรวจสอบว่าไฟล์มีอยู่ใน GCS หรือไม่
            if not blob.exists():
                raise Exception(f"ไฟล์ {source_blob_name} ไม่มีอยู่ใน GCS bucket {bucket_name}")
            
            blob.download_to_filename(destination_file_name)
            print(f"✅ ดาวน์โหลด '{source_blob_name}' สำเร็จ")
            return True
        except Exception as e:
            print(f"❌ ความพยายามที่ {attempt + 1} ล้มเหลว: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    return False

def verify_file_exists_and_not_empty(file_path, min_size=1024):
    """ตรวจสอบว่าไฟล์มีอยู่และมีขนาดเหมาะสม"""
    if not os.path.exists(file_path):
        return False, f"ไฟล์ไม่มีอยู่: {file_path}"
    
    size = os.path.getsize(file_path)
    if size == 0:
        return False, f"ไฟล์ว่างเปล่า: {file_path}"
    if size < min_size:
        return False, f"ไฟล์เล็กเกินไป ({size} bytes, ต้องการอย่างน้อย {min_size} bytes)"
    
    return True, f"ไฟล์ถูกต้อง ({size:,} bytes)"

def load_model_safely():
    """โหลดโมเดลอย่างปลอดภัยพร้อม error handling"""
    global model, model_load_error
    
    try:
        print("🚀 เริ่มโหลดโมเดลหลัก...")
        
        # สร้างโฟลเดอร์
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        
        # ดาวน์โหลดโมเดลถ้ายังไม่มี
        if not os.path.exists(LOCAL_MODEL_PATH):
            with timeout(MODEL_LOAD_TIMEOUT):
                download_from_gcs(GCS_BUCKET_NAME, MODEL_GCS_PATH, LOCAL_MODEL_PATH)
        
        # ตรวจสอบไฟล์
        is_valid, message = verify_file_exists_and_not_empty(LOCAL_MODEL_PATH, min_size=1024*1024)  # อย่างน้อย 1MB
        if not is_valid:
            raise RuntimeError(f"ไฟล์โมเดลไม่ถูกต้อง: {message}")
        
        # โหลดโมเดล
        print("📥 กำลังโหลดโมเดลเข้าสู่หน่วยความจำ...")
        with timeout(MODEL_LOAD_TIMEOUT):
            model = load_model(LOCAL_MODEL_PATH, compile=False)  # ไม่ compile เพื่อเร็วขึ้น
        
        # ทดสอบโมเดล
        print("🧪 ทดสอบโมเดล...")
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        print(f"✅ โหลดโมเดลสำเร็จ")
        print(f"   📊 Input shape: {model.input_shape}")
        print(f"   📊 Output shape: {model.output_shape}")
        return True
        
    except TimeoutError as e:
        error_msg = f"การโหลดโมเดล timeout: {e}"
        print(f"⏰ {error_msg}")
        model_load_error = error_msg
        return False
    except Exception as e:
        error_msg = f"ไม่สามารถโหลดโมเดลได้: {str(e)}"
        print(f"❌ {error_msg}")
        model_load_error = error_msg
        traceback.print_exc()
        return False

def load_embedding_safely():
    """โหลด embedding model และ reference data อย่างปลอดภัย"""
    global embedding_load_error
    
    if not USE_FILTER:
        print("🔄 การกรองใบมะม่วงถูกปิดใช้งาน")
        checkMango.mango_embeddings = np.array([])
        return True
    
    try:
        print("🚀 เริ่มโหลด embedding model และข้อมูลอ้างอิง...")
        
        # โหลด EfficientNetV2S
        checkMango.embedding_model = EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg")
        print("✅ โหลด EfficientNetV2S สำเร็จ")
        
        # ดาวน์โหลดและโหลด embeddings
        if not os.path.exists(LOCAL_EMBEDDING_PATH):
            with timeout(MODEL_LOAD_TIMEOUT):
                download_from_gcs(GCS_BUCKET_NAME, EMBEDDINGS_GCS_PATH, LOCAL_EMBEDDING_PATH)
        
        is_valid, message = verify_file_exists_and_not_empty(LOCAL_EMBEDDING_PATH, min_size=1024)
        if not is_valid:
            raise RuntimeError(f"ไฟล์ embedding ไม่ถูกต้อง: {message}")
        
        checkMango.mango_embeddings = np.load(LOCAL_EMBEDDING_PATH)
        print(f"✅ โหลด embeddings สำเร็จ: {checkMango.mango_embeddings.shape}")
        return True
        
    except Exception as e:
        error_msg = f"ไม่สามารถโหลด embedding ได้: {str(e)}"
        print(f"❌ {error_msg}")
        embedding_load_error = error_msg
        traceback.print_exc()
        
        # ตั้งค่า fallback
        checkMango.mango_embeddings = np.array([])
        return False

# =================== การโหลดโมเดลเมื่อเริ่มแอป ===================
print("\n" + "="*60)
print("🌟 เริ่มต้นระบบวิเคราะห์โรคใบมะม่วง")
print("="*60)

# โหลดโมเดลหลัก
model_loaded = load_model_safely()

# โหลด embedding model
embedding_loaded = load_embedding_safely()

print("\n" + "="*60)
if model_loaded:
    print("🎉 ระบบพร้อมใช้งาน!")
else:
    print("⚠️  ระบบเริ่มทำงานแล้วแต่โมเดลยังไม่พร้อม")
print("="*60 + "\n")

# =================== Helper Functions ===================

def load_and_prep_image(image_file):
    """เตรียมภาพสำหรับการประมวลผลโดยโมเดล AI"""
    try:
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {e}")

def validate_image_file(image_file):
    """ตรวจสอบความถูกต้องของไฟล์ภาพ"""
    if not image_file:
        raise ValueError("ไม่ได้ระบุไฟล์ภาพ")

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    filename = image_file.filename.lower() if image_file.filename else ""
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise ValueError("รูปแบบภาพไม่ถูกต้อง รูปแบบที่รองรับ: PNG, JPG, JPEG, GIF, BMP, WEBP")

    # ตรวจสอบขนาดไฟล์
    image_file.seek(0, 2)
    file_size = image_file.tell()
    image_file.seek(0)

    if file_size > 10 * 1024 * 1024:  # 10 MB
        raise ValueError("ขนาดไฟล์ใหญ่เกินไป ขนาดสูงสุดคือ 10MB")
    
    if file_size < 1024:  # 1 KB
        raise ValueError("ขนาดไฟล์เล็กเกินไป")

# =================== Error Handlers ===================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "กรุณาตรวจสอบ URL ที่เรียกใช้",
        "available_endpoints": ["/predict", "/upload", "/delete", "/health", "/config"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์",
        "details": str(error) if app.debug else "กรุณาลองใหม่อีกครั้ง"
    }), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "error": "File too large",
        "message": "ไฟล์ใหญ่เกินไป ขนาดสูงสุด 10MB"
    }), 413

# =================== API Endpoints ===================

@app.route('/health', methods=['GET'])
def health_check():
    """API สำหรับตรวจสอบสถานะของระบบ - ปรับปรุงแล้ว"""
    system_info = get_system_info()
    uptime = datetime.now() - app_start_time
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(uptime.total_seconds()),
        "uptime_human": str(uptime).split('.')[0],
        "system": system_info,
        "models": {
            "main_model": {
                "loaded": model is not None,
                "error": model_load_error,
                "path": LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else None
            },
            "embedding_model": {
                "loaded": hasattr(checkMango, 'embedding_model') and checkMango.embedding_model is not None,
                "embeddings_loaded": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
                "error": embedding_load_error,
                "use_filter": USE_FILTER
            }
        },
        "config": {
            "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
            "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
            "img_size": IMG_SIZE
        }
    }
    
    # ทดสอบโมเดลหลัก
    if model is not None:
        try:
            with timeout(10):  # 10 วินาที timeout สำหรับการทดสอบ
                dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                prediction = model.predict(dummy_input, verbose=0)
                health_status["models"]["main_model"]["test_prediction_shape"] = list(prediction.shape)
                health_status["models"]["main_model"]["test_status"] = "passed"
        except Exception as e:
            health_status["models"]["main_model"]["test_status"] = f"failed: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["status"] = "degraded"
    
    # กำหนด HTTP status code
    if health_status["status"] == "healthy":
        status_code = 200
    else:
        status_code = 503  # Service Unavailable
    
    return jsonify(health_status), status_code

@app.route('/predict', methods=['POST'])
def predict_image():
    """API หลักสำหรับทำนายโรคในใบมะม่วง - ปรับปรุงแล้ว"""
    try:
        # ตรวจสอบว่าโมเดลพร้อมใช้งาน
        if model is None:
            return jsonify({
                "error": "Model not available",
                "message": "โมเดลยังไม่พร้อมใช้งาน กรุณาลองใหม่อีกครั้ง",
                "model_error": model_load_error,
                "status": "model_not_ready"
            }), 503

        # ตรวจสอบไฟล์
        if 'image' not in request.files:
            return jsonify({"error": "ไม่ได้ระบุไฟล์ภาพ"}), 400
        
        image = request.files['image']
        validate_image_file(image)

        # ตรวจสอบใบมะม่วง
        similarity = 0.0
        if USE_FILTER and hasattr(checkMango, 'mango_embeddings') and len(checkMango.mango_embeddings) > 0:
            try:
                image.seek(0)
                with timeout(PREDICTION_TIMEOUT):
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
            except TimeoutError:
                return jsonify({
                    "error": "Mango detection timeout",
                    "message": "การตรวจสอบใบมะม่วงใช้เวลานานเกินไป"
                }), 504
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการตรวจจับใบมะม่วง: {e}")
                similarity = 0.0  # ใช้ค่าเริ่มต้น

        # ทำนายโรค
        image.seek(0)
        try:
            with timeout(PREDICTION_TIMEOUT):
                img_array = load_and_prep_image(image)
                prediction = model.predict(img_array, verbose=0)
        except TimeoutError:
            return jsonify({
                "error": "Prediction timeout",
                "message": "การทำนายใช้เวลานานเกินไป"
            }), 504

        # ประมวลผลลัพธ์
        class_id = int(np.argmax(prediction))
        class_eng = model_classes[class_id]
        class_th = class_map[class_eng]
        confidence = float(prediction[0][class_id])

        # ตรวจสอบความมั่นใจ
        if confidence < DISEASE_CONFIDENCE_THRESHOLD:
            return jsonify({
                "prediction": "ไม่พบโรคที่ตรงกับข้อมูลในระบบ",
                "confidence": confidence,
                "raw_class": class_eng,
                "accuracy": 0,
                "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
                "all_predictions": {
                    model_classes[i]: float(prediction[0][i]) 
                    for i in range(len(model_classes))
                },
                "status": "low_confidence"
            })

        # ส่งผลลัพธ์สำเร็จ
        response_data = {
            "prediction": class_th,
            "confidence": confidence,
            "raw_class": class_eng,
            "accuracy": 1,
            "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
            "all_predictions": {
                model_classes[i]: float(prediction[0][i]) 
                for i in range(len(model_classes))
            },
            "status": "success"
        }

        if USE_FILTER and hasattr(checkMango, 'mango_embeddings') and len(checkMango.mango_embeddings) > 0:
            response_data["mango_leaf_confidence"] = float(similarity)
            response_data["mango_leaf_threshold"] = MANGO_LEAF_THRESHOLD

        return jsonify(response_data)

    except ValueError as e:
        return jsonify({"error": str(e), "type": "validation_error"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "message": f"เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์: {str(e)}",
            "type": "server_error"
        }), 500

@app.route("/upload", methods=["POST"])
def upload_image():
    """API สำหรับอัปโหลดภาพไปยัง Cloudinary - ปรับปรุงแล้ว"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "ไม่ได้ระบุไฟล์ภาพ"}), 400

        image = request.files['image']
        validate_image_file(image)

        # อัปโหลดไปยัง Cloudinary พร้อม timeout
        with timeout(60):  # 1 นาที timeout
            upload_result = cloudinary.uploader.upload(
                image, 
                folder="mango_diseases",
                resource_type="image",
                timeout=60
            )
        
        return jsonify({
            "imageUrl": upload_result['secure_url'],
            "public_id": upload_result['public_id'],
            "upload_timestamp": datetime.now().isoformat()
        })
        
    except TimeoutError:
        return jsonify({
            "error": "Upload timeout",
            "message": "การอัปโหลดใช้เวลานานเกินไป"
        }), 504
    except ValueError as e:
        return jsonify({"error": str(e), "type": "validation_error"}), 400
    except Exception as e:
        return jsonify({
            "error": "Upload failed",
            "message": f"การอัปโหลดล้มเหลว: {str(e)}"
        }), 500

@app.route("/delete", methods=["POST"])
def delete_image():
    """API สำหรับลบภาพจาก Cloudinary - ปรับปรุงแล้ว"""
    try:
        public_id = request.form.get('public_id') or request.json.get('public_id')
        if not public_id:
            return jsonify({"error": "ไม่ได้ระบุ public_id"}), 400

        with timeout(30):  # 30 วินาที timeout
            result = cloudinary.uploader.destroy(public_id)
        
        return jsonify({
            "result": "ลบภาพสำเร็จ",
            "cloudinary_result": result,
            "public_id": public_id
        }), 200
        
    except TimeoutError:
        return jsonify({
            "error": "Delete timeout",
            "message": "การลบใช้เวลานานเกินไป"
        }), 504
    except Exception as e:
        return jsonify({
            "error": "Delete failed", 
            "message": f"การลบล้มเหลว: {str(e)}"
        }), 500

@app.route('/config', methods=['GET'])
def get_config():
    """API สำหรับดูการตั้งค่าปัจจุบันของระบบ"""
    return jsonify({
        "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
        "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
        "use_filter": USE_FILTER,
        "img_size": IMG_SIZE,
        "model_classes": model_classes,
        "class_map": class_map,
        "timeouts": {
            "model_load": MODEL_LOAD_TIMEOUT,
            "prediction": PREDICTION_TIMEOUT
        },
        "paths": {
            "model_path": LOCAL_MODEL_PATH,
            "embedding_path": LOCAL_EMBEDDING_PATH if USE_FILTER else None
        },
        "status": {
            "model_loaded": model is not None,
            "embeddings_loaded": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False
        }
    })

@app.route('/config', methods=['POST'])
def update_config():
    """API สำหรับอัปเดตการตั้งค่าระบบ - ปรับปรุงแล้ว"""
    global MANGO_LEAF_THRESHOLD, DISEASE_CONFIDENCE_THRESHOLD, USE_FILTER
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "ไม่ได้ระบุข้อมูลการตั้งค่า"}), 400

        changes_made = []
        
        if 'mango_leaf_threshold' in data:
            new_threshold = float(data['mango_leaf_threshold'])
            if 0.0 <= new_threshold <= 1.0:
                old_value = MANGO_LEAF_THRESHOLD
                MANGO_LEAF_THRESHOLD = new_threshold
                changes_made.append(f"mango_leaf_threshold: {old_value} → {new_threshold}")
            else:
                return jsonify({"error": "mango_leaf_threshold ต้องอยู่ระหว่าง 0.0-1.0"}), 400
                
        if 'disease_confidence_threshold' in data:
            new_threshold = float(data['disease_confidence_threshold'])
            if 0.0 <= new_threshold <= 1.0:
                old_value = DISEASE_CONFIDENCE_THRESHOLD
                DISEASE_CONFIDENCE_THRESHOLD = new_threshold
                changes_made.append(f"disease_confidence_threshold: {old_value} → {new_threshold}")
            else:
                return jsonify({"error": "disease_confidence_threshold ต้องอยู่ระหว่าง 0.0-1.0"}), 400
                
        if 'use_filter' in data:
            new_filter = bool(data['use_filter'])
            old_value = USE_FILTER
            USE_FILTER = new_filter
            changes_made.append(f"use_filter: {old_value} → {new_filter}")

        return jsonify({
            "message": "อัปเดตการตั้งค่าสำเร็จ",
            "changes": changes_made,
            "timestamp": datetime.now().isoformat(),
            "current_config": {
                "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
                "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
                "use_filter": USE_FILTER
            }
        }), 200
        
    except ValueError as e:
        return jsonify({
            "error": "Invalid value", 
            "message": f"ค่าที่ระบุไม่ถูกต้อง: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Configuration update failed",
            "message": f"ไม่สามารถอัปเดตการตั้งค่าได้: {str(e)}"
        }), 500

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """API สำหรับโหลดโมเดลใหม่ (สำหรับกรณีโมเดลเสีย)"""
    global model, model_load_error
    
    try:
        print("🔄 กำลังโหลดโมเดลใหม่...")
        model = None  # Clear existing model
        model_load_error = None
        
        success = load_model_safely()
        
        if success:
            return jsonify({
                "message": "โหลดโมเดลใหม่สำเร็จ",
                "timestamp": datetime.now().isoformat(),
                "model_status": "ready"
            }), 200
        else:
            return jsonify({
                "error": "Failed to reload model",
                "message": "ไม่สามารถโหลดโมเดลใหม่ได้",
                "model_error": model_load_error
            }), 500
            
    except Exception as e:
        return jsonify({
            "error": "Reload failed",
            "message": f"เกิดข้อผิดพลาดในการโหลดโมเดลใหม่: {str(e)}"
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """API สำหรับดูสถานะโดยรวมของระบบ (แบบย่อ)"""
    uptime = datetime.now() - app_start_time
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "uptime": str(uptime).split('.')[0],
        "ready": model is not None and (not USE_FILTER or len(checkMango.mango_embeddings) > 0),
        "models": {
            "main": model is not None,
            "embedding": not USE_FILTER or (hasattr(checkMango, 'embedding_model') and checkMango.embedding_model is not None)
        }
    }
    
    return jsonify(status)

@app.route('/', methods=['GET'])
def root():
    """Root endpoint สำหรับแสดงข้อมูลพื้นฐาน"""
    return jsonify({
        "service": "Mango Leaf Disease Detection API",
        "version": "2.0",
        "description": "API สำหรับการวิเคราะห์โรคในใบมะม่วงด้วย AI",
        "status": "running",
        "endpoints": {
            "POST /predict": "ทำนายโรคจากภาพใบมะม่วง",
            "POST /upload": "อัปโหลดภาพไปยัง Cloudinary",
            "POST /delete": "ลบภาพจาก Cloudinary",
            "GET /health": "ตรวจสอบสถานะระบบแบบละเอียด",
            "GET /status": "ตรวจสอบสถานะระบบแบบย่อ",
            "GET /config": "ดูการตั้งค่าปัจจุบัน",
            "POST /config": "อัปเดตการตั้งค่า",
            "POST /reload-model": "โหลดโมเดลใหม่"
        },
        "model_ready": model is not None,
        "uptime": str(datetime.now() - app_start_time).split('.')[0]
    })

# =================== เพิ่มการจัดการ Request ขนาดใหญ่ ===================
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

# =================== เพิ่ม Request Logging (สำหรับ Debug) ===================
@app.before_request
def log_request_info():
    if app.debug:
        print(f"📝 {request.method} {request.path} - {request.remote_addr}")

@app.after_request
def log_response_info(response):
    if app.debug:
        print(f"📤 {response.status_code} - {request.method} {request.path}")
    
    # เพิ่ม Security Headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response

# =================== การรันแอปพลิเคชัน ===================
if __name__ == '__main__':
    print("\n🚀 กำลังเริ่ม Flask App ในโหมด Development")
    print("📍 API endpoints:")
    print("   - Health check: http://127.0.0.1:5000/health")
    print("   - Status: http://127.0.0.1:5000/status")
    print("   - Predict: http://127.0.0.1:5000/predict")
    print("   - Config: http://127.0.0.1:5000/config")
    print("\n⚡ กด Ctrl+C เพื่อหยุดการทำงาน\n")
    
    # ตั้งค่าสำหรับ development
    app.run(
        host='0.0.0.0',  # รับ connection จากทุก IP
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development',
        threaded=True  # รองรับ multiple requests
    )