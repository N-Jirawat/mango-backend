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
from google.cloud import storage # เพิ่มการ import สำหรับ Google Cloud Storage

# สร้าง Flask App
app = Flask(__name__)

# -------------------------------
# CORS config
# -------------------------------
# แทนที่ 'https://YOUR_FRONTEND_RENDER_URL' ด้วย URL จริงของ Frontend ของคุณบน Render.com
# ตัวอย่าง: 'https://my-mango-frontend.onrender.com'
# หากมีหลาย URL ให้ใส่ในลิสต์ เช่น ['https://url1.onrender.com', 'https://url2.onrender.com']
# สำหรับการทดสอบเท่านั้น: หากต้องการอนุญาตทุกโดเมน (ไม่แนะนำสำหรับ Production) ให้ใช้ origins="*"
CORS(app, origins="https://mangoleafanalyzer.onrender.com")

# หรือสำหรับการทดสอบทุกโดเมน (ไม่แนะนำสำหรับ Production):
# CORS(app, origins="*")

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
# แนะนำให้เก็บ API keys ใน Environment Variables
# สำหรับ Local Development คุณสามารถตั้งค่า Environment Variables ใน Terminal
# หรือใช้ไฟล์ .env และไลบรารี python-dotenv เพื่อความสะดวก
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME', 'dsf25dlca'),
    api_key=os.environ.get('CLOUDINARY_API_KEY', '978124749794588'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET', 's_KmqxdLxYeW8H-dCbLkWFx_ZTQ'),
)

# -------------------------------
# กำหนดค่าสำหรับ Google Cloud Storage (GCS)
# -------------------------------
# เปลี่ยน 'your-mango-app-models-bucket' เป็นชื่อ Bucket GCS ของคุณที่สร้างไว้
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'mango-app-models-bucket') # ควรตั้งเป็น Environment Variable ด้วย
EMBEDDINGS_GCS_PATH = "mango_reference_embeddings.npy"
MODEL_GCS_PATH = "model_efficientnetv2s_224_R1.keras"

# กำหนด Path ที่จะเก็บไฟล์ชั่วคราวใน App Engine (หรือในเครื่อง)
# /tmp/ เป็นโฟลเดอร์ที่เขียนได้ใน App Engine Standard Environment
LOCAL_MODEL_DIR = "/tmp/models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model_efficientnetv2s_224_R1.keras")
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
# โหลดโมเดลหลักและ Embedding (จะถูกโหลดเพียงครั้งเดียวตอน Cold Start)
# -------------------------------
# สร้างโฟลเดอร์สำหรับเก็บโมเดลชั่วคราวถ้ายังไม่มี
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ดาวน์โหลดโมเดลหลักจาก GCS และโหลด
print(f"กำลังดาวน์โหลดโมเดลหลักจาก GCS: {MODEL_GCS_PATH}")
try:
    download_from_gcs(GCS_BUCKET_NAME, MODEL_GCS_PATH, LOCAL_MODEL_PATH)
    is_valid_model, model_message = verify_file_exists_and_not_empty(LOCAL_MODEL_PATH)
    if not is_valid_model:
        raise RuntimeError(f"ไฟล์โมเดลหลักไม่ถูกต้องหลังดาวน์โหลด: {model_message}")
    
    print("กำลังโหลดโมเดลหลัก...")
    model = load_model(LOCAL_MODEL_PATH)
    print(f"✅ โหลดโมเดลหลักสำเร็จจาก {LOCAL_MODEL_PATH}")
    print(f"   รูปร่างอินพุตของโมเดล: {model.input_shape}")
    print(f"   รูปร่างเอาต์พุตของโมเดล: {model.output_shape}")
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

    # ดาวน์โหลด reference embeddings จาก GCS และโหลด
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
    checkMango.mango_embeddings = np.array([]) # ตรวจสอบให้แน่ใจว่าเป็น array เสมอแม้จะว่างเปล่า

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

    image_file.seek(0, 2) # Move to end to get size
    file_size = image_file.tell()
    image_file.seek(0) # Reset to beginning for reading

    if file_size > 10 * 1024 * 1024: # 10 MB limit
        raise ValueError("ขนาดไฟล์ใหญ่เกินไป ขนาดสูงสุดคือ 10MB")

# -------------------------------
# API Routes
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "ไม่ได้ระบุไฟล์ภาพ"}), 400
        
        image = request.files['image']
        validate_image_file(image)

        # ตรวจสอบว่าเป็นใบมะม่วงหรือไม่ (ถ้า USE_FILTER = True)
        similarity = 0.0
        if USE_FILTER and hasattr(checkMango, 'mango_embeddings') and len(checkMango.mango_embeddings) > 0:
            try:
                image.seek(0) # Reset file pointer before passing to checkMango
                is_leaf, similarity = checkMango.is_mango_leaf_from_embedding(image, checkMango.mango_embeddings)
                if similarity < MANGO_LEAF_THRESHOLD:
                    return jsonify({
                        "prediction": "ไม่ใช่ภาพใบมะม่วง",
                        "confidence": float(similarity), # Use similarity as confidence for rejection
                        "raw_class": None,
                        "accuracy": 0,
                        "mango_leaf_confidence": float(similarity),
                        "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
                        "status": "rejected_not_mango_leaf"
                    })
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการตรวจจับใบมะม่วง: {e}")
                # ควรพิจารณาว่าจะ return error หรือทำนายต่อไปโดยไม่กรอง
                # ในที่นี้เลือกที่จะยังคงค่า similarity เป็น 0.0 และทำนายต่อไป
                # หรือจะ raise e อีกครั้งเพื่อหยุดการทำงานหากการกรองสำคัญมาก
                similarity = 0.0 

        # ทำนายโรค
        image.seek(0) # Reset file pointer again before loading for prediction
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
                "disease_at_threshold": DISEASE_CONFIDENCE_THRESHOLD, # Changed from disease_threshold to disease_at_threshold
                "status": "low_confidence"
            })

        # ส่งผลลัพธ์
        response_data = {
            "prediction": class_th,
            "confidence": confidence,
            "raw_class": class_eng,
            "accuracy": 1,
            "disease_at_threshold": DISEASE_CONFIDENCE_THRESHOLD, # Changed from disease_threshold to disease_at_threshold
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
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
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

@app.route("/delete", methods=["POST"])
def delete_image():
    try:
        public_id = request.form.get('public_id') or request.json.get('public_id')
        if not public_id:
            return jsonify({"error": "ไม่ได้ระบุ public_id"}), 400

        cloudinary.uploader.destroy(public_id)
        return jsonify({"result": "ลบภาพสำเร็จ"}), 200
    except Exception as e:
        return jsonify({"error": f"การลบล้มเหลว: {str(e)}"}), 500

@app.route('/config', methods=['GET'])
def get_config():
    return jsonify({
        "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
        "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
        "use_filter": USE_FILTER,
        "img_size": IMG_SIZE,
        "model_classes": model_classes,
        "has_mango_embeddings": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "model_path": LOCAL_MODEL_PATH, # เปลี่ยนเป็น Local Path ที่ดาวน์โหลดมา
        "embedding_path": LOCAL_EMBEDDING_PATH if USE_FILTER else None # เปลี่ยนเป็น Local Path ที่ดาวน์โหลดมา
    })

@app.route('/config', methods=['POST'])
def update_config():
    global MANGO_LEAF_THRESHOLD, DISEASE_CONFIDENCE_THRESHOLD, USE_FILTER
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

        return jsonify({"message": "อัปเดตการตั้งค่าสำเร็จ"}), 200
    except Exception as e:
        return jsonify({"error": f"ไม่สามารถอัปเดตการตั้งค่าได้: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """ตรวจสอบสถานะของระบบ"""
    return jsonify({
        "status": "healthy",
        "model_loaded": 'model' in globals() and model is not None,
        "embedding_model_loaded": hasattr(checkMango, 'embedding_model') and checkMango.embedding_model is not None,
        "mango_embeddings_loaded": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "use_filter": USE_FILTER
    })

# -------------------------------
# สำหรับการรันใน Local Development
# -------------------------------
if __name__ == '__main__':
    # สำหรับการรันใน Local Development:
    # 1. ตรวจสอบให้แน่ใจว่าได้ติดตั้ง google-cloud-storage แล้ว (pip install google-cloud-storage)
    # 2. ตั้งค่า Environment Variable 'GCS_BUCKET_NAME' ในเครื่องของคุณ
    #    (เช่น ใน PowerShell: $env:GCS_BUCKET_NAME="your-mango-app-models-bucket")
    # 3. ตรวจสอบว่าคุณได้ตั้งค่า Google Cloud Authentication สำหรับ Local Development แล้ว
    #    (เช่น gcloud auth application-default login)
    # 4. ไฟล์โมเดลจะถูกดาวน์โหลดจาก GCS ไปยัง /tmp/models/ ในเครื่องของคุณ
    #    (หรือ C:\Users\Asus\AppData\Local\Temp\models บน Windows)
    # 5. ถ้าคุณต้องการรันโดยไม่ดาวน์โหลดจาก GCS ใน Local
    #    คุณสามารถคอมเมนต์ส่วนดาวน์โหลด GCS ออกชั่วคราว
    #    และตรวจสอบให้แน่ใจว่าไฟล์โมเดลอยู่ใน api/models/ ในเครื่องของคุณ
    #    และเปลี่ยน LOCAL_MODEL_PATH/LOCAL_EMBEDDING_PATH กลับไปชี้ที่ api/models/
    #    (แต่แนะนำให้ทดสอบการดาวน์โหลดจาก GCS ใน Local ด้วย)

    print("\n--- กำลังเริ่ม Flask App ในโหมด Local Development ---")
    print("เข้าถึง API ได้ที่ http://127.0.0.1:5000/")
    print("กด Ctrl+C เพื่อออก.")
    app.run(debug=True)
