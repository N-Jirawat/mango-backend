# =================================================================
# Flask API สำหรับการวิเคราะห์โรคในใบมะม่วง
# ใช้ Machine Learning (EfficientNetV2S) ในการตรวจจับและจำแนกโรค
# =================================================================

# =================== การ Import Libraries ===================
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth, firestore
from flask_cors import CORS  # สำหรับจัดการ Cross-Origin Resource Sharing
from PIL import Image  # สำหรับการประมวลผลภาพ
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import cloudinary  # สำหรับการอัปโหลดภาพไปยัง Cloud
import cloudinary.uploader
import os, json
import checkMango  # โมดูลสำหรับตรวจสอบว่าเป็นใบมะม่วงหรือไม่
from google.cloud import storage  # สำหรับดาวน์โหลดโมเดลจาก Google Cloud Storage
from flask import Flask, request, jsonify
from flask_cors import CORS

# =================== การตั้งค่า Flask Application ===================
app = Flask(__name__)

# =================== การตั้งค่า CORS ===================
# กำหนดว่า API สามารถรับ Request จากโดเมนไหนได้บ้าง
CORS(app, resources={r"/*": {"origins": "https://mangoleafanalyzer.onrender.com"}})
# หรือสำหรับการทดสอบทุกโดเมน (ไม่แนะนำสำหรับ Production):
#CORS(app, origins="*")

# =================== การตั้งค่าพื้นฐานของระบบ ===================
IMG_SIZE = (224, 224)  # ขนาดภาพที่โมเดลต้องการ
USE_FILTER = True  # เปิด/ปิดการตรวจสอบว่าเป็นใบมะม่วงก่อนทำนาย

# ค่า Threshold สำหรับการตัดสินใจ
MANGO_LEAF_THRESHOLD = 0.70  # ค่าความมั่นใจขั้นต่ำที่จะถือว่าเป็นใบมะม่วง
DISEASE_CONFIDENCE_THRESHOLD = 0.80  # ค่าความมั่นใจขั้นต่ำในการทำนายโรค

# =================== การจำแนกโรคและการแปลภาษา ===================
model_classes = ['Anthracnose', 'Healthy', 'Sooty-mold', 'raised-spot']  # คลาสที่โมเดลสามารถทำนายได้
class_map = {
    'Anthracnose': 'โรคแอนแทรคโนส',
    'Healthy': 'ใบปกติ',
    'Sooty-mold': 'โรคราดำ',
    'raised-spot': 'โรคใบจุดนูน',
}

# =================== การตั้งค่า Cloudinary ===================
# Cloudinary ใช้สำหรับเก็บภาพในคลาวด์
# ควรเก็บ API keys ใน Environment Variables เพื่อความปลอดภัย
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
)

# =================== การตั้งค่า Google Cloud Storage ===================
# GCS ใช้สำหรับเก็บโมเดล AI และข้อมูล Reference
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'mango-app-models-bucket')
EMBEDDINGS_GCS_PATH = "mango_reference_embeddings.npy"  # ไฟล์ข้อมูลอ้างอิงใบมะม่วง
MODEL_GCS_PATH = "model_efficientnetv2s_224_R3.keras"  # ไฟล์โมเดล AI

# กำหนดตำแหน่งเก็บไฟล์ชั่วคราวใน Server
# /tmp/ เป็นโฟลเดอร์ที่เขียนได้ใน App Engine Standard Environment
LOCAL_MODEL_DIR = "/tmp/models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model_efficientnetv2s_224_R3.keras")
LOCAL_EMBEDDING_PATH = os.path.join(LOCAL_MODEL_DIR, "mango_reference_embeddings.npy")

# =================== ฟังก์ชันสำหรับดาวน์โหลดไฟล์จาก Google Cloud Storage ===================
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """ดาวน์โหลด Blob จาก GCS Bucket ไปยังไฟล์ในเครื่อง"""
    try:
        storage_client = storage.Client()  # สร้าง Client สำหรับเชื่อมต่อ GCS
        bucket = storage_client.bucket(bucket_name)  # เลือก Bucket
        blob = bucket.blob(source_blob_name)  # เลือกไฟล์ใน Bucket
        blob.download_to_filename(destination_file_name)  # ดาวน์โหลดไฟล์
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

# =================== การโหลดโมเดล AI และข้อมูลอ้างอิง ===================
# การโหลดจะทำงานเพียงครั้งเดียวตอนเริ่มต้น Server (Cold Start)

# สร้างโฟลเดอร์สำหรับเก็บโมเดลชั่วคราวถ้ายังไม่มี
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ========== โหลดโมเดลหลักสำหรับทำนายโรค ==========
print(f"กำลังดาวน์โหลดโมเดลหลักจาก GCS: {MODEL_GCS_PATH}")
try:
    # ดาวน์โหลดโมเดลจาก Google Cloud Storage
    download_from_gcs(GCS_BUCKET_NAME, MODEL_GCS_PATH, LOCAL_MODEL_PATH)
    
    # ตรวจสอบว่าไฟล์ดาวน์โหลดสมบูรณ์
    is_valid_model, model_message = verify_file_exists_and_not_empty(LOCAL_MODEL_PATH)
    if not is_valid_model:
        raise RuntimeError(f"ไฟล์โมเดลหลักไม่ถูกต้องหลังดาวน์โหลด: {model_message}")
    
    # โหลดโมเดลเข้าสู่หน่วยความจำ
    print("กำลังโหลดโมเดลหลัก...")
    model = load_model(LOCAL_MODEL_PATH)
    print(f"✅ โหลดโมเดลหลักสำเร็จจาก {LOCAL_MODEL_PATH}")
    print(f"   รูปร่างอินพุตของโมเดล: {model.input_shape}")
    print(f"   รูปร่างเอาต์พุตของโมเดล: {model.output_shape}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดลหลัก: {e}")
    raise RuntimeError(f"ไม่สามารถโหลดโมเดลหลักจาก GCS ได้: {e}")

# ========== โหลดโมเดลและข้อมูลสำหรับตรวจสอบใบมะม่วง ==========
if USE_FILTER:  # ถ้าเปิดใช้งานการกรองใบมะม่วง
    try:
        # โหลดโมเดล EfficientNetV2S สำหรับสกัด Feature จากภาพ
        checkMango.embedding_model = EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg")
        print("✅ โหลด EfficientNetV2S embedding model สำเร็จ")
    except Exception as e:
        print(f"❌ ไม่สามารถโหลด embedding model ได้: {e}")
        raise RuntimeError(f"ไม่สามารถโหลด embedding model ได้: {e}")

    # ดาวน์โหลดและโหลดข้อมูลอ้างอิงใบมะม่วง
    print(f"กำลังดาวน์โหลดไฟล์ Embedding จาก GCS: {EMBEDDINGS_GCS_PATH}")
    try:
        download_from_gcs(GCS_BUCKET_NAME, EMBEDDINGS_GCS_PATH, LOCAL_EMBEDDING_PATH)
        
        # ตรวจสอบไฟล์
        is_valid_embedding, embedding_message = verify_file_exists_and_not_empty(LOCAL_EMBEDDING_PATH)
        if not is_valid_embedding:
            raise RuntimeError(f"ไฟล์ Embedding ไม่ถูกต้องหลังดาวน์โหลด: {embedding_message}")
        
        # โหลดข้อมูล Embedding ของใบมะม่วงอ้างอิง
        checkMango.mango_embeddings = np.load(LOCAL_EMBEDDING_PATH)
        print(f"✅ โหลด {LOCAL_EMBEDDING_PATH} สำเร็จด้วยรูปร่าง {checkMango.mango_embeddings.shape}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลดไฟล์ Embedding: {e}")
        raise RuntimeError(f"ไม่สามารถโหลด mango embeddings จาก {EMBEDDINGS_GCS_PATH} ได้: {e}")
else:
    # ถ้าปิดการกรองใบมะม่วง ให้สร้าง array ว่างเปล่า
    print("🔄 การกรองใบมะม่วงถูกปิดใช้งาน (USE_FILTER = False)")
    checkMango.mango_embeddings = np.array([])

# =================== ฟังก์ชันช่วยเหลือ ===================

def load_and_prep_image(image_file):
    """เตรียมภาพสำหรับการประมวลผลโดยโมเดล AI"""
    try:
        image_file.seek(0)  # รีเซ็ตตำแหน่งไฟล์ไปที่จุดเริ่มต้น
        img = Image.open(image_file).convert("RGB").resize(IMG_SIZE)  # เปิดไฟล์, แปลงเป็น RGB และปรับขนาด
        arr = np.array(img)  # แปลงเป็น NumPy array
        arr = preprocess_input(arr)  # ปรับค่าพิกเซลตามที่โมเดล EfficientNet ต้องการ
        return np.expand_dims(arr, axis=0)  # เพิ่มมิติ batch (จาก 3D เป็น 4D)
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {e}")

def validate_image_file(image_file):
    """ตรวจสอบความถูกต้องของไฟล์ภาพ"""
    if not image_file:
        raise ValueError("ไม่ได้ระบุไฟล์ภาพ")

    # ตรวจสอบนามสกุลไฟล์
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    filename = image_file.filename.lower() if image_file.filename else ""
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise ValueError("รูปแบบภาพไม่ถูกต้อง รูปแบบที่รองรับ: PNG, JPG, JPEG, GIF, BMP, WEBP")

    # ตรวจสอบขนาดไฟล์
    image_file.seek(0, 2)  # ไปที่ท้ายไฟล์เพื่อหาขนาด
    file_size = image_file.tell()
    image_file.seek(0)  # กลับไปที่จุดเริ่มต้น

    if file_size > 10 * 1024 * 1024:  # จำกัดที่ 10 MB
        raise ValueError("ขนาดไฟล์ใหญ่เกินไป ขนาดสูงสุดคือ 10MB")

# =================== API Endpoints ===================

@app.route('/predict', methods=['POST'])
def predict_image():
    """API หลักสำหรับทำนายโรคในใบมะม่วง"""
    try:
        # ========== ตรวจสอบไฟล์ที่ส่งมา ==========
        if 'image' not in request.files:
            return jsonify({"error": "ไม่ได้ระบุไฟล์ภาพ"}), 400
        
        image = request.files['image']
        validate_image_file(image)  # ตรวจสอบความถูกต้องของไฟล์

        # ========== ตรวจสอบว่าเป็นใบมะม่วงหรือไม่ ==========
        similarity = 0.0
        if USE_FILTER and hasattr(checkMango, 'mango_embeddings') and len(checkMango.mango_embeddings) > 0:
            try:
                image.seek(0)  # รีเซ็ตตำแหน่งไฟล์
                # เรียกใช้ฟังก์ชันตรวจสอบใบมะม่วง
                is_leaf, similarity = checkMango.is_mango_leaf_from_embedding(image, checkMango.mango_embeddings)
                
                # ถ้าความมั่นใจต่ำกว่า threshold ให้ปฏิเสธ
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
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการตรวจจับใบมะม่วง: {e}")
                # กรณีเกิดข้อผิดพลาดในการกรอง ยังคงทำนายต่อไป
                similarity = 0.0 

        # ========== ทำนายโรคในใบมะม่วง ==========
        image.seek(0)  # รีเซ็ตตำแหน่งไฟล์อีกครั้ง
        img_array = load_and_prep_image(image)  # เตรียมภาพ
        prediction = model.predict(img_array, verbose=0)  # ทำนายด้วยโมเดล
        
        # หาคลาสที่มีความน่าจะเป็นสูงสุด
        class_id = int(np.argmax(prediction))
        class_eng = model_classes[class_id]  # ชื่อคลาสภาษาอังกฤษ
        class_th = class_map[class_eng]  # ชื่อคลาสภาษาไทย
        confidence = float(prediction[0][class_id])  # ค่าความมั่นใจ

        # ========== ตรวจสอบความมั่นใจในการทำนายโรค ==========
        if confidence < DISEASE_CONFIDENCE_THRESHOLD:
            return jsonify({
                "prediction": "ไม่พบโรคที่ตรงกับข้อมูลในระบบ",
                "confidence": confidence,
                "raw_class": class_eng,
                "accuracy": 0,
                "disease_at_threshold": DISEASE_CONFIDENCE_THRESHOLD,
                "status": "low_confidence"
            })

        # ========== ส่งผลลัพธ์สำเร็จ ==========
        response_data = {
            "prediction": class_th,  # ชื่อโรคภาษาไทย
            "confidence": confidence,  # ค่าความมั่นใจ
            "raw_class": class_eng,  # ชื่อคลาสภาษาอังกฤษ
            "accuracy": 1,  # แสดงว่าทำนายสำเร็จ
            "disease_at_threshold": DISEASE_CONFIDENCE_THRESHOLD,
            "status": "success"
        }

        # เพิ่มข้อมูล mango leaf confidence ถ้ามีการใช้ filter
        if USE_FILTER and hasattr(checkMango, 'mango_embeddings') and len(checkMango.mango_embeddings) > 0:
            response_data["mango_leaf_confidence"] = float(similarity)
            response_data["mango_leaf_threshold"] = MANGO_LEAF_THRESHOLD

        return jsonify(response_data)

    except ValueError as e:
        # ข้อผิดพลาดจากการตรวจสอบไฟล์
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # ข้อผิดพลาดอื่นๆ
        import traceback
        traceback.print_exc()  # แสดงข้อผิดพลาดใน Console สำหรับ Debug
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
def upload_image():
    """API สำหรับอัปโหลดภาพไปยัง Cloudinary"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "ไม่ได้ระบุไฟล์ภาพ"}), 400

        image = request.files['image']
        validate_image_file(image)  # ตรวจสอบไฟล์

        # อัปโหลดไปยัง Cloudinary
        upload_result = cloudinary.uploader.upload(image, folder="mango_diseases")
        return jsonify({
            "imageUrl": upload_result['secure_url'],  # URL ของภาพ
            "public_id": upload_result['public_id']   # ID สำหรับลบภาพ
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"การอัปโหลดล้มเหลว: {str(e)}"}), 500

@app.route("/delete", methods=["POST"])
def delete_image():
    """API สำหรับลบภาพจาก Cloudinary"""
    try:
        # รับ public_id จาก form data หรือ JSON
        public_id = request.form.get('public_id') or request.json.get('public_id')
        if not public_id:
            return jsonify({"error": "ไม่ได้ระบุ public_id"}), 400

        # ลบภาพจาก Cloudinary
        cloudinary.uploader.destroy(public_id)
        return jsonify({"result": "ลบภาพสำเร็จ"}), 200
    except Exception as e:
        return jsonify({"error": f"การลบล้มเหลว: {str(e)}"}), 500    

@app.route('/config', methods=['GET'])
def get_config():
    """API สำหรับดูการตั้งค่าปัจจุบันของระบบ"""
    return jsonify({
        "mango_leaf_threshold": MANGO_LEAF_THRESHOLD,
        "disease_confidence_threshold": DISEASE_CONFIDENCE_THRESHOLD,
        "use_filter": USE_FILTER,
        "img_size": IMG_SIZE,
        "model_classes": model_classes,
        "has_mango_embeddings": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "model_path": LOCAL_MODEL_PATH,
        "embedding_path": LOCAL_EMBEDDING_PATH if USE_FILTER else None
    })

@app.route('/config', methods=['POST'])
def update_config():
    """API สำหรับอัปเดตการตั้งค่าระบบ"""
    global MANGO_LEAF_THRESHOLD, DISEASE_CONFIDENCE_THRESHOLD, USE_FILTER
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "ไม่ได้ระบุข้อมูลการตั้งค่า"}), 400

        # อัปเดตค่าตั้งค่าตามที่ส่งมา
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
    """API สำหรับตรวจสอบสถานะของระบบ"""
    return jsonify({
        "status": "healthy",
        "model_loaded": 'model' in globals() and model is not None,
        "embedding_model_loaded": hasattr(checkMango, 'embedding_model') and checkMango.embedding_model is not None,
        "mango_embeddings_loaded": len(checkMango.mango_embeddings) > 0 if hasattr(checkMango, 'mango_embeddings') else False,
        "use_filter": USE_FILTER
    })

# =================== การรันแอปพลิเคชัน ===================
if __name__ == '__main__':
    # สำหรับการรันในโหมด Local Development
    print("\n--- กำลังเริ่ม Flask App ในโหมด Local Development ---")
    print("เข้าถึง API ได้ที่ http://127.0.0.1:5000/")
    print("กด Ctrl+C เพื่อออก.")
    app.run(debug=True)