import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# กำหนด path model ที่ดีที่สุด
MODEL_PATH = r"C:\Users\Asus\mango-app\backend\models\best_model.keras"
IMAGE_PATH = r"C:\Mango-Disease\val\Healthy\image_Healthy-spot_3.jpg"  # 👈 เปลี่ยน path ภาพของคุณ

# โหลดโมเดล
model = tf.keras.models.load_model(MODEL_PATH)

# คลาสที่ใช้
class_names = ['Anthracnose', 'Healthy', 'Sooty-mold', 'raised-spot']

# เตรียมภาพ
def load_and_prep_image(path, target_size=(224, 224)):
    img = Image.open(path).convert('RGB').resize(target_size)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # ✅ ใช้ preprocess_input แบบ EfficientNet
    return np.expand_dims(img_array, axis=0)

# TTA (Test Time Augmentation)
def predict_with_tta(image_path, model):
    original = load_and_prep_image(image_path)

    # Flip left-right
    flipped = tf.image.flip_left_right(original)

    # Predict ทั้ง 2 แบบ แล้วเฉลี่ย
    preds = model.predict(original)
    preds_flip = model.predict(flipped)

    final_pred = (preds + preds_flip) / 2.0
    return final_pred

# เรียกใช้
pred = predict_with_tta(IMAGE_PATH, model)

pred_class_id = np.argmax(pred)
pred_class_name = class_names[pred_class_id]
confidence = pred[0][pred_class_id]

print(f"Raw prediction: {pred}")
print(f"Predicted class id: {pred_class_id}")
print(f"Predicted class name: {pred_class_name}")
print(f"Confidence: {confidence:.4f}")
