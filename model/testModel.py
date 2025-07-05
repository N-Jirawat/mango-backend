import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ------------ Config ------------
MODEL_PATH = r'C:\Users\Asus\mango-app\backend\models\best_model.keras'
IMAGE_SIZE = (224, 224)
TEST_DIR = r'C:\Mango-Disease\test'  # test/Anthracnose, test/Healthy, ...
CLASS_NAMES = ['Anthracnose', 'Healthy', 'raised-spot', 'Sooty-mold']
print(CLASS_NAMES)
TTA_TIMES = 5
# --------------------------------

def preprocess_lab_clahe(img):
    """แปลงภาพเป็น LAB และใช้ CLAHE"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def load_images_from_folder(folder):
    """โหลดภาพและ label ทั้งหมดในโฟลเดอร์"""
    images = []
    labels = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(folder, class_name)
        if not os.path.exists(class_folder):
            print(f"❌ Not found: {class_folder}")
            continue
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)
            img = preprocess_lab_clahe(img)
            images.append(img / 255.0)
            labels.append(label_index)
    return np.array(images), np.array(labels)

def test_time_augmentation_predict(model, image, tta_times=5):
    """ทำ TTA โดยการ flip/brightness/contrast"""
    preds = []
    for _ in range(tta_times):
        aug_img = tf.image.random_flip_left_right(image)
        aug_img = tf.image.random_flip_up_down(aug_img)
        aug_img = tf.image.random_brightness(aug_img, max_delta=0.1)
        aug_img = tf.image.random_contrast(aug_img, 0.9, 1.1)
        pred = model.predict(tf.expand_dims(aug_img, axis=0), verbose=0)
        preds.append(pred[0])
    return np.mean(preds, axis=0)

# โหลดโมเดล
model = load_model(MODEL_PATH, compile=False)

# โหลดภาพทดสอบทั้งหมด
print("📥 Loading test images...")
X_test, y_test = load_images_from_folder(TEST_DIR)
print(f"✅ Loaded {len(X_test)} images")

# ทำนายผลทั้งหมดด้วย TTA
print("🔍 Predicting...")
y_pred = []
for i in range(len(X_test)):
    pred_probs = test_time_augmentation_predict(model, X_test[i], TTA_TIMES)
    pred_label = np.argmax(pred_probs)
    y_pred.append(pred_label)

# ตัวอย่างการทำนายภาพเดียว
img = image.load_img(r"C:\Users\Asus\mango-app\public\img\image_sooty_888.jpg", target_size=IMAGE_SIZE)
x = image.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)
pred = model.predict(x)
print("📌 Predict class:", CLASS_NAMES[np.argmax(pred)])

# รายงานผล
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues")

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
