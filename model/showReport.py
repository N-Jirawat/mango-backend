import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

# โหลด best model
model = tf.keras.models.load_model(r'C:\Users\Asus\mango-app\backend\models\model_efficientnetv2s_70_20_10.keras')

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'C:\Mango-Disease-70_20_10\test',
    image_size=(224, 224),  # ขนาดภาพตามที่ใช้เทรน EfficientNetV2S
    batch_size=32,
    shuffle=False  # เพื่อให้ลำดับตรงกับ label ที่เก็บไว้
)

# class_names (อัตโนมัติจากโฟลเดอร์ย่อย)
class_names = test_ds.class_names

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

# แสดงผลลัพธ์
print("y_true:", y_true)
print("y_pred:", y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
