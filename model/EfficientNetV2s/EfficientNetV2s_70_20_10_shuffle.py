import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support, roc_curve, auc, roc_auc_score
)
from sklearn.utils import class_weight
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input

# ----------------------------
# ตั้งค่า GPU Growth
# ----------------------------
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ----------------------------
# ฟังก์ชันช่วยเหลือ: CLAHE + EfficientNet Preprocessing
# ----------------------------
def apply_clahe_np(image):
    image = np.array(image)
    img_uint8 = image.astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def apply_clahe_tf(image):
    def _clahe(image):
        image = tf.py_function(func=apply_clahe_np, inp=[image], Tout=tf.uint8)
        image.set_shape([224, 224, 3])
        return image
    image = tf.map_fn(_clahe, image, fn_output_signature=tf.uint8)
    return image

def preprocess_with_clahe(image, label):
    image = apply_clahe_tf(image)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

# ----------------------------
# เตรียมข้อมูล (Data Pipeline) - แบบสุ่มข้อมูล
# ----------------------------
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_dir = r'C:\Mango-Disease-70_20_10\train'
val_dir = r'C:\Mango-Disease-70_20_10\val'
test_dir = r'C:\Mango-Disease-70_20_10\test'

# ตั้งค่า seed สำหรับการสุ่มข้อมูลแบบสม่ำเสมอ
tf.random.set_seed(None)  # ให้ใช้ random seed ทุกครั้ง
np.random.seed(None)      # ให้ใช้ random seed ทุกครั้ง

# Load จากโฟลเดอร์ -> ได้ (image, label) เป็น uint8, label แบบ one-hot (categorical)
train_raw = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,  # เปิดการสุ่มข้อมูล
    seed=None     # ไม่ใช้ seed คงที่
)
val_raw = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,  # เปิดการสุ่มข้อมูล
    seed=None     # ไม่ใช้ seed คงที่
)
test_raw = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True  # เปิดการสุ่มข้อมูลแม้ใน test set
)

class_names = train_raw.class_names
num_classes = len(class_names)

# ไม่ใช้ cache เพื่อให้ข้อมูลถูกสุ่มทุกครั้ง
# Map CLAHE + EfficientNet preprocess → แล้ว prefetch และ shuffle เพิ่มเติม
train_ds = train_raw.map(preprocess_with_clahe, num_parallel_calls=AUTOTUNE)\
                   .shuffle(buffer_size=1000)\
                   .prefetch(AUTOTUNE)
val_ds = val_raw.map(preprocess_with_clahe, num_parallel_calls=AUTOTUNE)\
                .shuffle(buffer_size=500)\
                .prefetch(AUTOTUNE)
test_ds = test_raw.map(preprocess_with_clahe, num_parallel_calls=AUTOTUNE)\
                  .shuffle(buffer_size=500)\
                  .prefetch(AUTOTUNE)

# ----------------------------
# คำนวณ Class Weights (ใช้ข้อมูลตัวอย่างเพื่อคำนวณ)
# ----------------------------
# เก็บข้อมูลตัวอย่างสำหรับคำนวณ class weights
sample_labels = []
for _, labels in train_raw.take(10):  # ใช้ batch แรก ๆ ในการคำนวณ
    sample_labels.extend(np.argmax(labels.numpy(), axis=1))

y_labels = np.array(sample_labels)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labels),
    y=y_labels
)
class_weights = dict(enumerate(class_weights))

# ----------------------------
# สร้างโมเดล (ฟังก์ชัน) พร้อม EfficientNetV2S + mixed precision
# ----------------------------
mixed_precision.set_global_policy('mixed_float16')

def create_model():
    base_model = EfficientNetV2S(
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        weights='imagenet'
    )
    base_model.trainable = False  # freeze base model ใน Phase 1
    
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

# สร้างโมเดลครั้งแรก
model, base_model = create_model()

# ----------------------------
# Phase 1: Pretrain (Freeze Base Model) - 20 epochs
# ----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
)

print("🚀 เริ่ม Phase 1: Pretrain (Freeze Base Model) - With Data Shuffling")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    verbose=1
)

# ----------------------------
# Phase 2: Fine-tune (Unfreeze Base Model) - 30 epochs
# ----------------------------
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
)

print("🚀 เริ่ม Phase 2: Fine-tune (Unfreeze Base Model) - With Data Shuffling")
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    initial_epoch=20,
    class_weight=class_weights,
    verbose=1
)

# ----------------------------
# บันทึกโมเดลสุดท้าย
# ----------------------------
final_model_path = r'C:\Users\Asus\mango-app\backend\models\EfficienNetV2s\model_efficientnetv2s_shuffled_50epochs.keras'
model.save(final_model_path)
print(f"✅ บันทึกโมเดลสุดท้าย (แบบสุ่มข้อมูล) สำเร็จที่: {final_model_path}")

# ----------------------------
# รวมกราฟ (Plot History)
# ----------------------------
def plot_history(h1, h2, title_suffix=""):
    def safe_get(history_obj, metric):
        if history_obj is None:
            return []
        return history_obj.history.get(metric, [])

    def combine(metric):
        return safe_get(h1, metric) + safe_get(h2, metric)

    if not (combine('accuracy') or combine('val_accuracy') or combine('loss') or combine('val_loss')):
        print("⚠️ ไม่มีข้อมูลใน history สำหรับ plot กราฟ")
        return

    epochs = range(1, len(combine('loss')) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, combine('accuracy'), label='Train Accuracy')
    plt.plot(epochs, combine('val_accuracy'), label='Val Accuracy')
    plt.title(f'Model Accuracy{title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, combine('loss'), label='Train Loss')
    plt.plot(epochs, combine('val_loss'), label='Val Loss')
    plt.title(f'Model Loss{title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'training_history_shuffled.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_history(history1, history2, " (Shuffled Data)")

# ----------------------------
# ประเมินผลบน Test Set (แบบครบถ้วน)
# ----------------------------
print("\n🔍 กำลังประเมินผลบน Test Set...")
import time

# เก็บข้อมูลทั้งหมดสำหรับการวิเคราะห์
y_true = []
y_pred = []
y_pred_proba = []

# วัดเวลาการ inference
start_time = time.time()
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred_proba.extend(preds)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))
inference_time = time.time() - start_time

# แปลงเป็น numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_proba = np.array(y_pred_proba)

# ----------------------------
# 1. Accuracy, Precision, Recall, F1-Score
# ----------------------------
print("\n" + "="*60)
print("📊 DETAILED CLASSIFICATION METRICS (Shuffled Data)")
print("="*60)

# รายงานรวม
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=class_names))

# สร้างตารางสรุปสำหรับแต่ละคลาส
print("\n📋 Per-Class Performance Summary:")
print("-" * 85)
print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
print("-" * 85)

for i, class_name in enumerate(class_names):
    precision = report[class_name]['precision']
    recall = report[class_name]['recall']
    f1 = report[class_name]['f1-score']
    support = report[class_name]['support']
    print(f"{class_name:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12.0f}")

print("-" * 85)
print(f"{'Macro Avg':<20} {report['macro avg']['precision']:<12.4f} {report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f} {report['macro avg']['support']:<12.0f}")
print(f"{'Weighted Avg':<20} {report['weighted avg']['precision']:<12.4f} {report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f} {report['weighted avg']['support']:<12.0f}")

# ----------------------------
# 2. Confusion Matrix (ปรับปรุงให้สวยงาม)
# ----------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (Shuffled Data)", fontsize=16, fontweight='bold')

# เพิ่มข้อมูลเปอร์เซ็นต์
for i in range(len(class_names)):
    for j in range(len(class_names)):
        percentage = cm[i, j] / np.sum(cm[i, :]) * 100
        plt.text(j, i-0.3, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('confusion_matrix_shuffled_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# 3. ROC Curve และ AUC (Multi-class)
# ----------------------------
plt.figure(figsize=(15, 10))

# คำนวณ ROC curve สำหรับแต่ละคลาส
fpr = dict()
tpr = dict()
roc_auc = dict()

# One-hot encode y_true สำหรับ multi-class ROC
y_true_onehot = np.eye(num_classes)[y_true]

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve สำหรับแต่ละคลาส
colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
plt.subplot(2, 2, 1)
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (One-vs-Rest)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# แสดงค่า AUC เฉลี่ย
macro_auc = np.mean(list(roc_auc.values()))
plt.text(0.6, 0.2, f'Macro-Average AUC: {macro_auc:.3f}', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# AUC Bar Chart
plt.subplot(2, 2, 2)
auc_values = [roc_auc[i] for i in range(num_classes)]
bars = plt.bar(class_names, auc_values, color=colors, alpha=0.7)
plt.ylim([0, 1])
plt.ylabel('AUC Score')
plt.title('AUC Score per Class')
plt.xticks(rotation=45)
# เพิ่มค่าบน bar
for bar, auc_val in zip(bars, auc_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{auc_val:.3f}', ha='center', va='bottom')

# Precision-Recall per Class
plt.subplot(2, 2, 3)
precision_vals = [report[class_names[i]]['precision'] for i in range(num_classes)]
recall_vals = [report[class_names[i]]['recall'] for i in range(num_classes)]
f1_vals = [report[class_names[i]]['f1-score'] for i in range(num_classes)]

x = np.arange(len(class_names))
width = 0.25

plt.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
plt.bar(x, recall_vals, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8)

plt.ylabel('Score')
plt.title('Precision, Recall, F1-Score Comparison')
plt.xticks(x, class_names, rotation=45)
plt.legend()
plt.ylim([0, 1])

# Model Performance Summary
plt.subplot(2, 2, 4)
plt.axis('off')
summary_text = f"""
MODEL PERFORMANCE SUMMARY (Shuffled Data)

Overall Accuracy: {report['accuracy']:.4f}
Macro-Average F1: {report['macro avg']['f1-score']:.4f}
Weighted-Average F1: {report['weighted avg']['f1-score']:.4f}
Macro-Average AUC: {macro_auc:.4f}

Training Details:
• Total Epochs: 50 (Phase1: 20, Phase2: 30)
• Data Shuffling: Yes (Random seed)
• Architecture: EfficientNetV2S
• Input Size: 224×224×3
• Classes: {num_classes}
"""
plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
         verticalalignment='top', fontsize=11, fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

plt.tight_layout()
plt.savefig('comprehensive_analysis_shuffled.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# 4. Model Information และ Performance Metrics
# ----------------------------
# คำนวณขนาดโมเดล
model_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)
total_params = model.count_params()
trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])

print("\n" + "="*80)
print("🚀 COMPREHENSIVE MODEL ANALYSIS (Shuffled Data)")
print("="*80)

# สร้างตารางเปรียบเทียบ
print("\n📊 PERFORMANCE METRICS TABLE:")
print("-" * 100)
print(f"{'Metric':<25} {'Value':<15} {'Interpretation':<60}")
print("-" * 100)
print(f"{'Overall Accuracy':<25} {report['accuracy']:<15.4f} {'Higher is better (0-1)':<60}")
print(f"{'Macro-Avg Precision':<25} {report['macro avg']['precision']:<15.4f} {'Average precision across all classes':<60}")
print(f"{'Macro-Avg Recall':<25} {report['macro avg']['recall']:<15.4f} {'Average recall across all classes':<60}")
print(f"{'Macro-Avg F1-Score':<25} {report['macro avg']['f1-score']:<15.4f} {'Harmonic mean of precision and recall':<60}")
print(f"{'Weighted-Avg F1':<25} {report['weighted avg']['f1-score']:<15.4f} {'F1-score weighted by class support':<60}")
print(f"{'Macro-Avg AUC':<25} {macro_auc:<15.4f} {'Area Under ROC Curve (average)':<60}")

print("\n📱 DEPLOYMENT METRICS:")
print("-" * 100)
print(f"{'Model Size (MB)':<25} {model_size_mb:<15.2f} {'Storage requirement':<60}")
print(f"{'Total Parameters':<25} {total_params:<15,} {'Model complexity':<60}")
print(f"{'Trainable Parameters':<25} {trainable_params:<15,} {'Parameters updated during training':<60}")
print(f"{'Inference Time (sec)':<25} {inference_time:<15.4f} {'Time to process all test samples':<60}")
print(f"{'Avg Time per Sample':<25} {inference_time/len(y_true):<15.6f} {'Average prediction time per image':<60}")

print("\n💡 MODEL VALUE ASSESSMENT:")
print("-" * 100)
# คำนวณความคุ้มค่า
accuracy_score = report['accuracy']
f1_score = report['macro avg']['f1-score']
efficiency_score = 1 / (model_size_mb / 100)  # normalized efficiency
speed_score = 1 / (inference_time / 10)  # normalized speed

overall_value = (accuracy_score * 0.4 + f1_score * 0.3 + efficiency_score * 0.15 + speed_score * 0.15)

print(f"{'Accuracy Score':<25} {accuracy_score:<15.4f} {'40% weight in overall value':<60}")
print(f"{'F1-Score':<25} {f1_score:<15.4f} {'30% weight in overall value':<60}")
print(f"{'Efficiency Score':<25} {efficiency_score:<15.4f} {'15% weight (based on model size)':<60}")
print(f"{'Speed Score':<25} {speed_score:<15.4f} {'15% weight (based on inference time)':<60}")
print("-" * 100)
print(f"{'OVERALL VALUE SCORE':<25} {overall_value:<15.4f} {'Combined weighted score (0-1)':<60}")

# ประเมินคุณภาพ
if overall_value >= 0.8:
    quality_assessment = "🌟 EXCELLENT - Ready for production deployment"
elif overall_value >= 0.7:
    quality_assessment = "✅ GOOD - Suitable for most applications"
elif overall_value >= 0.6:
    quality_assessment = "⚠️ FAIR - May need optimization"
else:
    quality_assessment = "❌ POOR - Requires significant improvement"

print(f"{'Quality Assessment':<25} {'':<15} {quality_assessment:<60}")
print("="*80)

# ----------------------------
# สรุปผลการทดลอง
# ----------------------------
test_accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\n📊 สรุปผลการทดลอง (แบบสุ่มข้อมูล):")
print(f"   - จำนวน Epochs ทั้งหมด: 50 (Phase 1: 20, Phase 2: 30)")
print(f"   - Test Accuracy: {test_accuracy:.4f}")
print(f"   - จำนวนคลาส: {num_classes}")
print(f"   - ชื่อคลาส: {', '.join(class_names)}")
print(f"   - โมเดลถูกบันทึกที่: {final_model_path}")
print(f"   - ใช้การสุ่มข้อมูล: ใช่ (ไม่ใช้ seed คงที่)")
print(f"   - Shuffle buffer size: Train=1000, Val/Test=500")