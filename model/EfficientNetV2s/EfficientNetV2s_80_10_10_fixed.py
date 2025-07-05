import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time
from datetime import datetime, timedelta

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support, roc_curve, auc, roc_auc_score
)
from sklearn.utils import class_weight
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ----------------------------
# ตั้งค่า GPU Growth
# ----------------------------
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ----------------------------
# เริ่มจับเวลาการทำงานทั้งหมด
# ----------------------------
script_start_time = time.time()
training_start_datetime = datetime.now()

# ----------------------------
# สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
# ----------------------------
report_dir = r'C:\Users\Asus\OneDrive\เอกสาร\Report_Model\EfficientNetV2S\EfficientNetV2s_ 80_R10'
os.makedirs(report_dir, exist_ok=True)

# สร้างไฟล์สำหรับบันทึกผลลัพธ์
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = os.path.join(report_dir, f'training_report_{timestamp}.txt')

def log_and_print(message):
    """ฟังก์ชันสำหรับแสดงผลและบันทึกข้อความ"""
    print(message)
    with open(report_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def format_time(seconds):
    """แปลงวินาทีเป็นรูปแบบ HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def get_elapsed_time(start_time):
    """คำนวณเวลาที่ผ่านไปจากเวลาเริ่มต้น"""
    return time.time() - start_time

# เริ่มบันทึกรายงาน
log_and_print("="*80)
log_and_print("🚀 EfficientNetV2S Training Report")
log_and_print(f"📅 Start Time: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
log_and_print("="*80)

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
# เตรียมข้อมูล (Data Pipeline)
# ----------------------------
data_prep_start = time.time()

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_dir = r'C:\Mango-Disease-80_10_10\train'
val_dir = r'C:\Mango-Disease-80_10_10\val'
test_dir = r'C:\Mango-Disease-80_10_10\test'

# ตั้ง seed เพื่อให้ได้ลำดับข้อมูลเดิมตลอด
tf.random.set_seed(42)
np.random.seed(42)

log_and_print("\n📊 Loading Dataset...")

train_raw = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)
val_raw = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)
test_raw = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

class_names = train_raw.class_names
num_classes = len(class_names)

log_and_print(f"✅ Dataset loaded successfully!")
log_and_print(f"📋 Number of classes: {num_classes}")
log_and_print(f"📋 Class names: {', '.join(class_names)}")

# Cache ข้อมูลเพื่อให้ใช้ลำดับเดิมตลอด
train_raw = train_raw.cache()
val_raw = val_raw.cache()
test_raw = test_raw.cache()

# Map CLAHE + EfficientNet preprocess → แล้ว prefetch
train_ds = train_raw.map(preprocess_with_clahe, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_raw.map(preprocess_with_clahe, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_raw.map(preprocess_with_clahe, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

data_prep_time = time.time() - data_prep_start
log_and_print(f"⏱️ Data preparation time: {format_time(data_prep_time)}")

# ----------------------------
# คำนวณ Class Weights
# ----------------------------
class_weight_start = time.time()
log_and_print("\n⚖️ Calculating class weights...")

y_train = np.concatenate([y.numpy() for _, y in train_raw], axis=0)
y_labels = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labels),
    y=y_labels
)
class_weights = dict(enumerate(class_weights))

class_weight_time = time.time() - class_weight_start
log_and_print("✅ Class weights calculated:")
for i, weight in class_weights.items():
    log_and_print(f"   {class_names[i]}: {weight:.4f}")
log_and_print(f"⏱️ Class weight calculation time: {format_time(class_weight_time)}")

# ----------------------------
# สร้างโมเดล พร้อม mixed precision
# ----------------------------
model_creation_start = time.time()
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

model_creation_time = time.time() - model_creation_start
log_and_print(f"\n🏗️ Model created successfully!")
log_and_print(f"📊 Total parameters: {model.count_params():,}")
log_and_print(f"⏱️ Model creation time: {format_time(model_creation_time)}")

# ----------------------------
# Phase 1: Pretrain (Freeze Base Model) - 20 epochs
# ----------------------------
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
)

phase1_start = time.time()
log_and_print("\n🚀 Starting Phase 1: Pretrain (Freeze Base Model) - 20 epochs")
log_and_print(f"⏰ Phase 1 start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=[reduce_lr],
    verbose=1
)

phase1_time = time.time() - phase1_start
log_and_print("✅ Phase 1 completed!")
log_and_print(f"⏱️ Phase 1 training time: {format_time(phase1_time)}")
log_and_print(f"⏰ Phase 1 end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ----------------------------
# Phase 2: Fine-tune (Unfreeze Base Model) - 30 epochs
# ----------------------------
base_model.trainable = True

reduce_lr_phase2 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=8,
    min_lr=1e-8,
    verbose=1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
)

phase2_start = time.time()
log_and_print("\n🚀 Starting Phase 2: Fine-tune (Unfreeze Base Model) - 30 epochs")
log_and_print(f"⏰ Phase 2 start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    initial_epoch=20,
    class_weight=class_weights,
    callbacks=[reduce_lr_phase2],
    verbose=1
)

phase2_time = time.time() - phase2_start
total_training_time = phase1_time + phase2_time
log_and_print("✅ Phase 2 completed!")
log_and_print(f"⏱️ Phase 2 training time: {format_time(phase2_time)}")
log_and_print(f"⏰ Phase 2 end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_and_print(f"⏱️ Total training time: {format_time(total_training_time)}")

# ----------------------------
# บันทึกโมเดลสุดท้าย
# ----------------------------
model_save_start = time.time()
final_model_path = r'C:\Users\Asus\mango-app\backend\models\EfficienNetV2s\model_efficientnetv2s_80_10_10_R10.keras'
model.save(final_model_path)
model_save_time = time.time() - model_save_start

log_and_print(f"\n✅ Final model saved: {final_model_path}")
log_and_print(f"⏱️ Model save time: {format_time(model_save_time)}")

# ----------------------------
# สร้างกราฟ Training History
# ----------------------------
def plot_and_save_history(h1, h2, save_dir, time_info):
    plot_start = time.time()
    
    def safe_get(history_obj, metric):
        if history_obj is None:
            return []
        return history_obj.history.get(metric, [])

    def combine(metric):
        return safe_get(h1, metric) + safe_get(h2, metric)

    if not (combine('accuracy') or combine('val_accuracy') or combine('loss') or combine('val_loss')):
        log_and_print("⚠️ No history data available for plotting")
        return {}

    epochs = range(1, len(combine('loss')) + 1)
    
    # สร้างกราฟขนาดใหญ่
    plt.figure(figsize=(20, 12))
    
    # กราฟ Accuracy
    plt.subplot(2, 3, 1)
    plt.plot(epochs, combine('accuracy'), 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=3)
    plt.plot(epochs, combine('val_accuracy'), 'r-', label='Val Accuracy', linewidth=2, marker='s', markersize=3)
    plt.axvline(x=20, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Phase 1 → Phase 2')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # กราฟ Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs, combine('loss'), 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    plt.plot(epochs, combine('val_loss'), 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    plt.axvline(x=20, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Phase 1 → Phase 2')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # กราฟ Top-2 Accuracy
    plt.subplot(2, 3, 3)
    plt.plot(epochs, combine('top_k_categorical_accuracy'), 'b-', label='Train Top-2 Acc', linewidth=2, marker='o', markersize=3)
    plt.plot(epochs, combine('val_top_k_categorical_accuracy'), 'r-', label='Val Top-2 Acc', linewidth=2, marker='s', markersize=3)
    plt.axvline(x=20, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Phase 1 → Phase 2')
    plt.title('Top-2 Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Top-2 Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # กราฎ Learning Rate (ถ้ามี)
    plt.subplot(2, 3, 4)
    if combine('lr'):
        plt.plot(epochs, combine('lr'), 'orange', linewidth=2, marker='d', markersize=3)
        plt.axvline(x=20, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Phase 1 → Phase 2')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nHistory\nNot Available', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    # สรุปผลการฝึกพร้อมเวลา
    plt.subplot(2, 3, 5)
    final_train_acc = combine('accuracy')[-1] if combine('accuracy') else 0
    final_val_acc = combine('val_accuracy')[-1] if combine('val_accuracy') else 0
    final_train_loss = combine('loss')[-1] if combine('loss') else 0
    final_val_loss = combine('val_loss')[-1] if combine('val_loss') else 0
    final_top2_acc = combine('val_top_k_categorical_accuracy')[-1] if combine('val_top_k_categorical_accuracy') else 0
    
    summary_text = f"""Final Training Results:
    
Train Accuracy: {final_train_acc:.4f}
Val Accuracy: {final_val_acc:.4f}
Train Loss: {final_train_loss:.4f}
Val Loss: {final_val_loss:.4f}
Val Top-2 Acc: {final_top2_acc:.4f}

Training Time:
Phase 1: {format_time(time_info['phase1_time'])}
Phase 2: {format_time(time_info['phase2_time'])}
Total: {format_time(time_info['total_training_time'])}
"""
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    plt.axis('off')
    plt.title('Training Summary', fontsize=14, fontweight='bold')
    
    # แสดงค่าสูงสุดของ validation accuracy
    plt.subplot(2, 3, 6)
    if combine('val_accuracy'):
        max_val_acc = max(combine('val_accuracy'))
        max_val_acc_epoch = combine('val_accuracy').index(max_val_acc) + 1
        
        performance_text = f"""Best Performance:
        
Max Val Accuracy: {max_val_acc:.4f}
At Epoch: {max_val_acc_epoch}

Model Configuration:
• EfficientNetV2S (Pretrained)
• Image Size: 224x224
• Batch Size: 16
• Mixed Precision: FP16
• CLAHE Enhancement
• Class Weights: Balanced

Time Efficiency:
• Avg Time/Epoch: {format_time(time_info['total_training_time']/50)}
• Total Training: {format_time(time_info['total_training_time'])}
"""
        
        plt.text(0.1, 0.9, performance_text, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.axis('off')
    plt.title('Best Performance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # บันทึกกราฟ
    graph_path = os.path.join(save_dir, f'training_history_{timestamp}.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    plot_time = time.time() - plot_start
    log_and_print(f"✅ Training history graph saved: {graph_path}")
    log_and_print(f"⏱️ Plot generation time: {format_time(plot_time)}")
    
    return {
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_top2_acc': final_top2_acc,
        'max_val_acc': max_val_acc if combine('val_accuracy') else 0,
        'max_val_acc_epoch': max_val_acc_epoch if combine('val_accuracy') else 0
    }

# สร้างและบันทึกกราฟ
time_info = {
    'phase1_time': phase1_time,
    'phase2_time': phase2_time,
    'total_training_time': total_training_time
}
training_summary = plot_and_save_history(history1, history2, report_dir, time_info)

# ----------------------------
# ประเมินผลโมเดลบน Test Set
# ----------------------------
log_and_print("\n🔍 Evaluating model on Test Set...")

y_true = []
y_pred = []
y_pred_proba = []

inference_start = time.time()
for batch_idx, (images, labels) in enumerate(test_ds):
    batch_start = time.time()
    preds = model.predict(images, verbose=0)
    batch_time = time.time() - batch_start
    
    y_pred_proba.extend(preds)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    if batch_idx == 0:  # Log timing for first batch
        log_and_print(f"   First batch ({len(images)} samples) inference time: {batch_time:.4f} seconds")

total_inference_time = time.time() - inference_start

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_proba = np.array(y_pred_proba)

# สร้าง Classification Report
evaluation_start = time.time()
log_and_print("\n" + "="*80)
log_and_print("📊 TEST SET EVALUATION RESULTS")
log_and_print("="*80)

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_str = classification_report(y_true, y_pred, target_names=class_names)
log_and_print(report_str)

test_accuracy = np.mean(y_true == y_pred)
log_and_print(f"\n🏆 Test Accuracy: {test_accuracy:.4f}")
log_and_print(f"⏱️ Total Inference Time: {format_time(total_inference_time)}")
log_and_print(f"⚡ Average Time per Sample: {total_inference_time/len(y_true):.6f} seconds")
log_and_print(f"🔢 Total Test Samples: {len(y_true)}")
log_and_print(f"📊 Inference Speed: {len(y_true)/total_inference_time:.2f} samples/second")

# คำนวณ Top-2 Accuracy
top2_predictions = np.argsort(y_pred_proba, axis=1)[:, -2:]
top2_accuracy = np.mean([y_true[i] in top2_predictions[i] for i in range(len(y_true))])
log_and_print(f"🎯 Top-2 Test Accuracy: {top2_accuracy:.4f}")

evaluation_time = time.time() - evaluation_start
log_and_print(f"⏱️ Evaluation computation time: {format_time(evaluation_time)}")

# ----------------------------
# สร้าง Confusion Matrix
# ----------------------------
cm_start = time.time()
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# บันทึก Confusion Matrix
cm_path = os.path.join(report_dir, f'confusion_matrix_{timestamp}.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

cm_time = time.time() - cm_start
log_and_print(f"✅ Confusion matrix saved: {cm_path}")
log_and_print(f"⏱️ Confusion matrix generation time: {format_time(cm_time)}")

# ----------------------------
# คำนวณเวลาทั้งหมด
# ----------------------------
total_script_time = time.time() - script_start_time
script_end_datetime = datetime.now()

# ----------------------------
# บันทึกรายงานสรุปสุดท้าย
# ----------------------------
log_and_print("\n" + "="*80)
log_and_print("📋 FINAL TRAINING SUMMARY")
log_and_print("="*80)
log_and_print(f"🏗️ Model Architecture: EfficientNetV2S")
log_and_print(f"📊 Total Parameters: {model.count_params():,}")
log_and_print(f"🔢 Number of Classes: {num_classes}")
log_and_print(f"📁 Classes: {', '.join(class_names)}")
log_and_print(f"🖼️ Image Size: {IMG_SIZE}")
log_and_print(f"📦 Batch Size: {BATCH_SIZE}")
log_and_print(f"🔄 Total Epochs: 50 (Phase 1: 20, Phase 2: 30)")
log_and_print(f"🎯 Fixed Seed: 42")
log_and_print(f"⚖️ Class Weights: Balanced")
log_and_print(f"🔧 Mixed Precision: FP16")
log_and_print(f"🖼️ Image Enhancement: CLAHE")

log_and_print(f"\n⏱️ Detailed Timing Information:")
log_and_print(f"   Data Preparation: {format_time(data_prep_time)}")
log_and_print(f"   Class Weight Calculation: {format_time(class_weight_time)}")
log_and_print(f"   Model Creation: {format_time(model_creation_time)}")
log_and_print(f"   Phase 1 Training: {format_time(phase1_time)}")
log_and_print(f"   Phase 2 Training: {format_time(phase2_time)}")
log_and_print(f"   Total Training: {format_time(total_training_time)}")
log_and_print(f"   Model Saving: {format_time(model_save_time)}")
log_and_print(f"   Test Inference: {format_time(total_inference_time)}")
log_and_print(f"   Evaluation: {format_time(evaluation_time)}")
log_and_print(f"   Confusion Matrix: {format_time(cm_time)}")
log_and_print(f"   Total Script Runtime: {format_time(total_script_time)}")

log_and_print(f"\n📈 Training Results:")
log_and_print(f"   Final Train Accuracy: {training_summary.get('final_train_acc', 0):.4f}")
log_and_print(f"   Final Val Accuracy: {training_summary.get('final_val_acc', 0):.4f}")
log_and_print(f"   Final Train Loss: {training_summary.get('final_train_loss', 0):.4f}")
log_and_print(f"   Final Val Loss: {training_summary.get('final_val_loss', 0):.4f}")
log_and_print(f"   Final Val Top-2 Accuracy: {training_summary.get('final_top2_acc', 0):.4f}")
log_and_print(f"   Best Val Accuracy: {training_summary.get('max_val_acc', 0):.4f} (Epoch {training_summary.get('max_val_acc_epoch', 0)})")

log_and_print(f"\n🎯 Test Results:")
log_and_print(f"   Test Accuracy: {test_accuracy:.4f}")
log_and_print(f"   Test Top-2 Accuracy: {top2_accuracy:.4f}")
log_and_print(f"   Inference Speed: {len(y_true)/total_inference_time:.2f} samples/second")
log_and_print(f"   Avg Time per Sample: {total_inference_time/len(y_true):.6f} seconds")

log_and_print(f"\n⏰ Time Summary:")
log_and_print(f"   Start Time: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
log_and_print(f"   End Time: {script_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
log_and_print(f"   Total Duration: {format_time(total_script_time)}")
log_and_print(f"   Training Efficiency: {50/total_training_time*3600:.2f} epochs/hour")
log_and_print(f"   Average Epoch Time: {format_time(total_training_time/50)}")

log_and_print(f"\n📁 Output Files:")
log_and_print(f"   Final Model: {final_model_path}")
log_and_print(f"   Training Report: {report_file}")
log_and_print(f"   Training Graph: {os.path.join(report_dir, f'training_history_{timestamp}.png')}")
log_and_print(f"   Confusion Matrix: {cm_path}")

log_and_print(f"\n📅 End Time: {script_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
log_and_print("="*80)
log_and_print("✅ Training completed successfully!")
log_and_print(f"🎉 Total execution time: {format_time(total_script_time)}")
log_and_print("="*80)

# ----------------------------
# สร้างรายงานสรุปเวลาแยกเป็นไฟล์
# ----------------------------
time_report_file = os.path.join(report_dir, f'time_analysis_{timestamp}.txt')
with open(time_report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("⏱️ DETAILED TIME ANALYSIS REPORT\n")
    f.write("="*80 + "\n")
    f.write(f"Training Start: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Training End: {script_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Duration: {format_time(total_script_time)}\n\n")
    
    f.write("📊 Time Breakdown:\n")
    f.write(f"├── Data Preparation: {format_time(data_prep_time)} ({data_prep_time/total_script_time*100:.1f}%)\n")
    f.write(f"├── Class Weight Calculation: {format_time(class_weight_time)} ({class_weight_time/total_script_time*100:.1f}%)\n")
    f.write(f"├── Model Creation: {format_time(model_creation_time)} ({model_creation_time/total_script_time*100:.1f}%)\n")
    f.write(f"├── Training (Total): {format_time(total_training_time)} ({total_training_time/total_script_time*100:.1f}%)\n")
    f.write(f"│   ├── Phase 1 (20 epochs): {format_time(phase1_time)} ({phase1_time/total_training_time*100:.1f}% of training)\n")
    f.write(f"│   └── Phase 2 (30 epochs): {format_time(phase2_time)} ({phase2_time/total_training_time*100:.1f}% of training)\n")
    f.write(f"├── Model Saving: {format_time(model_save_time)} ({model_save_time/total_script_time*100:.1f}%)\n")
    f.write(f"├── Test Inference: {format_time(total_inference_time)} ({total_inference_time/total_script_time*100:.1f}%)\n")
    f.write(f"├── Evaluation: {format_time(evaluation_time)} ({evaluation_time/total_script_time*100:.1f}%)\n")
    f.write(f"└── Visualization: {format_time(cm_time)} ({cm_time/total_script_time*100:.1f}%)\n\n")
    
    f.write("🏃 Performance Metrics:\n")
    f.write(f"• Training Speed: {50/total_training_time*3600:.2f} epochs/hour\n")
    f.write(f"• Average Epoch Time: {format_time(total_training_time/50)}\n")
    f.write(f"• Phase 1 Avg Epoch: {format_time(phase1_time/20)}\n")
    f.write(f"• Phase 2 Avg Epoch: {format_time(phase2_time/30)}\n")
    f.write(f"• Inference Speed: {len(y_true)/total_inference_time:.2f} samples/second\n")
    f.write(f"• Time per Sample: {total_inference_time/len(y_true)*1000:.2f} ms\n\n")
    
    f.write("🎯 Efficiency Analysis:\n")
    f.write(f"• Training Efficiency: {(phase1_time + phase2_time)/total_script_time*100:.1f}% of total time\n")
    f.write(f"• Setup Overhead: {(data_prep_time + class_weight_time + model_creation_time)/total_script_time*100:.1f}% of total time\n")
    f.write(f"• Evaluation Overhead: {(total_inference_time + evaluation_time + cm_time)/total_script_time*100:.1f}% of total time\n")
    
    if phase2_time > 0 and phase1_time > 0:
        phase_ratio = phase2_time / phase1_time
        f.write(f"• Phase 2 vs Phase 1 Time Ratio: {phase_ratio:.2f}x\n")
        f.write(f"• Time per Epoch Comparison: Phase 2 takes {(phase2_time/30)/(phase1_time/20):.2f}x longer per epoch\n")

log_and_print(f"\n📊 Detailed time analysis saved: {time_report_file}")

# ----------------------------
# บันทึกข้อมูล JSON สำหรับการวิเคราะห์เพิ่มเติม
# ----------------------------
import json

timing_data = {
    "experiment_info": {
        "model": "EfficientNetV2S",
        "total_epochs": 50,
        "phase1_epochs": 20,
        "phase2_epochs": 30,
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "num_classes": num_classes,
        "total_parameters": int(model.count_params()),
        "start_datetime": training_start_datetime.isoformat(),
        "end_datetime": script_end_datetime.isoformat()
    },
    "timing_breakdown": {
        "data_preparation_seconds": float(data_prep_time),
        "class_weight_calculation_seconds": float(class_weight_time),
        "model_creation_seconds": float(model_creation_time),
        "phase1_training_seconds": float(phase1_time),
        "phase2_training_seconds": float(phase2_time),
        "total_training_seconds": float(total_training_time),
        "model_saving_seconds": float(model_save_time),
        "inference_seconds": float(total_inference_time),
        "evaluation_seconds": float(evaluation_time),
        "visualization_seconds": float(cm_time),
        "total_script_seconds": float(total_script_time)
    },
    "performance_metrics": {
        "epochs_per_hour": float(50/total_training_time*3600),
        "average_epoch_seconds": float(total_training_time/50),
        "phase1_avg_epoch_seconds": float(phase1_time/20),
        "phase2_avg_epoch_seconds": float(phase2_time/30),
        "inference_samples_per_second": float(len(y_true)/total_inference_time),
        "milliseconds_per_sample": float(total_inference_time/len(y_true)*1000),
        "test_samples": len(y_true)
    },
    "training_results": {
        "final_train_accuracy": float(training_summary.get('final_train_acc', 0)),
        "final_val_accuracy": float(training_summary.get('final_val_acc', 0)),
        "best_val_accuracy": float(training_summary.get('max_val_acc', 0)),
        "best_val_accuracy_epoch": int(training_summary.get('max_val_acc_epoch', 0)),
        "test_accuracy": float(test_accuracy),
        "test_top2_accuracy": float(top2_accuracy)
    }
}

json_report_file = os.path.join(report_dir, f'timing_data_{timestamp}.json')
with open(json_report_file, 'w', encoding='utf-8') as f:
    json.dump(timing_data, f, indent=2, ensure_ascii=False)

log_and_print(f"📈 Timing data (JSON) saved: {json_report_file}")
log_and_print("\n🎊 All reports and analyses completed successfully!")