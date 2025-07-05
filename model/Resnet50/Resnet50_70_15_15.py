import os
import re
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import seaborn as sns

# ----------------------------
# เตรียมข้อมูล
# ----------------------------
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_dir = r'C:\Mango-Disease-70_15_15\train'
val_dir = r'C:\Mango-Disease-70_15_15\val'
test_dir = r'C:\Mango-Disease-70_15_15\test'

train_raw = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode='categorical', shuffle=True
)
class_names = train_raw.class_names
train_ds = train_raw.prefetch(AUTOTUNE)

val_raw = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode='categorical'
)
val_ds = val_raw.prefetch(AUTOTUNE)

test_raw = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode='categorical'
)
test_ds = test_raw.prefetch(AUTOTUNE)

num_classes = len(class_names)

# ----------------------------
# สร้างโมเดล (ฟังก์ชัน) พร้อม EfficientNetV2S + mixed precision
# ----------------------------
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras import layers, models, mixed_precision

mixed_precision.set_global_policy('mixed_float16')

def create_model():
    base_model = ResNet50(
        include_top=False,
        input_shape=IMG_SIZE + (3,),
        weights='imagenet'
    )
    base_model.trainable = False  # เริ่มต้น freeze base model
    
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, x)
    return model, base_model

# สร้างโมเดลครั้งแรก
model, base_model = create_model()

# ----------------------------
# คำนวณ class weights
# ----------------------------
y_train = np.concatenate([y.numpy() for x, y in train_raw])
y_labels = np.argmax(y_train, axis=1)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labels),
    y=y_labels
)
class_weights = dict(enumerate(class_weights))

# ----------------------------
# Phase 1: Pretrain
# ----------------------------
checkpoint_dir1 = r'C:\Users\Asus\mango-app\backend\models\Resnet50\checkpoints15\pretrain'
os.makedirs(checkpoint_dir1, exist_ok=True)
latest_ckpt1 = tf.train.latest_checkpoint(checkpoint_dir1)

initial_epoch1 = 0
if latest_ckpt1:
    print(f"🔄 Loading Phase 1 weights from {latest_ckpt1}")
    model.load_weights(latest_ckpt1)
    match = re.search(r'ckpt-(\d+)', latest_ckpt1)
    if match:
        initial_epoch1 = int(match.group(1))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
)

checkpoint_cb1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir1, 'ckpt-{epoch:02d}'),
    save_weights_only=True,
    save_freq='epoch'
)
earlystop_cb1 = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    initial_epoch=initial_epoch1,
    class_weight=class_weights,
    callbacks=[checkpoint_cb1, earlystop_cb1]
)

model_path_phase1 = r'C:\Users\Asus\mango-app\backend\models\Resnet50\checkpoints15\model_Resnet50_70_15_15_phase1.keras'
model.save(model_path_phase1)
print(f"✅ บันทึกโมเดล Phase 1 สำเร็จที่: {model_path_phase1}")

# ----------------------------
# Phase 2: Fine-tune
# ----------------------------
# โหลดโมเดลที่บันทึกจาก Phase 1
model = tf.keras.models.load_model(model_path_phase1)

# เปิดให้ base_model trainable
# ต้องหา base_model ในโมเดลที่โหลดมาใหม่
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):  # กรณี base_model ถูกฝังเป็น Model
        base_model = layer
        break
else:
    base_model = None

if base_model is None:
    # ถ้าไม่เจอ ให้ลองวิธีอื่น หรือกำหนด trainable ทั้งโมเดล
    print("❗️ไม่พบ base_model ในโมเดลที่โหลดมา, เปิด trainable ทั้งโมเดลแทน")
    model.trainable = True
else:
    base_model.trainable = True

checkpoint_dir2 = r'C:\Users\Asus\mango-app\backend\models\Resnet50\checkpoints15\finetune'
os.makedirs(checkpoint_dir2, exist_ok=True)

latest_ckpt2 = tf.train.latest_checkpoint(checkpoint_dir2)
initial_epoch2 = 0
if latest_ckpt2:
    print(f"🔄 Loading Phase 2 weights from {latest_ckpt2}")
    model.load_weights(latest_ckpt2)
    match = re.search(r'ckpt-(\d+)', latest_ckpt2)
    if match:
        initial_epoch2 = int(match.group(1))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2)]
)

checkpoint_cb2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir2, 'ckpt-{epoch:02d}'),
    save_weights_only=True,
    save_freq='epoch'
)

best_model_path = r'C:\Users\Asus\mango-app\backend\models\Resnet50\model_Resnet50_70_15_15_Round2.keras'
best_model_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False
)

earlystop_cb2 = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    initial_epoch=initial_epoch2,
    class_weight=class_weights,
    callbacks=[checkpoint_cb2, earlystop_cb2, best_model_cb]
)

# ----------------------------
# รวมกราฟ
# ----------------------------
def plot_history(h1, h2):
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
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, combine('loss'), label='Train Loss')
    plt.plot(epochs, combine('val_loss'), label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history1, history2)

# ----------------------------
# ประเมินผล Test Set
# ----------------------------
# โหลดโมเดลที่ดีที่สุดจาก Phase 2
model = tf.keras.models.load_model(best_model_path)

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
