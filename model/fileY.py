import os
import shutil
import random

# กำหนด path
original_dataset_dir = r'C:\Mango-Disease-All'
output_base_dir = r'D:\Download\Mango-Disease-80_10_10'

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# ตรวจสอบว่า ratio รวมกันได้ 1
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

# สร้างโฟลเดอร์ output
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_base_dir, split), exist_ok=True)

# วนลูปแต่ละคลาส
for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # สร้างโฟลเดอร์ class ในแต่ละ split
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base_dir, split, class_name), exist_ok=True)

    # สุ่มและแบ่งภาพ
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # คัดลอกภาพไปยังโฟลเดอร์ใหม่
    for split, split_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        print(f"Copying {len(split_files)} files to {split}/{class_name}...")
        for filename in split_files:
            src = os.path.join(class_path, filename)
            dst = os.path.join(output_base_dir, split, class_name, filename)
            shutil.copy2(src, dst)