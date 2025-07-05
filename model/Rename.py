import os
import shutil

# กำหนดพาธของโฟลเดอร์ต้นทางและปลายทาง
source_folder = r'D:\Download\Healthy'
destination_folder = r'D:\Download\Healthy2'

# ตรวจสอบว่าโฟลเดอร์ปลายทางมีอยู่หรือไม่ ถ้าไม่มีให้สร้าง
os.makedirs(destination_folder, exist_ok=True)

# ตรวจสอบว่ามีไฟล์ในโฟลเดอร์หรือไม่
if os.path.exists(source_folder):
    for idx, filename in enumerate(os.listdir(source_folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.heic')):
            old_file_path = os.path.join(source_folder, filename)
            new_filename = f"image_Healthy-spot_{idx+1}{os.path.splitext(filename)[1]}"
            new_file_path = os.path.join(destination_folder, new_filename)
            
            # ย้ายไฟล์ไปยังโฟลเดอร์ปลายทางหลังจากเปลี่ยนชื่อ
            shutil.move(old_file_path, new_file_path)  # หรือใช้ shutil.copy() ถ้าต้องการคัดลอกแทน
            print(f'เปลี่ยนชื่อไฟล์: {filename} -> {new_filename}')
else:
    print("ไม่พบโฟลเดอร์ที่กำหนด")
