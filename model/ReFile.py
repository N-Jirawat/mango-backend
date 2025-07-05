import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

input_folder = r'D:\Download\Sooty-mold'  # หรือแฟลชไดรฟ์ เช่น r'G:\Photos'
output_format = 'jpg'
output_folder = os.path.join(input_folder, f"_converted_to_{output_format}")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpeg', '.jpg', '.webp', '.bmp', '.tiff', '.heic')):
        filepath = os.path.join(input_folder, filename)
        try:
            with Image.open(filepath) as img:
                base_filename = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_filename}.{output_format}")

                rgb_image = img.convert('RGB')
                save_format = 'JPEG' if output_format.lower() == 'jpg' else output_format.upper()
                rgb_image.save(output_path, save_format)
                print(f"✅ แปลงแล้ว: {filename}")
        except Exception as e:
            print(f"❌ ข้ามไฟล์: {filename} (error: {e})")

print(f"\n🎉 เสร็จสิ้น! ไฟล์ใหม่ถูกเก็บไว้ใน: {output_folder}")
