import shutil

# ย้ายไฟล์ไปยังโฟลเดอร์ใหม่
# shutil.move(r"D:\EfficientNetV2s\model_efficientnetv2s_80_10_10.keras", r"C:\Users\Asus\mango-app\backend\models\model_efficientnetv2s_80_10_10.keras")

# shutil.move(r"C:\Users\Asus\mango-app\backend\models\model_efficientnetv2s_80_10_10.keras", r"D:\EfficientNetV2s\model_efficientnetv2s_80_10_10.keras")

# shutil.copy(r"D:\EfficientNetV2s\model_efficientnetv2s_80_10_10.keras", r"C:\Users\Asus\mango-app\backend\models\model_efficientnetv2s_80_10_10_InSystem.keras")

shutil.copy(r"C:\Users\Asus\mango-app\backend\mango_reference_embeddings.npy", r"D:\EfficientNetV2s\mango_reference_embeddings_OutSystem.npy")
