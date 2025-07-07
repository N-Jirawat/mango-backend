# ใช้ Python 3.10 เป็น Base Image
FROM python:3.10-slim-buster

# กำหนด Working Directory ภายใน Container
WORKDIR /app

# คัดลอกไฟล์ requirements.txt เข้าไปใน Container
COPY requirements.txt .

# ติดตั้ง Python Dependencies ทั้งหมด
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดแอปพลิเคชันทั้งหมด (รวมถึง api/ และ checkMango.py)
COPY . .

# กำหนด Environment Variable สำหรับ GCS Bucket Name
ENV GCS_BUCKET_NAME=mango-app-models-bucket
# << ตรวจสอบชื่อ Bucket ของคุณ

# กำหนด Port ที่แอปพลิเคชันจะฟัง
# Render จะ inject PORT variable ให้เอง
ENV PORT=8000 

# ใช้ 8000 หรืออะไรก็ได้, Render จะ override ให้

# ระบุคำสั่งที่จะรันแอปพลิเคชันเมื่อ Container เริ่มต้น
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 api.index:app