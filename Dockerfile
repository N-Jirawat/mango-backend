# ใช้ Python 3.10 เป็น Base Image ที่มีขนาดเล็ก
FROM python:3.10-slim-buster

# กำหนด Working Directory ภายใน Container
WORKDIR /app

# อัปเดต pip และติดตั้ง Dependencies ที่จำเป็นสำหรับการ Build
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# คัดลอกไฟล์ requirements.txt เข้าไปใน Container
# เพื่อให้ Layer Cache ทำงานได้ดี
COPY requirements.txt .

# ติดตั้ง Python Dependencies ทั้งหมด
# ตรวจสอบให้แน่ใจว่า gunicorn ถูกติดตั้ง
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดแอปพลิเคชันทั้งหมด (รวมถึง api/ และ checkMango.py)
COPY . .

# กำหนด Environment Variable สำหรับ GCS Bucket Name
ENV GCS_BUCKET_NAME=mango-app-models-465207-bucket 
# << ตรวจสอบชื่อ Bucket ของคุณอีกครั้ง

# กำหนด Port ที่แอปพลิเคชันจะฟัง (Render จะ inject PORT variable ให้เอง)
# ENV PORT=8000 # ไม่จำเป็นต้องตั้งค่าใน Dockerfile, Render จะกำหนดให้เอง

# ระบุคำสั่งที่จะรันแอปพลิเคชันเมื่อ Container เริ่มต้น
# ใช้ exec เพื่อให้ gunicorn เป็น PID 1 และรับสัญญาณจาก Docker
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 api.index:app