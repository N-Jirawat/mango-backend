# ใช้ Python 3.10 เป็น Base Image ที่มีขนาดเล็ก
FROM python:3.10-slim-buster

# กำหนด Working Directory ภายใน Container
WORKDIR /app

# อัปเดต pip และติดตั้ง Dependencies ที่จำเป็นสำหรับการ Build
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# คัดลอกไฟล์ requirements.txt เข้าไปใน Container
COPY requirements.txt .

# ติดตั้ง Python Dependencies ทั้งหมด
RUN pip install --no-cache-dir -r requirements.txt

# *** เพิ่มบรรทัดนี้ ***
# กำหนด PATH เพื่อให้แน่ใจว่า executable ของ pip (เช่น gunicorn) ถูกพบ
ENV PATH="/usr/local/bin:$PATH"

# คัดลอกโค้ดแอปพลิเคชันทั้งหมด
COPY . .

# กำหนด Environment Variable สำหรับ GCS Bucket Name
ENV GCS_BUCKET_NAME=mango-app-models-bucket
# << ตรวจสอบชื่อ Bucket

# ระบุคำสั่งที่จะรันแอปพลิเคชันเมื่อ Container เริ่มต้น
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 api.index:app