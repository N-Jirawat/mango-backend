FROM python:3.10-slim-buster

WORKDIR /app

# อัปเดต pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# คัดลอก requirements
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# กำหนด PATH ให้เจอ executable
ENV PATH="/usr/local/bin:$PATH"

# คัดลอก source code
COPY . .

# Environment Variables
ENV GCS_BUCKET_NAME=mango-app-models-bucket

# ใช้ Gunicorn รัน Flask app
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 api.index:app


