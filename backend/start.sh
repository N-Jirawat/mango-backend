#!/bin/bash

echo "🚀 Starting Mango Disease Detection API"
exec gunicorn backend.main:app --bind 0.0.0.0:$PORT

chmod +x backend/start.sh
