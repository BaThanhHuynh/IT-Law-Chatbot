FROM python:3.10-slim

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements
COPY requirements.txt .

# Nâng cấp pip và cài đặt Torch phiên bản CPU trước để ổn định và nhẹ (chỉ ~150MB thay vì 500MB+)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các dependencies còn lại
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Mở cổng API
EXPOSE 5000

# Các biến môi trường mặc định
ENV PYTHONUNBUFFERED=1
ENV API_PORT=5000
ENV EMBEDDING_MODEL=/app/models/embedding_model

# Lệnh khởi chạy ứng dụng
CMD ["python", "-m", "app.main"]
