FROM python:3.10-slim

WORKDIR /app

# Cài đặt công cụ hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    API_PORT=5000 \
    EMBEDDING_MODEL=/app/models/law_v2_model_20260505_1418 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5

# Nâng cấp pip
RUN pip install --no-cache-dir --upgrade pip

# Cài đặt PyTorch CPU trực tiếp từ repo chính thức
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements và cài đặt các dependencies còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng (tuân thủ .dockerignore)
COPY . .

# Mở cổng API
EXPOSE 5000

# Lệnh khởi chạy ứng dụng
CMD ["python", "-m", "app.main"]
