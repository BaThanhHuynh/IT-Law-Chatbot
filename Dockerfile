# ==========================================
# Giai đoạn 1: Build dependencies
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Cài đặt công cụ build cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tạo môi trường ảo
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

# Copy requirements file
COPY requirements.txt .

# Cài đặt toàn bộ dependencies với PyTorch CPU index
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt




# ==========================================
# Giai đoạn 2: Runner image sạch
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Sao chép môi trường ảo chứa các package đã build từ stage 1
COPY --from=builder /opt/venv /opt/venv

# Thiết lập đường dẫn môi trường ảo
ENV PATH="/opt/venv/bin:$PATH"

# Sao chép mã nguồn ứng dụng (tuân thủ .dockerignore)
COPY . .

# Mở cổng API
EXPOSE 5000

# Các biến môi trường mặc định
ENV PYTHONUNBUFFERED=1
ENV API_PORT=5000
ENV EMBEDDING_MODEL=/app/models/law_v2_model_20260505_1418

# Lệnh khởi chạy ứng dụng
CMD ["python", "-m", "app.main"]

