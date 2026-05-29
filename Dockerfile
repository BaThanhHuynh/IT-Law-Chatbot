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
ENV PATH="/opt/venv/bin:$PATH"

# Nâng cấp pip và cài đặt PyTorch phiên bản CPU
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các dependencies theo nhóm để tránh lỗi I/O của WSL2 khi giải nén các wheel lớn
RUN pip install --no-cache-dir "numpy<2"
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-dotenv google-genai slowapi
RUN pip install --no-cache-dir qdrant-client neo4j langchain-neo4j langchain-text-splitters
RUN pip install --no-cache-dir openpyxl python-docx tqdm
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu sentence-transformers


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
ENV EMBEDDING_MODEL=/app/models/embedding_model

# Lệnh khởi chạy ứng dụng
CMD ["python", "-m", "app.main"]

