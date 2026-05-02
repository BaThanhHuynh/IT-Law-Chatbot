"""
embed_to_qdrant.py
==================
Đọc law_chunks.jsonl → embed bằng PhoBERT → nạp vào Qdrant

AI Engineer: Phan Quyết Tâm Phú
Project: Chatbot AI tra cứu Luật CNTT VN

Cài đặt:
    pip install transformers torch qdrant-client tqdm

Chạy Qdrant local (Docker):
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Cách dùng:
    # Nạp toàn bộ (lần đầu)
    python embed_to_qdrant.py --input data/law_chunks_hier.jsonl

    # Dùng GPU nếu có
    python embed_to_qdrant.py --input data/law_chunks_hier.jsonl --device cuda

    # Dùng model fine-tuned (sau khi Thành xong)
    python embed_to_qdrant.py --input data/law_chunks.jsonl --model ./phobert-law-finetuned
"""

import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Cấu hình ─────────────────────────────────────────────────────────────────
MODEL_NAME      = r"C:\law_final_model_v2"   # Fine-tuned model path
COLLECTION_NAME = "it_law_chunks"
VECTOR_DIM      = 384                        # MiniLM-L12-v2 output dim
BATCH_SIZE      = 256                        # số chunk xử lý mỗi lần
QDRANT_URL      = "http://localhost:6333"


# ── Embedding ─────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    log.info(f"📥 Load model: {model_name} → device={device}")
    model = SentenceTransformer(model_name, device=device)
    log.info("✅ Model loaded")
    return model


def embed_batch(texts: list, model, device: str) -> list:
    """Embed một batch texts, trả về list vector."""
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


# ── Qdrant ────────────────────────────────────────────────────────────────────

def init_collection(client: QdrantClient, collection_name: str, vector_dim: int, recreate: bool = False):
    """Tạo collection nếu chưa có, hoặc recreate nếu chỉ định."""
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            log.info(f"🗑  Xoá và tạo lại collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            log.info(f"✅ Collection đã tồn tại: {collection_name} (bỏ qua tạo mới)")
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,   # cosine vì đã normalize L2
        )
    )
    log.info(f"✅ Tạo collection: {collection_name} (dim={vector_dim}, metric=COSINE)")


def upload_chunks(client: QdrantClient, collection_name: str,
                  chunks: list, model, device: str,
                  batch_size: int):
    """Embed và upload từng batch vào Qdrant."""
    total = len(chunks)
    log.info(f"🚀 Bắt đầu embed và upload {total:,} chunks...")

    points_buffer = []

    for i in tqdm(range(0, total, batch_size), desc="Embedding"):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]

        vectors = embed_batch(texts, model, device)

        for chunk, vector in zip(batch, vectors):
            point = PointStruct(
                id=int(chunk["id"], 16) % (2**63),  # Qdrant cần int ID
                vector=vector,
                payload={
                    # Metadata đầy đủ để filter và trả về kết quả
                    "chunk_id":          chunk["id"],
                    "text":              chunk["text"],       # text gốc để trả về cho user
                    **chunk.get("payload", {})
                }
            )
            points_buffer.append(point)

        # Upload mỗi 200 points để tránh timeout
        if len(points_buffer) >= 200:
            client.upsert(collection_name=collection_name, points=points_buffer)
            points_buffer = []

    # Upload phần còn lại
    if points_buffer:
        client.upsert(collection_name=collection_name, points=points_buffer)

    # Verify
    count = client.count(collection_name=collection_name).count
    log.info(f"✅ Upload hoàn tất: {count:,} vectors trong Qdrant")


# ── Search demo ───────────────────────────────────────────────────────────────

def search_demo(client: QdrantClient, collection_name: str,
                model, device: str):
    """Test tìm kiếm với câu hỏi mẫu."""
    queries = [
        "Quyền của người dùng đối với dữ liệu cá nhân",
        "Xử phạt vi phạm về an toàn thông tin mạng",
        "Điều kiện cấp phép kinh doanh dịch vụ viễn thông",
    ]

    log.info("\n🔍 Demo tìm kiếm:")
    for query in queries:
        vec = embed_batch([query], model, device)[0]
        results = client.query_points(
            collection_name=collection_name,
            query=vec,
            limit=3,
            # Chỉ tìm trong văn bản còn hiệu lực
            query_filter=Filter(
                must=[FieldCondition(key="trang_thai", match=MatchValue(value="con_hieu_luc"))]
            ),
            with_payload=True,
        )
        print(f"\n  Query: {query}")
        for r in results.points:
            print(f"    [{r.score:.4f}] {r.payload.get('ten_van_ban','')} Điều {r.payload.get('dieu_so','')}: {r.payload.get('noi_dung_chunk','')[:100]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embed law_chunks.jsonl vào Qdrant")
    parser.add_argument("--input",      "-i", default="../data/law_chunks.jsonl")
    parser.add_argument("--model",      "-m", default=MODEL_NAME, help="Model path hoặc HF ID")
    parser.add_argument("--device",     "-d", default="cpu")
    parser.add_argument("--batch_size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--qdrant_url", default=QDRANT_URL)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--recreate",   action="store_true", help="Xoá và tạo lại collection")
    parser.add_argument("--demo",       action="store_true", help="Chạy demo search sau khi upload")
    args = parser.parse_args()

    # Load data
    script_dir = Path(__file__).parent.absolute()
    input_path = script_dir / args.input
    if not input_path.exists():
        log.error(f"Không tìm thấy: {input_path}")
        return

    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log.info(f"📂 Đọc {len(chunks):,} chunks từ {input_path}")

    # Load model
    model = load_model(args.model, args.device)

    # Kết nối Qdrant
    log.info(f"🔗 Kết nối Qdrant: {args.qdrant_url}")
    client = QdrantClient(url=args.qdrant_url)

    # Init collection
    init_collection(client, args.collection, VECTOR_DIM, recreate=args.recreate)

    # Embed & upload
    upload_chunks(
        client, args.collection,
        chunks, model, args.device,
        args.batch_size
    )

    # Demo search
    if args.demo:
        search_demo(client, args.collection, model, args.device)

    log.info("\n🎉 Hoàn tất! Qdrant collection sẵn sàng cho RAG.")
    log.info(f"   Collection: {args.collection}")
    log.info(f"   Qdrant UI:  {args.qdrant_url}/dashboard")


if __name__ == "__main__":
    main()
