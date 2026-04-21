"""
test_qdrant.py - Phiên bản Hierarchical (in full_dieu_text rõ ràng hơn)
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ========================== CẤU HÌNH ==========================
COLLECTION_NAME = "law_cntt_vn"

QUERIES = [
    "những hành vi nào bị nghiêm cấm trên không gian mạng"
]

# ============================================================

print("🔄 Đang load model embedding...")
model = SentenceTransformer("vinai/phobert-base-v2")

print("🔗 Kết nối Qdrant...")
client = QdrantClient(url="http://localhost:6333", timeout=60)

print("\n" + "="*90)
print("BẮT ĐẦU TEST RETRIEVAL HIERARCHICAL")
print("="*90)

for idx, query in enumerate(QUERIES, 1):
    print(f"\n[{idx}/5] Query: {query}")
    
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
        with_payload=True
    )

    for i, scored_point in enumerate(results.points, 1):
        payload = scored_point.payload
        score = scored_point.score

        full_text = payload.get("full_dieu_text", "")
        short_chunk = payload.get("noi_dung_chunk", "")[:180]

        print(f"  {i:2d}. Score: {score:.4f}")
        print(f"      Văn bản : {payload.get('ten_van_ban', 'N/A')}")
        print(f"      Điều    : Điều {payload.get('dieu_so', 'N/A')} - {payload.get('dieu_ten', 'N/A')}")
        print(f"      Chunk   : {short_chunk}...")
        print(f"      Full Điều (Parent - {len(full_text):,} ký tự):")
        print(f"      {full_text[:800]}{' ... (còn {len(full_text)-800} ký tự)' if len(full_text) > 800 else ''}\n")

print("\n✅ Test hoàn tất. full_dieu_text đã được lưu đầy đủ trong payload.")