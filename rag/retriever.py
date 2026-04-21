import numpy as np
from db import fetch_all
from rag.embeddings import get_embedding, deserialize_embedding, cosine_similarity
from config import Config


def vector_search(query: str, top_k: int = None) -> list:
    """
    Perform vector similarity search against stored chunk embeddings.
    Returns top_k most similar chunks with their scores.
    """
    top_k = top_k or Config.TOP_K_RESULTS
    query_embedding = get_embedding(query)

    # Fetch all embeddings from DB
    rows = fetch_all("""
        SELECT ce.chunk_id, ce.embedding,
               dc.content, dc.context_text, dc.dieu_so, dc.dieu_ten,
               dc.chuong_so, dc.chuong_ten, dc.muc_so, dc.muc_ten,
               dc.chunk_tier, dc.is_repealed,
               ld.ten_van_ban, ld.so_hieu, ld.loai_van_ban,
               ld.trang_thai, ld.nhom
        FROM chunk_embeddings ce
        JOIN document_chunks dc ON ce.chunk_id = dc.id
        JOIN law_documents ld ON dc.document_id = ld.id
        WHERE dc.is_repealed = FALSE
    """)

    if not rows:
        return []

    # Calculate similarity scores
    results = []
    for row in rows:
        stored_embedding = deserialize_embedding(row["embedding"])
        score = cosine_similarity(query_embedding, stored_embedding)
        results.append({
            "chunk_id": row["chunk_id"],
            "content": row["content"],
            "context_text": row["context_text"],
            "dieu_so": row["dieu_so"],
            "dieu_ten": row["dieu_ten"],
            "chuong_so": row["chuong_so"],
            "chuong_ten": row["chuong_ten"],
            "article": f"Điều {row['dieu_so']}" + (f". {row['dieu_ten']}" if row["dieu_ten"] else "") if row["dieu_so"] else "",
            "chapter": f"Chương {row['chuong_so']}" + (f". {row['chuong_ten']}" if row["chuong_ten"] else "") if row["chuong_so"] else "",
            "doc_title": row["ten_van_ban"],
            "so_hieu": row["so_hieu"],
            "loai_van_ban": row["loai_van_ban"],
            "trang_thai": row["trang_thai"],
            "nhom": row["nhom"],
            "score": score,
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def get_context_from_results(results: list) -> str:
    """Format search results into context string for LLM."""
    if not results:
        return "Không tìm thấy thông tin liên quan."

    context_parts = []
    for i, r in enumerate(results, 1):
        source_info = f"[{r.get('doc_title', 'N/A')} ({r.get('so_hieu', '')})"
        if r.get('article'):
            source_info += f" - {r['article']}"
        source_info += f"] (Độ liên quan: {r['score']:.2f})"

        context_parts.append(f"--- Đoạn {i} {source_info} ---\n{r['content']}")

    return "\n\n".join(context_parts)
