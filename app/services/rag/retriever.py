import os
from qdrant_client import QdrantClient
from app.services.rag.embeddings import get_embedding
from app.core.config import Config
from app.core.logger import logger

_client = None

def get_qdrant_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=Config.QDRANT_URL)
    return _client


def vector_search(query: str, top_k: int = None) -> list:
    """
    Perform vector similarity search against Qdrant database.
    Returns top_k most similar chunks with their scores.
    """
    top_k = top_k or Config.TOP_K_RESULTS
    query_embedding = get_embedding(query)

    try:
        client = get_qdrant_client()
        response = client.query_points(
            collection_name=Config.QDRANT_COLLECTION,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        search_result = response.points
    except Exception as e:
        logger.error(f"[Error] Qdrant search failed: {e}")
        return []
    
    return _parse_qdrant_results(search_result)


def multi_query_search(queries: list, top_k: int = None) -> list:
    """
    Multi-query retrieval: run multiple semantically diverse queries,
    merge and deduplicate results by chunk_id, keeping the highest score.
    
    This ensures broader coverage across different legal documents
    when a single query might only match one perspective.
    
    Args:
        queries: List of query strings (original + LLM-generated variants)
        top_k: Max results PER query (final result may be larger before truncation)
    
    Returns:
        Merged, deduplicated, and sorted list of search results
    """
    top_k = top_k or Config.TOP_K_RESULTS
    
    # Collect all results, deduplicate by chunk_id (keep best score)
    merged = {}  # key: chunk_id or (doc_title, dieu_so) -> result dict
    
    for query in queries:
        try:
            query_embedding = get_embedding(query)
            client = get_qdrant_client()
            response = client.query_points(
                collection_name=Config.QDRANT_COLLECTION,
                query=query_embedding.tolist(),
                limit=top_k,
                with_payload=True
            )
            
            results = _parse_qdrant_results(response.points)
            
            for r in results:
                # Dedup key: prefer chunk_id, fallback to (doc, article)
                dedup_key = r.get("chunk_id") or f"{r.get('doc_title')}_{r.get('dieu_so')}"
                
                if dedup_key not in merged or r["score"] > merged[dedup_key]["score"]:
                    merged[dedup_key] = r
                    
        except Exception as e:
            logger.error(f"[Error] Multi-query search failed for query '{query[:50]}...': {e}")
            continue
    
    # Sort by score descending and return top results
    all_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    
    logger.info(f"[MultiQuery] {len(queries)} queries → {len(all_results)} unique chunks (from {sum(1 for _ in queries)} searches)")
    
    return all_results[:top_k + 3]  # Return slightly more to ensure diversity


def _parse_qdrant_results(search_result) -> list:
    """Parse Qdrant search results into standardized dicts."""
    results = []
    for hit in search_result:
        p = hit.payload
        # Build article and chapter string
        article = f"Điều {p.get('dieu_so', '')}" + (f". {p.get('dieu_ten', '')}" if p.get('dieu_ten') else "") if p.get('dieu_so') else ""
        chapter = f"Chương {p.get('chuong_so', '')}" + (f". {p.get('chuong_ten', '')}" if p.get('chuong_ten') else "") if p.get('chuong_so') else ""
        
        results.append({
            "chunk_id": p.get("chunk_id"),
            # Parent-Child Chunking: Use the full article text (parent) if available, otherwise fallback to the matched chunk (child)
            "content": p.get("full_dieu_text") or p.get("noi_dung_chunk") or p.get("content", ""),
            "context_text": p.get("context_text", ""),
            "dieu_so": p.get("dieu_so", ""),
            "dieu_ten": p.get("dieu_ten", ""),
            "chuong_so": p.get("chuong_so", ""),
            "chuong_ten": p.get("chuong_ten", ""),
            "article": article,
            "chapter": chapter,
            "doc_title": p.get("ten_van_ban", ""),
            "so_hieu": p.get("so_hieu", ""),
            "loai_van_ban": p.get("loai_van_ban", ""),
            "trang_thai": p.get("trang_thai", ""),
            "nhom": p.get("nhom", ""),
            "score": float(hit.score),
        })
            
    return results


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
