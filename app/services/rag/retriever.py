import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from app.services.rag.embeddings import get_embedding
from app.core.config import Config
from app.core.logger import logger
from app.services.rag.bm25 import get_bm25_service
from app.services.rag.reranker import get_reranker_service

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


def _single_query_search(query: str, top_k: int) -> list:
    """Run a single query against Qdrant. Called from thread pool."""
    try:
        query_embedding = get_embedding(query)
        return _single_query_search_with_embedding(query_embedding.tolist(), top_k)
    except Exception as e:
        logger.error(f"[Error] Search failed for query '{query[:50]}': {e}")
        return []


def _single_query_search_with_embedding(query_embedding: list, top_k: int) -> list:
    """Run a single query against Qdrant using a pre-computed embedding list."""
    try:
        client = get_qdrant_client()
        response = client.query_points(
            collection_name=Config.QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        return _parse_qdrant_results(response.points)
    except Exception as e:
        logger.error(f"[Error] Qdrant search failed: {e}")
        return []


def multi_query_search(queries: list, top_k: int = None) -> list:
    """
    Hybrid Multi-Query search with BM25 and Cross-Encoder Reranker.
    - Retrieving candidate chunks from Vector DB (parallel search).
    - Retrieving candidate chunks from BM25.
    - Merging both candidate pools and deduplicating.
    - Reranking using BAAI/bge-reranker-v2-m3 against queries[0].
    """
    top_k = top_k or Config.TOP_K_RESULTS

    # Bound the rerank pool: each candidate is one cross-encoder forward pass.
    # Keep the strongest dense candidates + a smaller set of sparse (BM25) ones.
    pool_size = Config.RERANK_POOL_SIZE
    n_vector = min(max(top_k * 3, 15), pool_size)
    n_bm25 = max(pool_size - n_vector, 8)

    # 1. Vector Retrieval (Dense)
    vector_candidates: dict = {}
    max_workers = min(len(queries), 8)

    # Batch embed all sub-queries to avoid thread execution contention / GIL bottlenecks
    from app.services.rag.embeddings import get_embeddings_batch
    try:
        query_embeddings = get_embeddings_batch(queries)
        query_embeddings_list = [emb.tolist() for emb in query_embeddings]
    except Exception as e:
        logger.error(f"[Error] Batch embedding subqueries failed, falling back: {e}")
        query_embeddings_list = [get_embedding(q).tolist() for q in queries]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {}
        for idx, q in enumerate(queries):
            limit = n_vector if idx == 0 else max(top_k, 5)
            q_emb = query_embeddings_list[idx]
            future_to_query[executor.submit(_single_query_search_with_embedding, q_emb, limit)] = q

        for future in as_completed(future_to_query):
            results = future.result()
            for r in results:
                dedup_key = r.get("chunk_id") or f"{r.get('doc_title')}_{r.get('dieu_so')}"
                if dedup_key not in vector_candidates or r["score"] > vector_candidates[dedup_key]["score"]:
                    r["_source"] = "vector"
                    vector_candidates[dedup_key] = r

    # Keep only the strongest dense candidates so the rerank pool stays bounded
    top_vector = sorted(vector_candidates.values(), key=lambda x: x["score"], reverse=True)[:n_vector]
    logger.info(f"[HybridSearch] Vector search: {len(vector_candidates)} unique → kept top {len(top_vector)}.")

    # 2. BM25 Retrieval (Sparse)
    bm25_candidates = []
    primary_query = queries[0] if queries else ""
    if primary_query:
        try:
            bm25_service = get_bm25_service()
            bm25_candidates = bm25_service.search(primary_query, top_k=n_bm25)
            logger.info(f"[HybridSearch] BM25 retrieved {len(bm25_candidates)} candidate chunks.")
        except Exception as e:
            logger.error(f"[HybridSearch Error] BM25 search failed: {e}")

    # 3. Merge Vector and BM25 candidates
    merged_candidates: dict = {}

    # Add dense candidates first
    for r in top_vector:
        dedup_key = r.get("chunk_id") or f"{r.get('doc_title')}_{r.get('dieu_so')}"
        merged_candidates[dedup_key] = r

    # Add BM25 candidates (deduplicate)
    for r in bm25_candidates:
        dedup_key = r.get("chunk_id") or f"{r.get('doc_title')}_{r.get('dieu_so')}"
        if dedup_key not in merged_candidates:
            merged_candidates[dedup_key] = r
        else:
            merged_candidates[dedup_key]["_source"] = "hybrid"

    candidate_list = list(merged_candidates.values())
    logger.info(f"[HybridSearch] Rerank pool size: {len(candidate_list)} chunks (cap={pool_size}).")

    # 4. Rerank candidates using Cross-Encoder Reranker
    if primary_query and candidate_list:
        try:
            reranker = get_reranker_service()
            rerank_limit = top_k + 3
            reranked_results = reranker.rerank(primary_query, candidate_list, top_k=rerank_limit)
            logger.info(f"[HybridSearch] Reranking completed. Returned top {len(reranked_results)} chunks.")
            return reranked_results
        except Exception as e:
            logger.error(f"[HybridSearch Error] Reranking failed, falling back: {e}")
            # Fallback to dense candidates sorted by original score
            return sorted(top_vector, key=lambda x: x["score"], reverse=True)[:top_k + 3]

    return sorted(top_vector, key=lambda x: x["score"], reverse=True)[:top_k + 3]


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
            "score": float(hit.score) if hasattr(hit, "score") and hit.score is not None else 0.0,
        })
            
    return results


def get_context_from_results(results: list) -> str:
    """
    Format search results into context string for LLM.

    Two-tier layout:
    - Tier 1: Chunks from vector search (higher confidence) — labeled as primary sources
    - Tier 2: Chunks from graph expansion (supplementary) — labeled separately
    This helps the LLM understand which sources to prioritize for citations.
    """
    if not results:
        return "Không tìm thấy thông tin liên quan."

    # Split by source: vector search results vs graph-expanded results
    vector_chunks = [r for r in results if r.get("_source") != "graph_expand"]
    graph_chunks  = [r for r in results if r.get("_source") == "graph_expand"]

    context_parts = []

    # Tier 1: Vector search results (primary, citation-worthy)
    for i, r in enumerate(vector_chunks, 1):
        source_info = f"[{r.get('doc_title', 'N/A')} ({r.get('so_hieu', '')})"
        if r.get('article'):
            source_info += f" - {r['article']}"
        source_info += f"] (Độ liên quan: {r['score']:.2f})"
        context_parts.append(f"--- Đoạn {i} {source_info} ---\n{r['content']}")

    # Tier 2: Graph-expanded results (supplementary context)
    if graph_chunks:
        context_parts.append("\n--- [Bổ sung từ Knowledge Graph — dùng để hiểu bối cảnh, KHÔNG trích dẫn thêm] ---")
        for r in graph_chunks:
            source_info = f"[{r.get('doc_title', 'N/A')} ({r.get('so_hieu', '')})"
            if r.get('article'):
                source_info += f" - {r['article']}"
            source_info += f"] (Độ liên quan KG: {r['score']:.2f})"
            context_parts.append(f"{source_info}\n{r['content']}")

    return "\n\n".join(context_parts)


def fetch_specific_article(doc_title: str, dieu_so: str) -> list:
    """
    Fetch a specific article from Qdrant by document title and article number.
    Used as fallback for critical articles that vector search might miss.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = get_qdrant_client()
    try:
        response, _ = client.scroll(
            collection_name=Config.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="dieu_so", match=MatchValue(value=dieu_so)),
                    FieldCondition(key="ten_van_ban", match=MatchValue(value=doc_title))
                ]
            ),
            limit=5,
            with_payload=True
        )
        if response:
            return _parse_qdrant_results(response)
        return []
    except Exception as e:
        logger.error(f"[Error] Fetch specific article failed: {e}")
        return []
