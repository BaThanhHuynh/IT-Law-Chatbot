import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from app.services.rag.embeddings import get_embedding
from app.core.config import Config
from app.core.logger import logger
from app.services.rag.bm25 import get_bm25_service

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


def _dedup_key(r: dict) -> str:
    """Stable key to deduplicate a chunk across the dense / BM25 result lists."""
    return r.get("chunk_id") or f"{r.get('doc_title')}_{r.get('dieu_so')}"


def _normalize_bm25_scores(bm25_results: list) -> None:
    """Map raw BM25 scores into a cosine-like ~[0.30, 0.60] band, in place.

    Downstream calibrate_score() and the graph-injection threshold both assume a
    cosine-similarity scale (~0.3–0.7). Raw BM25 scores are unbounded and would
    saturate those functions, so we min-max normalize the sparse hits onto a
    comparable band purely for display/calibration. Final ordering is decided by
    RRF rank, not by these values.
    """
    if not bm25_results:
        return
    raw = [r.get("score", 0.0) for r in bm25_results]
    lo, hi = min(raw), max(raw)
    span = (hi - lo) or 1.0
    for r in bm25_results:
        norm = (r.get("score", 0.0) - lo) / span          # 0..1
        r["score"] = round(0.30 + 0.30 * norm, 3)          # 0.30..0.60


def multi_query_search(queries: list, top_k: int = None) -> list:
    """
    Reranker-free hybrid retrieval using Reciprocal Rank Fusion (RRF).

    Pipeline:
      1. Run each (sub-)query against the dense vector index in parallel,
         keeping one ranked list per query.
      2. Run the primary query against the sparse BM25 index (one ranked list).
      3. Fuse every ranked list with RRF:  score(d) = Σ 1 / (k + rank_i(d)).

    RRF replaces the BAAI/bge-reranker-v2-m3 cross-encoder. It needs no model
    forward pass (removing the dominant CPU latency cost) yet preserves precision
    by rewarding chunks that rank highly across multiple query reformulations and
    BM25 — the same consensus signal the cross-encoder approximated.
    """
    top_k = top_k or Config.TOP_K_RESULTS
    pool_size = Config.HYBRID_POOL_SIZE
    primary_query = queries[0] if queries else ""

    # Retrieval depth: the primary query goes deeper, variants stay shallow.
    n_vector = min(max(top_k * 3, 15), pool_size)
    n_bm25 = max(pool_size - n_vector, 8)

    # 1. Dense retrieval — one ranked list per sub-query (parallel).
    from app.services.rag.embeddings import get_embeddings_batch
    try:
        query_embeddings = get_embeddings_batch(queries)
        query_embeddings_list = [emb.tolist() for emb in query_embeddings]
    except Exception as e:
        logger.error(f"[Error] Batch embedding subqueries failed, falling back: {e}")
        query_embeddings_list = [get_embedding(q).tolist() for q in queries]

    per_query_results: list = [[] for _ in queries]
    max_workers = min(len(queries), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, q in enumerate(queries):
            limit = n_vector if idx == 0 else max(top_k, 5)
            future_to_idx[executor.submit(
                _single_query_search_with_embedding, query_embeddings_list[idx], limit
            )] = idx
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results = future.result()
            for r in results:
                r["_source"] = "vector"
            per_query_results[idx] = results

    # 2. Sparse retrieval — single BM25 ranked list on the primary query.
    bm25_results: list = []
    if primary_query:
        try:
            bm25_results = get_bm25_service().search(primary_query, top_k=n_bm25)
            _normalize_bm25_scores(bm25_results)
            logger.info(f"[HybridSearch] BM25 retrieved {len(bm25_results)} candidates.")
        except Exception as e:
            logger.error(f"[HybridSearch Error] BM25 search failed: {e}")

    # 3. Pick one representative object per unique chunk.
    #    Prefer the dense copy (carries a real cosine score); flag hybrid hits.
    obj_by_key: dict = {}
    for results in per_query_results:
        for r in results:
            key = _dedup_key(r)
            if key not in obj_by_key or r.get("score", 0) > obj_by_key[key].get("score", 0):
                obj_by_key[key] = r
    for r in bm25_results:
        key = _dedup_key(r)
        if key in obj_by_key:
            obj_by_key[key]["_source"] = "hybrid"   # retrieved by both dense & sparse
        else:
            obj_by_key[key] = r                       # BM25-only candidate

    # 4. Reciprocal Rank Fusion across every ranked list (dense lists + BM25).
    rrf_k = Config.RRF_K
    fused: dict = {}
    for results in per_query_results + [bm25_results]:
        for rank, r in enumerate(results):
            key = _dedup_key(r)
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked_keys = sorted(fused, key=lambda k: fused[k], reverse=True)

    final = []
    for key in ranked_keys[: top_k + 3]:
        cand = obj_by_key.get(key)
        if cand is None:
            continue
        cand["rrf_score"] = round(fused[key], 5)
        final.append(cand)

    logger.info(
        f"[HybridSearch] RRF fused {len(fused)} unique chunks from "
        f"{len(queries)} dense lists + BM25 -> returning top {len(final)} "
        f"(pool={pool_size}, k={rrf_k})."
    )
    return final


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
