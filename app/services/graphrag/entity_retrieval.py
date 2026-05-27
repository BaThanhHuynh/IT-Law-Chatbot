"""
Entity-Centric GraphRAG retrieval.

Thay thế approach query-expansion (LLM sinh sub-queries + abbreviation rules
+ static domain queries) bằng cách:
  1. Trích xuất entities có cấu trúc (JSON) từ query qua LLM
  2. Embed mỗi entity → tìm KG node tương đồng nhất
  3. Traverse từ matched nodes để lấy article liên quan
  4. Fusion với full-query embedding + chunk-level RAG

Ưu điểm so với query expansion:
- Bỏ hardcoded abbreviation list (CNTT, SHTT, ANM...)
- Bỏ static domain queries (rule-based)
- Giảm số LLM call từ 5 → 1-2
- Explainable: trace được "answer này vì matched entity X"
- Phù hợp paradigm Microsoft GraphRAG
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.core.config import Config
from app.core.logger import logger
from app.services.rag.embeddings import get_embedding, cosine_similarity
from app.services.rag.retriever import vector_search
from app.services.graphrag.knowledge_graph import get_knowledge_graph


# ────────────────────────────────────────────────────────────────────────────
#  CACHE TOÀN BỘ KG ENTITY EMBEDDINGS (load 1 lần, search bằng matmul)
# ────────────────────────────────────────────────────────────────────────────
_KG_ENTITIES = None      # list[dict] — metadata mỗi entity
_KG_EMBED_MATRIX = None  # np.ndarray (N, D), đã L2-normalize
_KG_LOAD_LOCK = threading.Lock()


def _load_kg_entity_index():
    """
    Load tất cả entity (kèm embedding) từ Neo4j vào RAM 1 lần.
    Sau đó mọi entity-to-node matching chỉ là matmul O(N×D).
    Thread-safe (double-checked locking).
    """
    global _KG_ENTITIES, _KG_EMBED_MATRIX
    if _KG_ENTITIES is not None:
        return _KG_ENTITIES, _KG_EMBED_MATRIX

    with _KG_LOAD_LOCK:
        if _KG_ENTITIES is not None:
            return _KG_ENTITIES, _KG_EMBED_MATRIX

        logger.info("[EntityRetrieval] Loading KG entity index from Neo4j...")
        kg = get_knowledge_graph()
        cypher = """
        MATCH (n:Entity)
        WHERE n.embedding IS NOT NULL
        RETURN n.entity_id AS entity_id,
               n.name AS name,
               n.description AS description,
               [l IN labels(n) WHERE l <> 'Entity'][0] AS entity_type,
               n.embedding AS embedding
        """
        rows = kg.graph.query(cypher)

        entities, embeds = [], []
        for r in rows:
            emb = r.get("embedding")
            if not emb:
                continue
            entities.append({
                "entity_id": r.get("entity_id"),
                "name": r.get("name", "") or "",
                "description": r.get("description", "") or "",
                "entity_type": r.get("entity_type") or "UNKNOWN",
            })
            embeds.append(emb)

        _KG_ENTITIES = entities
        mat = np.array(embeds, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        _KG_EMBED_MATRIX = mat / norms

        logger.info(f"[EntityRetrieval] Loaded {len(entities)} entities (dim={_KG_EMBED_MATRIX.shape[1]})")
        return _KG_ENTITIES, _KG_EMBED_MATRIX


# ────────────────────────────────────────────────────────────────────────────
#  STRUCTURED ENTITY EXTRACTION (1 LLM call → JSON output)
# ────────────────────────────────────────────────────────────────────────────
def extract_entities_structured(query: str) -> list:
    """
    Gọi LLM (Gemini) trích xuất entities có cấu trúc từ query.
    Trả về list[{text, type}].

    Có 3 mức fallback:
      1. Parse JSON từ LLM response (chuẩn)
      2. Regex tìm patterns nếu JSON parse fail
      3. Fallback đơn giản: split + filter stopwords
    """
    from app.services.chatbot.engine import get_llm
    from app.services.chatbot.prompts import ENTITY_EXTRACTION_STRUCTURED_PROMPT

    try:
        model = get_llm()
        prompt = ENTITY_EXTRACTION_STRUCTURED_PROMPT.format(query=query)
        response = model.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
        )
        text = response.text.strip()

        # Loại bỏ markdown code fence nếu LLM trả về ```json
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        data = json.loads(text)
        entities = data.get("entities", [])

        # Validate format
        valid = []
        for e in entities:
            if isinstance(e, dict) and "text" in e and e["text"].strip():
                valid.append({
                    "text": e["text"].strip(),
                    "type": e.get("type", "UNKNOWN"),
                })
        if valid:
            return valid

    except json.JSONDecodeError as e:
        logger.warning(f"[EntityExtraction] JSON parse fail: {e}. Falling back to regex.")
    except Exception as e:
        logger.error(f"[EntityExtraction] LLM call failed: {e}")

    # Fallback: split câu, lọc stopwords Việt
    return _fallback_entity_extraction(query)


_STOPWORDS_VI = {
    "là", "của", "và", "có", "được", "trong", "ra", "với", "cho", "từ",
    "đến", "như", "thế", "nào", "gì", "bao", "nhiêu", "thì", "mà", "này",
    "đó", "các", "những", "một", "khi", "nếu", "hay", "hoặc", "tại", "về",
    "theo", "bị", "sẽ", "đã", "có", "không", "phải", "sao",
}


def _fallback_entity_extraction(query: str) -> list:
    """Khi LLM fail, dùng heuristic đơn giản: word-level + filter stopwords."""
    # Tách câu thành ngrams (2-3 từ)
    tokens = re.findall(r"\w+", query.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS_VI and len(t) >= 2]

    entities = []
    # Bigrams
    for i in range(len(tokens) - 1):
        entities.append({"text": f"{tokens[i]} {tokens[i+1]}", "type": "UNKNOWN"})
    # Unigrams quan trọng (>= 4 ký tự)
    for t in tokens:
        if len(t) >= 4:
            entities.append({"text": t, "type": "UNKNOWN"})

    return entities[:8]  # giới hạn


# ────────────────────────────────────────────────────────────────────────────
#  PURE VECTOR SEARCH ENTITY → KG NODE
# ────────────────────────────────────────────────────────────────────────────
def kg_node_search_by_embedding(query_emb: np.ndarray, top_k: int) -> list:
    """Cosine similarity giữa 1 embedding và toàn bộ KG nodes."""
    entities, mat = _load_kg_entity_index()
    if not entities:
        return []

    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    scores = mat @ q

    k = min(top_k, len(scores))
    if k <= 0:
        return []
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [
        {"entity": entities[i], "score": float(scores[i])}
        for i in top_idx
    ]


# ────────────────────────────────────────────────────────────────────────────
#  ENTITY-CENTRIC RETRIEVAL (core)
# ────────────────────────────────────────────────────────────────────────────
def entity_centric_retrieval(
    query: str,
    top_k: int = 5,
    entities_per_query: int = 3,
    nodes_per_entity: int = 3,
) -> dict:
    """
    Pipeline entity-centric:
      1. Extract entities từ query (LLM JSON)
      2. Embed mỗi entity song song
      3. Mỗi entity → top-N node tương đồng nhất
      4. Aggregate score: 1 node có thể được kéo bởi nhiều entity → cộng dồn
      5. Sort & return top_k articles

    Returns:
        {
          "matched_entities": [{text, type, top_nodes: [...]}],
          "ranked_nodes": [{entity, aggregated_score, hit_count}],
          "extraction_used_llm": bool,
        }
    """
    # Bước 1: Extract entities
    extracted = extract_entities_structured(query)
    if not extracted:
        return {"matched_entities": [], "ranked_nodes": [], "extraction_used_llm": False}

    # Lấy tối đa entities_per_query entity quan trọng nhất
    # Ưu tiên type LUẬT/ĐIỀU/HÀNH_VI/KHÁI_NIỆM > CHỦ_THỂ/CHẾ_TÀI
    priority = {
        "LUẬT": 1, "ĐIỀU": 2, "HÀNH_VI": 3, "KHÁI_NIỆM": 4,
        "CHỦ_THỂ": 5, "CHẾ_TÀI": 6, "UNKNOWN": 7,
    }
    extracted_sorted = sorted(extracted, key=lambda x: priority.get(x.get("type", "UNKNOWN"), 9))
    chosen = extracted_sorted[:entities_per_query]

    # Bước 2-3: Embed song song + tìm node tương đồng
    aggregated = {}  # entity_id → {entity, agg_score, hit_count}
    matched_entities_out = []

    def _process_one(ent):
        emb = get_embedding(ent["text"])
        nodes = kg_node_search_by_embedding(emb, top_k=nodes_per_entity)
        return ent, nodes

    with ThreadPoolExecutor(max_workers=min(len(chosen), 4)) as executor:
        for ent, nodes in executor.map(_process_one, chosen):
            matched_entities_out.append({**ent, "top_nodes": nodes})
            for n in nodes:
                eid = n["entity"]["entity_id"]
                if eid not in aggregated:
                    aggregated[eid] = {
                        "entity": n["entity"],
                        "agg_score": 0.0,
                        "hit_count": 0,
                    }
                aggregated[eid]["agg_score"] += n["score"]
                aggregated[eid]["hit_count"] += 1

    # Bước 4-5: Sort theo (hit_count, agg_score) — node được nhiều entity match là quan trọng nhất
    ranked = sorted(
        aggregated.values(),
        key=lambda x: (x["hit_count"], x["agg_score"]),
        reverse=True,
    )

    return {
        "matched_entities": matched_entities_out,
        "ranked_nodes": ranked[:top_k],
        "extraction_used_llm": True,
    }


# ────────────────────────────────────────────────────────────────────────────
#  HYBRID 3-WAY: ENTITY-CENTRIC + QUERY-EMBED + CHUNK-RAG
# ────────────────────────────────────────────────────────────────────────────
def hybrid_entity_retrieval(query: str, top_k: int = 5) -> dict:
    """
    Hybrid 3 nguồn fuse bằng Reciprocal Rank Fusion (RRF):
      Branch A — Entity-centric: trích entity → match KG node
      Branch B — Query-embed: embed full query → tìm KG node tương đồng
      Branch C — Chunk-RAG: vector search trên Qdrant chunks

    Mỗi candidate có (doc_title, dieu_so). Cộng RRF từ 3 branch để rank cuối.
    """
    rrf_k = 60

    # ── Branch A ──
    entity_result = entity_centric_retrieval(query, top_k=top_k * 2)
    branch_a = entity_result.get("ranked_nodes", [])

    # ── Branch B ──
    query_emb = get_embedding(query)
    branch_b = kg_node_search_by_embedding(query_emb, top_k=top_k * 2)

    # ── Branch C ──
    try:
        branch_c = vector_search(query, top_k=top_k * 2)
    except Exception as e:
        logger.error(f"[Hybrid] Chunk RAG failed: {e}")
        branch_c = []

    # Fuse bằng RRF — key thống nhất là entity_id (cho KG branches) hoặc (doc, dieu) cho chunks
    fused = {}  # key → {score, source: dict}

    def _add(key, rank, branch_name, source_obj):
        s = 1.0 / (rrf_k + rank + 1)
        if key not in fused:
            fused[key] = {"score": 0.0, "branches": set(), "source": source_obj}
        fused[key]["score"] += s
        fused[key]["branches"].add(branch_name)

    # Branch A: dùng entity_id của KG node làm key
    for i, item in enumerate(branch_a):
        eid = item["entity"]["entity_id"]
        _add(eid, i, "entity", {"type": "kg_node", "entity": item["entity"], "score": item["agg_score"]})

    # Branch B: cũng entity_id
    for i, item in enumerate(branch_b):
        eid = item["entity"]["entity_id"]
        _add(eid, i, "query_embed", {"type": "kg_node", "entity": item["entity"], "score": item["score"]})

    # Branch C: (doc_title, dieu_so)
    for i, r in enumerate(branch_c):
        key = f"chunk::{r.get('doc_title','')}__D{r.get('dieu_so','')}"
        _add(key, i, "chunk", {"type": "chunk", "chunk": r, "score": r.get("score", 0.0)})

    # Sort theo RRF score
    ranked = sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)[:top_k]

    return {
        "ranked_results": [
            {"key": k, "rrf_score": v["score"], "branches": list(v["branches"]), **v["source"]}
            for k, v in ranked
        ],
        "entity_branch": branch_a,
        "query_branch": branch_b,
        "chunk_branch": branch_c,
        "matched_entities": entity_result.get("matched_entities", []),
    }
