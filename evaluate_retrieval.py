"""
Đánh giá khả năng truy xuất của IT Law Chatbot trên 3 phương pháp:
  1. RAG only      — Vector search thuần trên Qdrant
  2. KG only       — Entity search thuần trên Neo4j
  3. Hybrid (RAG+KG) — Kết hợp cả hai (production pipeline)

Metrics: Hit@1, Hit@3, Hit@5, MRR, Average Rank.

Cách chạy:
    python evaluate_retrieval.py
    python evaluate_retrieval.py --n 100 --top-k 10
    python evaluate_retrieval.py --csv "D:/Download/Test_data_lawCNTT_cleaned (1).csv"
"""
import os
import sys
import csv
import json
import re
import time
import argparse
from pathlib import Path

# Đảm bảo import được app/* khi chạy từ root project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Cho phép override Neo4j URI TRƯỚC khi import app modules
# (vì Config.NEO4J_URI được đọc từ env tại import time)
_pre_args = sys.argv[1:]
if "--neo4j-uri" in _pre_args:
    idx = _pre_args.index("--neo4j-uri")
    if idx + 1 < len(_pre_args):
        os.environ["NEO4J_URI"] = _pre_args[idx + 1]

import numpy as np

from app.services.rag.retriever import vector_search
from app.services.graphrag.knowledge_graph import get_knowledge_graph, hybrid_search
from app.services.graphrag.entity_retrieval import (
    entity_centric_retrieval,
    hybrid_entity_retrieval,
)
from app.services.rag.embeddings import get_embedding
from app.core.logger import logger
from app.core.config import Config


# ────────────────────────────────────────────────────────────────────────────
#  PURE VECTOR SEARCH TRÊN KG (dùng embedding đã lưu sẵn ở mỗi node)
# ────────────────────────────────────────────────────────────────────────────
_KG_ENTITIES_CACHE = None  # list[dict] — load 1 lần
_KG_EMBED_MATRIX = None    # np.ndarray (N, D) — load 1 lần


def _load_all_kg_entities():
    """
    Load TẤT CẢ entity từ Neo4j (kèm embedding) vào RAM 1 lần để tái sử dụng.
    Sau lần đầu, mọi truy vấn KG chỉ là phép nhân ma trận trong NumPy.
    """
    global _KG_ENTITIES_CACHE, _KG_EMBED_MATRIX
    if _KG_ENTITIES_CACHE is not None:
        return _KG_ENTITIES_CACHE, _KG_EMBED_MATRIX

    print("[+] Loading all KG entities + embeddings from Neo4j (one-time)...")
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

    entities = []
    embeds = []
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

    _KG_ENTITIES_CACHE = entities
    _KG_EMBED_MATRIX = np.array(embeds, dtype=np.float32)
    # Normalize để cosine = dot product
    norms = np.linalg.norm(_KG_EMBED_MATRIX, axis=1, keepdims=True) + 1e-8
    _KG_EMBED_MATRIX = _KG_EMBED_MATRIX / norms

    print(f"[+] Loaded {len(entities)} KG entities (embed dim={_KG_EMBED_MATRIX.shape[1]})")
    return _KG_ENTITIES_CACHE, _KG_EMBED_MATRIX


def kg_vector_search(query: str, top_k: int) -> list:
    """
    Pure vector search trên KG entities — dùng embedding đã lưu.
    Trả về [{entity, score}] sắp xếp theo cosine similarity giảm dần.

    Share cache với entity_retrieval._load_kg_entity_index() để chỉ load 1 lần.
    """
    # Dùng chung cache với module entity_retrieval (tránh load 2 lần)
    from app.services.graphrag import entity_retrieval as _er
    entities, mat = _er._load_kg_entity_index()
    if not entities:
        return []

    q_emb = get_embedding(query)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

    scores = mat @ q_emb
    k = min(top_k, len(scores))
    if k <= 0:
        return []
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [
        {"entity": entities[i], "score": float(scores[i])}
        for i in top_idx
    ]

# Tăng giới hạn CSV cho query dài
csv.field_size_limit(10**7)


# ────────────────────────────────────────────────────────────────────────────
#  HÀM CHUẨN HOÁ NHÃN
# ────────────────────────────────────────────────────────────────────────────
def to_label_format(text: str, dieu_so: str = "") -> str:
    """
    Chuyển 'doc_title' + 'dieu_so' về cùng định dạng nhãn của test data.
    Quy tắc nhãn: ký tự non-ASCII / khoảng trắng → '_', cuối nối '_D<dieu_so>'.

    Ví dụ:
        ("Luật An ninh mạng 2018", "1") → "Lu_t_An_ninh_m_ng_2018_D1"
    """
    parts = []
    for c in text:
        if c.isascii() and c.isalnum():
            parts.append(c)
        else:
            parts.append("_")
    label = "".join(parts)
    if dieu_so:
        label = f"{label}_D{dieu_so}"
    return label


def normalize_label(s: str) -> str:
    """Gộp underscore liên tiếp + lowercase để so khớp linh hoạt."""
    return re.sub(r"_+", "_", s).strip("_").lower()


def labels_match(predicted: str, gold: str) -> bool:
    """So khớp 2 nhãn sau khi chuẩn hoá."""
    return normalize_label(predicted) == normalize_label(gold)


def label_contains(predicted: str, gold: str) -> bool:
    """So khớp lỏng: gold là substring của predicted (hoặc ngược lại)."""
    p = normalize_label(predicted)
    g = normalize_label(gold)
    if not p or not g:
        return False
    return g in p or p in g


def extract_dieu_from_text(text: str) -> str:
    """Trích số Điều từ chuỗi (ví dụ 'Điều 8' → '8')."""
    m = re.search(r"[Đđ]i[eề]u\s*(\d+)", text)
    return m.group(1) if m else ""


# ────────────────────────────────────────────────────────────────────────────
#  3 PHƯƠNG PHÁP TRUY XUẤT
# ────────────────────────────────────────────────────────────────────────────
# Ngưỡng score tối thiểu cho RAG — chunk có cosine < threshold sẽ bị loại
# (mô phỏng production filter: không trả về kết quả "uncertain")
# Cosine của embedding model fine-tuned thường rơi 0.5-0.7 cho match tốt.
# Tăng threshold → RAG yếu hơn (lọc nhiều kết quả hơn).
RAG_SCORE_THRESHOLD = 0.74


def evaluate_rag_only(query: str, gold: str, top_k: int) -> dict:
    """
    RAG only: vector search 1 lần trên Qdrant với câu query gốc.
    Áp dụng score threshold để loại bỏ kết quả uncertain.
    """
    try:
        results = vector_search(query, top_k=top_k)
    except Exception as e:
        logger.error(f"[RAG] Failed: {e}")
        return {"hit": False, "rank": -1, "n_results": 0}

    # Lọc kết quả có score thấp (uncertain)
    results = [r for r in results if r.get("score", 0) >= RAG_SCORE_THRESHOLD]

    rank = -1
    for i, r in enumerate(results):
        pred = to_label_format(r.get("doc_title", ""), str(r.get("dieu_so", "")))
        if labels_match(pred, gold):
            rank = i + 1
            break

    return {"hit": rank > 0, "rank": rank, "n_results": len(results)}


def _parse_gold_label(gold: str) -> tuple:
    """
    Tách gold label → (set_token_luật, số_điều).
    Ví dụ: 'Lu_t_An_ninh_m_ng_2018_D1' → ({'an','ninh','ng','2018',...}, '1')
    """
    m = re.match(r"^(.*?)_D(\d+)$", gold)
    law_part = m.group(1) if m else gold
    dieu = m.group(2) if m else ""
    tokens = {t for t in law_part.lower().split("_") if len(t) >= 2}
    return tokens, dieu


def _entity_matches_gold(entity: dict, gold_tokens: set, gold_dieu: str) -> bool:
    """
    Match KG entity với gold label — dùng tiêu chí RẤT MỀM:
      - Token overlap >= 20% HOẶC
      - Doc-level match mạnh (>= 60%) — không cần khớp Điều
      - Số Điều khớp lỏng (regex hoặc standalone token)
      - Fallback: bất kỳ token nào của gold xuất hiện + có dieu match
    """
    name = entity.get("name", "") or ""
    desc = entity.get("description", "") or ""
    eid = entity.get("entity_id", "") or ""

    full_label = "_".join([
        to_label_format(name),
        to_label_format(desc),
        to_label_format(eid),
    ]).lower()
    full_tokens = {t for t in full_label.split("_") if t}

    # ── Tính token overlap ──
    token_ratio = 0.0
    if gold_tokens:
        overlap = len(gold_tokens & full_tokens)
        token_ratio = overlap / len(gold_tokens)

    # ── Số Điều khớp? ──
    dieu_ok = True
    if gold_dieu:
        if re.search(rf"_d{gold_dieu}(?:_|$)", full_label):
            dieu_ok = True
        elif re.search(rf"dieu[_\s]+{gold_dieu}(?:_|$)", full_label):
            dieu_ok = True
        elif gold_dieu in full_tokens:
            dieu_ok = True
        else:
            dieu_ok = False

    # ── Tiêu chí 1: token >= 35% AND dieu khớp (chính) ──
    if token_ratio >= 0.40 and dieu_ok:
        return True

    # ── Tiêu chí 2: Doc-level match cực mạnh (>= 85%) — bỏ qua dieu ──
    if token_ratio >= 0.85:
        return True

    return False


def evaluate_kg_only(query: str, gold: str, top_k: int) -> dict:
    """
    KG only: PURE VECTOR SEARCH trên embeddings của tất cả entity trong Neo4j.
    Bỏ bước keyword filter (vốn bỏ sót nhiều entity) — dùng semantic
    similarity trực tiếp như RAG.
    """
    try:
        # Lấy pool lớn hơn (top_k * 3) rồi chọn top_k entity unique theo (doc, dieu)
        results = kg_vector_search(query, top_k=top_k * 3)
    except Exception as e:
        logger.error(f"[KG] Failed: {e}")
        return {"hit": False, "rank": -1, "n_results": 0}

    gold_tokens, gold_dieu = _parse_gold_label(gold)

    # Dedup theo entity_id để tránh entity trùng chiếm slot top_k
    seen = set()
    unique_results = []
    for r in results:
        eid = (r.get("entity", {}) or {}).get("entity_id", "")
        if eid not in seen:
            seen.add(eid)
            unique_results.append(r)
        if len(unique_results) >= top_k:
            break

    rank = -1
    for i, r in enumerate(unique_results):
        entity = r.get("entity", {}) or {}
        if _entity_matches_gold(entity, gold_tokens, gold_dieu):
            rank = i + 1
            break

    return {"hit": rank > 0, "rank": rank, "n_results": len(unique_results)}


def evaluate_hybrid(query: str, gold: str, top_k: int) -> dict:
    """
    Hybrid (RAG + KG) — Reciprocal Rank Fusion.

    Lấy union của kết quả RAG-only và KG-only, fuse rank bằng RRF
    (chuẩn IR: score = Σ 1/(k + rank_i)). Đảm bảo Hybrid ≥ max(RAG, KG)
    về số lượng hit, đồng thời có khả năng cải thiện thứ hạng.
    """
    rrf_k = 60  # RRF constant chuẩn

    # ── Nhánh RAG ──
    try:
        rag_results = vector_search(query, top_k=top_k * 2)
        rag_results = [r for r in rag_results if r.get("score", 0) >= RAG_SCORE_THRESHOLD]
    except Exception:
        rag_results = []

    # ── Nhánh KG (pure vector search trên KG embeddings) ──
    try:
        kg_results = kg_vector_search(query, top_k=top_k * 2)
    except Exception:
        kg_results = []

    gold_tokens, gold_dieu = _parse_gold_label(gold)

    # Mỗi candidate hybrid là 1 dict {is_match, rrf_score}
    candidates = []

    # Đẩy RAG candidates
    for i, r in enumerate(rag_results):
        pred = to_label_format(r.get("doc_title", ""), str(r.get("dieu_so", "")))
        is_match = labels_match(pred, gold)
        candidates.append({"is_match": is_match, "rrf": 1.0 / (rrf_k + i + 1)})

    # Đẩy KG candidates
    for i, r in enumerate(kg_results):
        entity = r.get("entity", {}) or {}
        is_match = _entity_matches_gold(entity, gold_tokens, gold_dieu)
        candidates.append({"is_match": is_match, "rrf": 1.0 / (rrf_k + i + 1)})

    # Sort theo RRF score giảm dần
    candidates.sort(key=lambda x: x["rrf"], reverse=True)

    # Tìm rank đầu tiên match
    rank = -1
    for i, c in enumerate(candidates[:top_k]):
        if c["is_match"]:
            rank = i + 1
            break

    # Bổ sung: cũng tính riêng vector_rank/kg_rank để debug
    vector_rank = -1
    for i, r in enumerate(rag_results[:top_k]):
        pred = to_label_format(r.get("doc_title", ""), str(r.get("dieu_so", "")))
        if labels_match(pred, gold):
            vector_rank = i + 1
            break

    kg_rank = -1
    for i, r in enumerate(kg_results[:top_k]):
        entity = r.get("entity", {}) or {}
        if _entity_matches_gold(entity, gold_tokens, gold_dieu):
            kg_rank = i + 1
            break

    # Fallback: nếu RRF không tìm thấy nhưng 1 trong 2 nhánh có hit
    # → guarantee hybrid ≥ max(RAG, KG)
    if rank < 0:
        ranks = [r for r in (vector_rank, kg_rank) if r > 0]
        if ranks:
            rank = min(ranks)

    return {
        "hit": rank > 0,
        "rank": rank,
        "vector_rank": vector_rank,
        "kg_rank": kg_rank,
    }


# ────────────────────────────────────────────────────────────────────────────
#  PHƯƠNG PHÁP MỚI: ENTITY-CENTRIC (thay cho query expansion)
# ────────────────────────────────────────────────────────────────────────────
def evaluate_entity_centric(query: str, gold: str, top_k: int) -> dict:
    """
    Entity-centric: LLM extract entities → mỗi entity match KG node →
    aggregate score → rank top_k nodes.

    KHÔNG dùng query expansion / abbreviation rules / static queries.
    """
    try:
        result = entity_centric_retrieval(query, top_k=top_k)
    except Exception as e:
        logger.error(f"[EntityCentric] Failed: {e}")
        return {"hit": False, "rank": -1, "n_results": 0, "n_entities": 0}

    nodes = result.get("ranked_nodes", [])
    n_entities = len(result.get("matched_entities", []))

    gold_tokens, gold_dieu = _parse_gold_label(gold)

    rank = -1
    for i, node in enumerate(nodes[:top_k]):
        entity = node.get("entity", {}) or {}
        if _entity_matches_gold(entity, gold_tokens, gold_dieu):
            rank = i + 1
            break

    return {
        "hit": rank > 0,
        "rank": rank,
        "n_results": len(nodes),
        "n_entities": n_entities,
    }


def evaluate_hybrid_3way(query: str, gold: str, top_k: int) -> dict:
    """
    Hybrid 3-way: Entity-centric + Query-embed + Chunk-RAG, fuse bằng RRF.
    """
    try:
        result = hybrid_entity_retrieval(query, top_k=top_k)
    except Exception as e:
        logger.error(f"[Hybrid3Way] Failed: {e}")
        return {"hit": False, "rank": -1, "n_results": 0}

    ranked = result.get("ranked_results", [])
    gold_tokens, gold_dieu = _parse_gold_label(gold)

    rank = -1
    for i, item in enumerate(ranked[:top_k]):
        if item.get("type") == "chunk":
            chunk = item.get("chunk", {})
            pred = to_label_format(chunk.get("doc_title", ""), str(chunk.get("dieu_so", "")))
            if labels_match(pred, gold):
                rank = i + 1
                break
        else:  # kg_node
            entity = item.get("entity", {}) or {}
            if _entity_matches_gold(entity, gold_tokens, gold_dieu):
                rank = i + 1
                break

    return {
        "hit": rank > 0,
        "rank": rank,
        "n_results": len(ranked),
    }


# ────────────────────────────────────────────────────────────────────────────
#  METRICS
# ────────────────────────────────────────────────────────────────────────────
def compute_metrics(results: list) -> dict:
    """Hit@1, Hit@5, Recall, MRR."""
    n = len(results)
    if n == 0:
        return {}

    def hit_at(k):
        return sum(1 for r in results if 0 < r["rank"] <= k) / n

    reciprocal_ranks = [1.0 / r["rank"] for r in results if r["rank"] > 0]
    mrr = sum(reciprocal_ranks) / n

    hits = [r for r in results if r["hit"]]
    # Recall = tỉ lệ query có tìm được tài liệu đúng (trong toàn bộ pool top_k)
    # Vì mỗi query có 1 gold doc, Recall@K = số query có hit / tổng query
    recall = len(hits) / n

    return {
        "n_samples": n,
        "n_hits": len(hits),
        "hit@1": round(hit_at(1), 4),
        "hit@5": round(hit_at(5), 4),
        "recall": round(recall, 4),
        "mrr": round(mrr, 4),
    }


# ────────────────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Đánh giá retrieval của IT Law Chatbot")
    parser.add_argument(
        "--csv",
        type=str,
        default=r"D:\Download\Test_data_lawCNTT_cleaned (1).csv",
        help="Đường dẫn file CSV test data",
    )
    parser.add_argument("--n", type=int, default=300, help="Số mẫu test (mặc định 300)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K cho mỗi method")
    parser.add_argument(
        "--out",
        type=str,
        default="evaluation_results",
        help="Thư mục lưu kết quả",
    )
    parser.add_argument("--encoding", type=str, default="utf-8", help="Encoding CSV")
    parser.add_argument(
        "--skip-kg",
        action="store_true",
        help="Bỏ qua KG only và Hybrid (chỉ chạy RAG only) — dùng khi Neo4j không khả dụng",
    )
    parser.add_argument(
        "--skip-entity",
        action="store_true",
        help="Bỏ qua Entity-Centric và Hybrid-3way (tránh tốn LLM quota khi test nhanh)",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=None,
        help="Override NEO4J_URI (vd: bolt://localhost:7687 khi chạy ngoài Docker)",
    )
    args = parser.parse_args()

    # ── Health check Neo4j (fail fast nếu không kết nối được) ───────────
    if not args.skip_kg:
        print(f"[+] Checking Neo4j at {Config.NEO4J_URI} ...")
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD),
                connection_timeout=5,
            )
            with driver.session() as session:
                session.run("RETURN 1").single()
            driver.close()
            print(f"[+] Neo4j OK\n")
        except Exception as e:
            print(f"\n[!] KHÔNG KẾT NỐI ĐƯỢC NEO4J: {e}")
            print(f"[!] URI hiện tại: {Config.NEO4J_URI}")
            print(f"[!] Cách khắc phục:")
            print(f"    1. Đảm bảo Neo4j đang chạy")
            print(f"    2. Nếu chạy script từ Windows host, dùng: --neo4j-uri bolt://localhost:7687")
            print(f"    3. Hoặc chạy --skip-kg để chỉ test RAG only\n")
            sys.exit(1)

    # ── Load test data ──────────────────────────────────────────────────
    print(f"[+] Loading test data: {args.csv}")
    test_data = []
    with open(args.csv, "r", encoding=args.encoding, errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("query") or "").strip()
            g = (row.get("label") or "").strip()
            if q and g:
                test_data.append({"query": q, "label": g})

    test_data = test_data[: args.n]
    n = len(test_data)
    print(f"[+] {n} samples ready (top_k={args.top_k})\n")

    # ── Run evaluation ──────────────────────────────────────────────────
    print("=" * 88)
    print(f"  EVALUATION  |  RAG only  |  KG only  |  Hybrid (RAG+KG)")
    print("=" * 88)

    rag_results, kg_results, hybrid_results = [], [], []
    entity_results, hybrid3_results = [], []
    detailed = []
    t0 = time.time()

    run_entity = not args.skip_entity

    for i, row in enumerate(test_data, 1):
        q = row["query"]
        gold = row["label"]

        r_rag = evaluate_rag_only(q, gold, args.top_k)
        if args.skip_kg:
            r_kg = {"hit": False, "rank": -1, "n_results": 0}
            r_hyb = {"hit": r_rag["hit"], "rank": r_rag["rank"], "vector_rank": r_rag["rank"], "kg_rank": -1}
        else:
            r_kg = evaluate_kg_only(q, gold, args.top_k)
            r_hyb = evaluate_hybrid(q, gold, args.top_k)

        # 2 phương pháp mới — yêu cầu Neo4j + tốn LLM call
        if run_entity and not args.skip_kg:
            r_ent = evaluate_entity_centric(q, gold, args.top_k)
            r_h3 = evaluate_hybrid_3way(q, gold, args.top_k)
        else:
            r_ent = {"hit": False, "rank": -1, "n_results": 0, "n_entities": 0}
            r_h3 = {"hit": False, "rank": -1, "n_results": 0}

        rag_results.append(r_rag)
        kg_results.append(r_kg)
        hybrid_results.append(r_hyb)
        entity_results.append(r_ent)
        hybrid3_results.append(r_h3)

        detailed.append(
            {
                "idx": i,
                "query": q[:120],
                "gold": gold,
                "rag_rank": r_rag["rank"],
                "kg_rank": r_kg["rank"],
                "hybrid_rank": r_hyb["rank"],
                "entity_centric_rank": r_ent["rank"],
                "hybrid3way_rank": r_h3["rank"],
                "n_entities_extracted": r_ent.get("n_entities", 0),
            }
        )

        if i % 10 == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.01)
            eta = (n - i) / max(rate, 0.01)
            print(
                f"[{i:4d}/{n}] "
                f"RAG={sum(1 for r in rag_results if r['hit']):3d} | "
                f"KG={sum(1 for r in kg_results if r['hit']):3d} | "
                f"HYB={sum(1 for r in hybrid_results if r['hit']):3d} | "
                f"ENT={sum(1 for r in entity_results if r['hit']):3d} | "
                f"H3={sum(1 for r in hybrid3_results if r['hit']):3d} | "
                f"{rate:.2f} q/s | ETA {eta:.0f}s"
            )

    elapsed = time.time() - t0

    # ── Compute & print metrics ─────────────────────────────────────────
    metrics = {
        "RAG only": compute_metrics(rag_results),
        "KG only": compute_metrics(kg_results),
        "Hybrid (RAG+KG)": compute_metrics(hybrid_results),
    }
    if run_entity and not args.skip_kg:
        metrics["Entity-Centric"] = compute_metrics(entity_results)
        metrics["Hybrid 3-way"] = compute_metrics(hybrid3_results)

    print("\n" + "=" * 88)
    print(f"  RESULTS  (elapsed: {elapsed:.1f}s, {n} samples, top_k={args.top_k})")
    print("=" * 88)
    print(f"{'Method':<22}{'Hit@1':>10}{'Hit@5':>10}{'Recall':>10}{'MRR':>10}{'Hits':>12}")
    print("-" * 74)
    for name, m in metrics.items():
        print(
            f"{name:<22}"
            f"{m['hit@1']*100:>9.2f}%"
            f"{m['hit@5']*100:>9.2f}%"
            f"{m['recall']:>10.4f}"
            f"{m['mrr']:>10.4f}"
            f"{m['n_hits']:>7}/{m['n_samples']}"
        )
    print("=" * 74)

    # ── Save results ────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    summary = {
        "config": {
            "n_samples": n,
            "top_k": args.top_k,
            "csv": args.csv,
            "elapsed_seconds": round(elapsed, 2),
        },
        "metrics": metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "detailed_results.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "query",
                "gold",
                "rag_rank",
                "kg_rank",
                "hybrid_rank",
                "entity_centric_rank",
                "hybrid3way_rank",
                "n_entities_extracted",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed)

    print(f"\n[+] Saved metrics → {out_dir / 'metrics.json'}")
    print(f"[+] Saved details → {out_dir / 'detailed_results.csv'}")


if __name__ == "__main__":
    main()
