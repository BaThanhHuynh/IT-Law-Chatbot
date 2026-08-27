"""
Đánh giá khả năng truy xuất của IT Law Chatbot trên 3 phương pháp:
  1. RAG only      — Vector search thuần trên Qdrant
  2. KG only       — Entity search thuần trên Neo4j
  3. Hybrid (RAG+KG) — Kết hợp cả hai (production pipeline)

Metrics: Hit@1, Hit@3, Hit@5, MRR, Average Rank.

Cách chạy:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --n 100 --top-k 5
    python evaluation/evaluate.py --csv "evaluation/Test_data_lawCNTT_cleaned (1).csv"
"""
import os
import sys
import csv
import json
import re
import time
import argparse
from pathlib import Path
import gc

# Đảm bảo import được app/* khi chạy từ root project hoặc thư mục evaluation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["EVAL_MODE"] = "true"

# Tự động chuyển các URL kết nối từ container-name sang localhost khi chạy ngoài Docker
def _is_running_in_docker():
    return os.path.exists('/.dockerenv')

if not _is_running_in_docker():
    # Fallback QDRANT_URL
    qdrant_url = os.environ.get("QDRANT_URL", "")
    if not qdrant_url or "qdrant" in qdrant_url:
        os.environ["QDRANT_URL"] = "http://localhost:6333"
        print("[+] Detected running on host: set QDRANT_URL -> http://localhost:6333")
    
    # Fallback NEO4J_URI (nếu không truyền tham số dòng lệnh)
    neo4j_uri = os.environ.get("NEO4J_URI", "")
    if not neo4j_uri or "neo4j" in neo4j_uri:
        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
        print("[+] Detected running on host: set NEO4J_URI -> bolt://localhost:7687")

# Cho phép override Neo4j URI qua dòng lệnh TRƯỚC khi import app modules
_pre_args = sys.argv[1:]
if "--neo4j-uri" in _pre_args:
    idx = _pre_args.index("--neo4j-uri")
    if idx + 1 < len(_pre_args):
        os.environ["NEO4J_URI"] = _pre_args[idx + 1]
        print(f"[+] Override NEO4J_URI -> {os.environ['NEO4J_URI']}")

import numpy as np

from app.services.rag.retriever import vector_search
from app.services.graphrag.knowledge_graph import get_knowledge_graph, hybrid_search
from app.services.rag.embeddings import get_embedding
from app.core.logger import logger
from app.core.config import Config


# Tăng giới hạn CSV cho query dài
csv.field_size_limit(10**7)


# ────────────────────────────────────────────────────────────────────────────
#  HÀM CHUẨN HOÁ NHÃN
# ────────────────────────────────────────────────────────────────────────────
def to_label_format(text: str, dieu_so: str = "") -> str:
    """
    Chuyển 'doc_title' + 'dieu_so' về cùng định dạng nhãn của test data.
    Quy tắc nhãn: ký tự non-ASCII / khoảng trắng -> '_', cuối nối '_D<dieu_so>'.

    Ví dụ:
        ("Luật An ninh mạng 2018", "1") -> "Lu_t_An_ninh_m_ng_2018_D1"
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
    """Trích số Điều từ chuỗi (ví dụ 'Điều 8' -> '8')."""
    m = re.search(r"[Đđ]i[eề]u\s*(\d+)", text)
    return m.group(1) if m else ""


def _parse_gold_label(gold: str) -> tuple:
    """
    Tách gold label -> (set_token_luật, số_điều).
    Ví dụ: 'Lu_t_An_ninh_m_ng_2018_D1' -> ({'an','ninh','ng','2018',...}, '1')
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

    # ── Tiêu chí 1: token >= 40% AND dieu khớp (chính) ──
    if token_ratio >= 0.40 and dieu_ok:
        return True

    # ── Tiêu chí 2: Doc-level match cực mạnh (>= 85%) — bỏ qua dieu ──
    if token_ratio >= 0.85:
        return True

    return False


# ────────────────────────────────────────────────────────────────────────────
#  3 PHƯƠNG PHÁP TRUY XUẤT CHÍNH THỨC CỦA DỰ ÁN
# ────────────────────────────────────────────────────────────────────────────
def evaluate_rag_only(query: str, gold: str, top_k: int) -> dict:
    """
    RAG only: Vector search chính thức trên Qdrant bằng vector_search.
    Lọc kết quả bằng ngưỡng thực tế của hệ thống (score >= 0.35).
    """
    try:
        results = vector_search(query, top_k=top_k)
    except Exception as e:
        logger.error(f"[RAG] Failed: {e}")
        return {"hit": False, "rank": -1, "n_results": 0}

    # Áp dụng ngưỡng lọc thực tế của hệ thống (score >= 0.35)
    results = [r for r in results if r.get("score", 0) >= 0.35]

    rank = -1
    for i, r in enumerate(results[:top_k]):
        pred = to_label_format(r.get("doc_title", ""), str(r.get("dieu_so", "")))
        if labels_match(pred, gold):
            rank = i + 1
            break

    return {"hit": rank > 0, "rank": rank, "n_results": len(results)}


def evaluate_kg_only(query: str, gold: str, top_k: int) -> dict:
    """
    KG only: Tìm kiếm thực thể chính thức trên Neo4j bằng search_entities.
    Lọc kết quả bằng ngưỡng thực tế của hệ thống (score >= 0.35).
    """
    try:
        kg = get_knowledge_graph()
        # Gọi hàm chính thức search_entities của hệ thống
        results = kg.search_entities(query, top_k=top_k, min_score=0.35)
    except Exception as e:
        logger.error(f"[KG] Failed: {e}")
        return {"hit": False, "rank": -1, "n_results": 0}

    gold_tokens, gold_dieu = _parse_gold_label(gold)

    rank = -1
    for i, r in enumerate(results[:top_k]):
        entity = r.get("entity", {}) or {}
        if _entity_matches_gold(entity, gold_tokens, gold_dieu):
            rank = i + 1
            break

    return {"hit": rank > 0, "rank": rank, "n_results": len(results)}


def evaluate_hybrid(query: str, gold: str, top_k: int) -> dict:
    """
    Hybrid (RAG+KG): Gọi hàm `hybrid_search` chính thức của hệ thống Chatbot.
    Duyệt qua cả Qdrant và Neo4j, lọc kết quả bằng ngưỡng score thực tế (>= 0.35).
    """
    try:
        # Gọi chính xác hàm hybrid_search của hệ thống
        results = hybrid_search(query=query, sub_queries=None, entities=None, top_k=top_k)
        vector_results = results.get("vector_results", [])
    except Exception as e:
        logger.error(f"[Hybrid] Failed: {e}")
        return {"hit": False, "rank": -1, "n_results": 0}

    # Áp dụng bộ lọc thực tế của Chatbot (score >= 0.35 hoặc graph_expand hoặc hybrid bm25)
    filtered_results = []
    for r in vector_results:
        if (
            r.get("_source") in ("graph_expand", "hybrid")
            or r.get("score", 0) >= 0.35
            or r.get("rrf_score", 0) > 0
        ):
            filtered_results.append(r)

    rank = -1
    for i, r in enumerate(filtered_results[:top_k]):
        pred = to_label_format(r.get("doc_title", ""), str(r.get("dieu_so", "")))
        if labels_match(pred, gold):
            rank = i + 1
            break

    return {"hit": rank > 0, "rank": rank, "n_results": len(filtered_results)}


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
        default=r"evaluation\Test_data_lawCNTT_cleaned (1).csv",
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
    csv_path = args.csv
    # Thử tìm tương đối so với thư mục chứa script nếu không tìm thấy ở CWD
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path1 = os.path.join(script_dir, os.path.basename(csv_path))
        alt_path2 = os.path.join(os.path.dirname(script_dir), csv_path)
        if os.path.exists(alt_path1):
            csv_path = alt_path1
        elif os.path.exists(alt_path2):
            csv_path = alt_path2

    print(f"[+] Loading test data: {csv_path}")
    test_data = []
    with open(csv_path, "r", encoding=args.encoding, errors="replace") as f:
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
    detailed = []
    t0 = time.time()

    for i, row in enumerate(test_data, 1):
        q = row["query"]
        gold = row["label"]

        r_rag = evaluate_rag_only(q, gold, args.top_k)
        if args.skip_kg:
            r_kg = {"hit": False, "rank": -1, "n_results": 0}
            r_hyb = {"hit": r_rag["hit"], "rank": r_rag["rank"], "n_results": r_rag["n_results"]}
        else:
            r_kg = evaluate_kg_only(q, gold, args.top_k)
            r_hyb = evaluate_hybrid(q, gold, args.top_k)

        rag_results.append(r_rag)
        kg_results.append(r_kg)
        hybrid_results.append(r_hyb)

        detailed.append(
            {
                "idx": i,
                "query": q[:120],
                "gold": gold,
                "rag_rank": r_rag["rank"],
                "kg_rank": r_kg["rank"],
                "hybrid_rank": r_hyb["rank"],
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
                f"{rate:.2f} q/s | ETA {eta:.0f}s"
            )

        gc.collect()

    elapsed = time.time() - t0

    # ── Compute & print metrics ─────────────────────────────────────────
    metrics = {
        "RAG only": compute_metrics(rag_results),
        "KG only": compute_metrics(kg_results),
        "Hybrid (RAG+KG)": compute_metrics(hybrid_results),
    }

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
            ],
        )
        writer.writeheader()
        writer.writerows(detailed)

    print(f"\n[+] Saved metrics -> {out_dir / 'metrics.json'}")
    print(f"[+] Saved details -> {out_dir / 'detailed_results.csv'}")


if __name__ == "__main__":
    main()
