"""
validate_data.py
================
Kiểm tra chất lượng dữ liệu sau khi crawl:
- Điều trống nội dung
- Điều chưa có metadata
- Phân bố độ dài chunk (cho RAG chunking strategy)
- Thống kê theo văn bản

DE: Lục Sỹ Minh Hiền  |  Project: Chatbot CNTT VN

Cách dùng:
    python validate_data.py --input ./data/law_data_output.xlsx
"""

import sys
import csv
import argparse
import logging
from pathlib import Path
from collections import defaultdict

try:
    import openpyxl
except ImportError:
    print("pip install openpyxl")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def read_excel(path: str) -> list[dict]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Dữ liệu luật"]
    rows = list(ws.values)
    headers = [str(h).strip() if h else "" for h in rows[0]]
    records = []
    for row in rows[1:]:
        if all(v is None or str(v).strip() == "" for v in row):
            continue
        rec = {headers[i]: (str(v).strip() if v is not None else "") for i, v in enumerate(row)}
        records.append(rec)
    return records


def validate(records: list[dict]):
    issues = []
    stats_by_file = defaultdict(lambda: {"count": 0, "empty_content": 0, "no_meta": 0, "lengths": []})

    for i, rec in enumerate(records, 1):
        sf = rec.get("Source File", "?")
        stats_by_file[sf]["count"] += 1

        noi_dung = rec.get("Nội dung điều", "").strip()
        do_dai = len(noi_dung)
        stats_by_file[sf]["lengths"].append(do_dai)

        # Kiểm tra nội dung trống
        if not noi_dung:
            stats_by_file[sf]["empty_content"] += 1
            issues.append(f"ROW {i}: Nội dung trống – {sf} Điều {rec.get('Điều số','?')}")

        # Kiểm tra metadata thiếu
        if not rec.get("Tên văn bản", "").strip():
            stats_by_file[sf]["no_meta"] += 1
            issues.append(f"ROW {i}: Thiếu metadata tên văn bản – {sf}")

        # Chunk quá ngắn (< 50 ký tự) → có thể parse sai
        if 0 < do_dai < 50:
            issues.append(f"ROW {i}: Chunk quá ngắn ({do_dai} ký tự) – {sf} Điều {rec.get('Điều số','?')}")

        # Chunk quá dài (> 3000 ký tự) → nên xem xét split thêm
        if do_dai > 3000:
            issues.append(f"ROW {i}: Chunk dài ({do_dai} ký tự, nên split) – {sf} Điều {rec.get('Điều số','?')}")

    # ── In báo cáo ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  BÁO CÁO KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
    print("="*65)
    print(f"\nTổng số điều:    {len(records):>6,}")
    print(f"Tổng số vấn đề: {len(issues):>6,}")

    print("\n── Thống kê theo văn bản ─────────────────────────────────────")
    print(f"{'Tên file':<40} {'Điều':>6} {'Trống':>6} {'NoMeta':>7} {'MaxLen':>8} {'AvgLen':>8}")
    print("-"*65)
    for sf, s in sorted(stats_by_file.items()):
        lengths = s["lengths"]
        avg = int(sum(lengths)/len(lengths)) if lengths else 0
        mx = max(lengths) if lengths else 0
        print(f"{sf[:40]:<40} {s['count']:>6} {s['empty_content']:>6} {s['no_meta']:>7} {mx:>8,} {avg:>8,}")

    if issues:
        print(f"\n── Danh sách vấn đề ({len(issues)}) ──────────────────────────────")
        # Hiển thị tối đa 30 issue
        for iss in issues[:30]:
            print(f"  ⚠ {iss}")
        if len(issues) > 30:
            print(f"  ... và {len(issues)-30} vấn đề khác")

    # Gợi ý chunking strategy cho RAG
    all_lengths = [len(r.get("Nội dung điều","")) for r in records if r.get("Nội dung điều","")]
    if all_lengths:
        all_lengths.sort()
        p50 = all_lengths[len(all_lengths)//2]
        p75 = all_lengths[int(len(all_lengths)*0.75)]
        p95 = all_lengths[int(len(all_lengths)*0.95)]
        print(f"\n── Phân bố độ dài nội dung (ký tự) ──────────────────────────")
        print(f"  Min: {all_lengths[0]:,}")
        print(f"  P50: {p50:,}  (median)")
        print(f"  P75: {p75:,}")
        print(f"  P95: {p95:,}")
        print(f"  Max: {all_lengths[-1]:,}")
        print(f"\n  💡 Gợi ý chunking cho PhoBERT (max 256 token ≈ 512 ký tự VN):")
        short = sum(1 for l in all_lengths if l <= 512)
        long  = sum(1 for l in all_lengths if l > 512)
        print(f"     Chunk ≤ 512 ký tự (dùng thẳng):  {short:,} điều ({short*100//len(all_lengths)}%)")
        print(f"     Chunk > 512 ký tự (cần split):   {long:,} điều ({long*100//len(all_lengths)}%)")
        print(f"\n  Khuyến nghị: chunk_size=400, overlap=50 ký tự")

    print("\n" + "="*65 + "\n")
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng dữ liệu")
    parser.add_argument("--input", "-i", required=True)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        log.error(f"Không tìm thấy: {path}")
        sys.exit(1)

    log.info(f"📖 Đọc: {path}")
    records = read_excel(str(path))
    log.info(f"   {len(records):,} dòng")

    ok = validate(records)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
