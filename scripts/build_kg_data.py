"""
scripts/build_kg_data.py
========================
Tự động phân tích toàn diện và chi tiết dữ liệu văn bản luật từ law_chunks_hier.jsonl
để tạo file Đồ thị Tri thức kg_data.json chuẩn cấu trúc GraphRAG nạp vào Neo4j.

Đảm bảo trích xuất:
1. 100% Văn bản quy phạm pháp luật (VAN_BAN) kèm số hiệu, loại văn bản, nhóm luật, hiệu lực.
2. 100% Chương (CHUONG) và Mục (MUC) có trong văn bản.
3. 100% Điều luật (DIEU_LUAT) với nội dung chi tiết.
4. Toàn bộ Khái niệm / Thuật ngữ pháp lý (KHAI_NIEM) được định nghĩa trong các điều "Giải thích từ ngữ".
5. Hệ thống quan hệ đầy đủ:
   - THUOC_VAN_BAN, THUOC_CHUONG, THUOC_MUC
   - DINH_NGHIA, THUOC_DIEU
   - NGHIEM_CAM (Các hành vi bị nghiêm cấm)
   - XU_PHAT (Xử lý vi phạm, chế tài)
   - THAM_CHIEU (Viện dẫn văn bản luật khác)
   - LIEN_QUAN (Viện dẫn các điều luật khác trong cùng văn bản)
"""

import os
import sys
import json
import re
import unicodedata
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def slugify(text: str) -> str:
    """Chuyển chuỗi tiếng Việt thành slug an toàn làm ID duy nhất."""
    if not text:
        return "unknown"
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "d")
    text = re.sub(r"[^\w\s-]", "_", text).strip().lower()
    text = re.sub(r"[-\s_]+", "_", text)
    return text[:60].strip("_")


def clean_text(text: str) -> str:
    """Làm sạch khoảng trắng thừa trong văn bản."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


def extract_definitions(full_text: str, vb_name: str, dieu_so: str):
    """
    Trích xuất chính xác các thuật ngữ và định nghĩa từ các Điều 'Giải thích từ ngữ' / 'Định nghĩa'.
    Hỗ trợ cả định dạng xuống dòng và định dạng số thứ tự liên tiếp (1. ... 2. ...).
    """
    definitions = []
    text = clean_text(full_text)
    
    # Pattern bao quát các kiểu: 1. Thuật ngữ là/được hiểu là ... hoặc a) Thuật ngữ là ...
    pattern = r"(?:^|(?<=\.\s)|(?<=\n))\s*(?:\d+[\.\)]|[a-z]\))\s*([A-ZÀ-Ỵa-zà-ỹ0-9\s\-_\(\)/,]+?)\s+(?:là|được hiểu là)\s+(.+?)(?=(?:\.\s+\d+[\.\)]|\.\s+[a-z]\)|\Z))"
    
    matches = re.finditer(pattern, text)
    for m in matches:
        term = m.group(1).strip().strip(":,.;-")
        definition = m.group(2).strip().strip(";,")
        
        # Tiêu chí lọc thuật ngữ hợp lệ
        if 2 <= len(term) <= 80 and len(definition) >= 15 and not term.lower().startswith("khoản") and not term.lower().startswith("điều"):
            definitions.append((term, definition))
            
    return definitions


def build_knowledge_graph(input_file: str, output_file: str):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        logger.error(f"❌ Không tìm thấy file input: {input_path}")
        return

    logger.info(f"📖 Đọc dữ liệu từ: {input_path}")
    
    van_ban_map = {}       # vb_key -> metadata dict
    chuong_map = {}        # (vb_key, chuong_so) -> chuong dict
    muc_map = {}           # (vb_key, chuong_so, muc_so) -> muc dict
    dieu_map = {}          # (vb_key, dieu_so) -> dieu dict
    
    total_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            p = item.get("payload", {})
            
            ten_vb = clean_text(p.get("ten_van_ban", ""))
            so_hieu = clean_text(p.get("so_hieu", ""))
            dieu_so = clean_text(str(p.get("dieu_so", "")))
            dieu_ten = clean_text(p.get("dieu_ten", ""))
            chuong_so = clean_text(str(p.get("chuong_so", "")))
            chuong_ten = clean_text(p.get("chuong_ten", ""))
            muc_so = clean_text(str(p.get("muc_so", "")))
            muc_ten = clean_text(p.get("muc_ten", ""))
            loai_vb = clean_text(p.get("loai_van_ban", "Luật"))
            nhom = clean_text(p.get("nhom", ""))
            ngay_hl = clean_text(p.get("ngay_hieu_luc", ""))
            ngay_het_hl = clean_text(p.get("ngay_het_hieu_luc", ""))
            full_text = clean_text(p.get("full_dieu_text") or p.get("noi_dung_chunk") or "")
            
            if not ten_vb or not dieu_so:
                continue
                
            vb_key = so_hieu if so_hieu else slugify(ten_vb)
            
            # 1. Thu thập Văn bản
            if vb_key not in van_ban_map:
                vb_id = f"vb_{slugify(vb_key)}"
                van_ban_map[vb_key] = {
                    "entity_id": vb_id,
                    "name": f"{ten_vb} ({so_hieu})" if so_hieu else ten_vb,
                    "ten_vb_clean": ten_vb,
                    "so_hieu": so_hieu,
                    "loai_van_ban": loai_vb,
                    "nhom": nhom,
                    "description": f"Văn bản quy phạm pháp luật: {ten_vb}. Loại: {loai_vb}. Số hiệu: {so_hieu}. Nhóm: {nhom}. Hiệu lực: {ngay_hl} đến {ngay_het_hl}."
                }
            
            # 2. Thu thập Chương
            if chuong_so and (vb_key, chuong_so) not in chuong_map:
                ch_id = f"chuong_{slugify(vb_key)}_{chuong_so}"
                ch_name = f"Chương {chuong_so}: {chuong_ten}" if chuong_ten else f"Chương {chuong_so}"
                chuong_map[(vb_key, chuong_so)] = {
                    "entity_id": ch_id,
                    "vb_key": vb_key,
                    "chuong_so": chuong_so,
                    "chuong_ten": chuong_ten,
                    "name": f"{ch_name} ({ten_vb})",
                    "description": f"{ch_name} thuộc văn bản {ten_vb}."
                }
            
            # 3. Thu thập Mục
            if muc_so and (vb_key, chuong_so, muc_so) not in muc_map:
                m_id = f"muc_{slugify(vb_key)}_{chuong_so}_{muc_so}"
                m_name = f"Mục {muc_so}: {muc_ten}" if muc_ten else f"Mục {muc_so}"
                muc_map[(vb_key, chuong_so, muc_so)] = {
                    "entity_id": m_id,
                    "vb_key": vb_key,
                    "chuong_so": chuong_so,
                    "muc_so": muc_so,
                    "name": f"{m_name} (Chương {chuong_so} - {ten_vb})",
                    "description": f"{m_name} thuộc Chương {chuong_so}, văn bản {ten_vb}."
                }

            # 4. Thu thập Điều luật
            dieu_key = (vb_key, dieu_so)
            if dieu_key not in dieu_map:
                d_id = f"dieu_{dieu_so}_{slugify(vb_key)}"
                dieu_map[dieu_key] = {
                    "entity_id": d_id,
                    "vb_key": vb_key,
                    "ten_vb": ten_vb,
                    "dieu_so": dieu_so,
                    "dieu_ten": dieu_ten,
                    "chuong_so": chuong_so,
                    "muc_so": muc_so,
                    "full_text": full_text
                }
            else:
                if len(full_text) > len(dieu_map[dieu_key]["full_text"]):
                    dieu_map[dieu_key]["full_text"] = full_text

    logger.info(f"📊 Đã tổng hợp đầy đủ từ {total_lines:,} chunks:")
    logger.info(f"   - Văn bản: {len(van_ban_map)}")
    logger.info(f"   - Chương: {len(chuong_map)}")
    logger.info(f"   - Mục: {len(muc_map)}")
    logger.info(f"   - Điều luật: {len(dieu_map)}")

    entities = []
    relationships = []
    
    seen_entity_ids = set()
    seen_rel_keys = set()
    
    def add_entity(e_id, name, desc, e_type):
        if e_id not in seen_entity_ids:
            seen_entity_ids.add(e_id)
            entities.append({
                "entity_id": e_id,
                "name": name,
                "description": desc,
                "entity_type": e_type
            })
            
    def add_rel(src_id, tgt_id, r_type, desc="", weight=1.0):
        key = (src_id, tgt_id, r_type)
        if key not in seen_rel_keys and src_id in seen_entity_ids and tgt_id in seen_entity_ids:
            seen_rel_keys.add(key)
            relationships.append({
                "source_entity_id": src_id,
                "target_entity_id": tgt_id,
                "relationship_type": r_type,
                "description": desc,
                "weight": weight
            })

    # 1. Thêm Nodes Văn bản
    for vb_key, vb in van_ban_map.items():
        add_entity(vb["entity_id"], vb["name"], vb["description"], "VAN_BAN")

    # 2. Thêm Nodes Chương & Mục
    for (vb_key, chuong_so), ch in chuong_map.items():
        vb_id = van_ban_map[vb_key]["entity_id"]
        add_entity(ch["entity_id"], ch["name"], ch["description"], "CHUONG")
        add_rel(ch["entity_id"], vb_id, "THUOC_VAN_BAN", f"Chương thuộc {van_ban_map[vb_key]['name']}")

    for (vb_key, chuong_so, muc_so), m in muc_map.items():
        if (vb_key, chuong_so) in chuong_map:
            ch_id = chuong_map[(vb_key, chuong_so)]["entity_id"]
            add_entity(m["entity_id"], m["name"], m["description"], "MUC")
            add_rel(m["entity_id"], ch_id, "THUOC_CHUONG", f"Mục thuộc Chương {chuong_so}")

    # 3. Thêm Nodes Điều luật & Quan hệ phân cấp
    for (vb_key, dieu_so), d in dieu_map.items():
        vb_id = van_ban_map[vb_key]["entity_id"]
        dieu_header = f"Điều {d['dieu_so']}. {d['dieu_ten']}" if d['dieu_ten'] else f"Điều {d['dieu_so']}"
        desc = f"{dieu_header} ({d['ten_vb']}): {d['full_text'][:1500]}"
        add_entity(d["entity_id"], dieu_header, desc, "DIEU_LUAT")
        
        # Điều -> Văn bản
        add_rel(d["entity_id"], vb_id, "THUOC_VAN_BAN", f"Điều thuộc {d['ten_vb']}")
        
        # Điều -> Chương
        if d["chuong_so"] and (vb_key, d["chuong_so"]) in chuong_map:
            ch_id = chuong_map[(vb_key, d["chuong_so"])]["entity_id"]
            add_rel(d["entity_id"], ch_id, "THUOC_CHUONG", f"Điều thuộc Chương {d['chuong_so']}")
            
        # Điều -> Mục
        if d["muc_so"] and (vb_key, d["chuong_so"], d["muc_so"]) in muc_map:
            m_id = muc_map[(vb_key, d["chuong_so"], d["muc_so"])]["entity_id"]
            add_rel(d["entity_id"], m_id, "THUOC_MUC", f"Điều thuộc Mục {d['muc_so']}")

    # 4. Trích xuất Khái niệm / Định nghĩa pháp lý
    logger.info("🔍 Trích xuất các khái niệm / thuật ngữ định nghĩa pháp lý...")
    total_definitions_extracted = 0
    for (vb_key, dieu_so), d in dieu_map.items():
        dieu_ten_lower = d["dieu_ten"].lower()
        if "giải thích" in dieu_ten_lower or "từ ngữ" in dieu_ten_lower or "định nghĩa" in dieu_ten_lower or dieu_so in ["2", "3", "4"]:
            defs = extract_definitions(d["full_text"], d["ten_vb"], d["dieu_so"])
            for term, def_text in defs:
                kn_id = f"kn_{slugify(term)}"
                kn_name = term.strip()
                kn_desc = f"Khái niệm '{kn_name}' theo Điều {d['dieu_so']} ({d['ten_vb']}): {def_text[:400]}"
                add_entity(kn_id, kn_name, kn_desc, "KHAI_NIEM")
                add_rel(d["entity_id"], kn_id, "DINH_NGHIA", f"Định nghĩa thuật ngữ '{kn_name}'", weight=1.0)
                add_rel(kn_id, d["entity_id"], "THUOC_DIEU", f"Quy định tại Điều {d['dieu_so']}", weight=1.0)
                total_definitions_extracted += 1

    logger.info(f"✨ Đã trích xuất {total_definitions_extracted} định nghĩa thuật ngữ.")

    # 5. Phân tích Quan hệ Pháp lý Đặc biệt & Tham chiếu chéo
    logger.info("🔗 Phân tích tham chiếu chéo giữa các điều luật và văn bản...")
    for (vb_key, dieu_so), d in dieu_map.items():
        src_id = d["entity_id"]
        full_text = d["full_text"]
        text_lower = full_text.lower()
        dieu_ten_lower = d["dieu_ten"].lower()
        vb_id = van_ban_map[vb_key]["entity_id"]
        
        # A. Hành vi bị nghiêm cấm
        if "nghiêm cấm" in dieu_ten_lower or "hành vi bị nghiêm cấm" in text_lower:
            add_rel(src_id, vb_id, "NGHIEM_CAM", f"Quy định hành vi bị nghiêm cấm trong {d['ten_vb']}", weight=1.2)

        # B. Xử lý vi phạm / Xử phạt
        if "xử lý vi phạm" in dieu_ten_lower or "xử phạt" in dieu_ten_lower or "chế tài" in text_lower:
            add_rel(src_id, vb_id, "XU_PHAT", f"Quy định về xử lý vi phạm trong {d['ten_vb']}", weight=1.1)

        # C. Tham chiếu Điều khác trong cùng văn bản
        internal_refs = set(re.findall(r"(?:quy định tại|theo|căn cứ|chiếu theo|áp dụng)\s+(?:khoản\s+\d+\s+)?Điều\s+(\d+)", full_text, re.IGNORECASE))
        for ref_dieu in internal_refs:
            if ref_dieu != dieu_so:
                target_key = (vb_key, ref_dieu)
                if target_key in dieu_map:
                    target_id = dieu_map[target_key]["entity_id"]
                    add_rel(src_id, target_id, "LIEN_QUAN", f"Tham chiếu Điều {ref_dieu} ({d['ten_vb']})", weight=0.9)

        # D. Tham chiếu sang Văn bản luật khác
        for other_vb_key, other_vb in van_ban_map.items():
            if other_vb_key != vb_key:
                clean_name = other_vb["ten_vb_clean"]
                so_hieu = other_vb["so_hieu"]
                if (so_hieu and so_hieu in full_text) or (clean_name and clean_name.lower() in text_lower):
                    add_rel(src_id, other_vb["entity_id"], "THAM_CHIEU", f"Tham chiếu đến {clean_name}", weight=1.0)

    # 6. Xuất kết quả
    kg_result = {
        "entities": entities,
        "relationships": relationships
    }

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kg_result, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Hoàn tất tạo Đồ thị Tri thức!")
    logger.info(f"📁 File xuất ra: {output_path}")
    logger.info(f"🌟 Tổng số Nodes (Entities): {len(entities):,}")
    logger.info(f"🔗 Tổng số Edges (Relationships): {len(relationships):,}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_input = str(project_root / "law_crawler" / "data" / "law_chunks_hier.jsonl")
    default_output = str(project_root / "law_crawler" / "data" / "kg_data.json")

    parser = argparse.ArgumentParser(description="Tạo file kg_data.json từ law_chunks_hier.jsonl cho Neo4j")
    parser.add_argument("--input", "-i", default=default_input, help="Đường dẫn file chunks jsonl")
    parser.add_argument("--output", "-o", default=default_output, help="Đường dẫn file kg_data.json đầu ra")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        alt_input = str(project_root / "law_crawler" / "data" / "law_chunks.jsonl")
        if os.path.exists(alt_input):
            logger.info(f"Dùng file thay thế: {alt_input}")
            args.input = alt_input

    build_knowledge_graph(args.input, args.output)


if __name__ == "__main__":
    main()
