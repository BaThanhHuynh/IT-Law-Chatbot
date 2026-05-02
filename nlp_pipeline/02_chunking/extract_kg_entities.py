import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def extract_entities_and_relationships(jsonl_path: str, output_path: str):
    if not os.path.exists(jsonl_path):
        logger.error(f"Không tìm thấy file: {jsonl_path}")
        return

    logger.info(f"Đọc dữ liệu từ {jsonl_path}...")
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    entities_dict = {}
    relationships_set = set()

    def add_entity(e_id, e_name, e_desc, e_type):
        if e_id not in entities_dict:
            entities_dict[e_id] = {
                "entity_id": e_id,
                "name": e_name,
                "description": e_desc,
                "entity_type": e_type
            }

    def add_relationship(src, tgt, r_type, desc=""):
        rel = (src, tgt, r_type, desc)
        relationships_set.add(rel)

    for chunk in chunks:
        p = chunk.get("payload", {})
        chunk_id = p.get("chunk_id", chunk.get("id"))
        
        # Lấy metadata
        ten_van_ban = p.get("ten_van_ban", "")
        so_hieu = p.get("so_hieu", "")
        chuong_so = p.get("chuong_so", "")
        chuong_ten = p.get("chuong_ten", "")
        dieu_so = p.get("dieu_so", "")
        dieu_ten = p.get("dieu_ten", "")

        # 1. Thực thể Văn bản
        vb_id = ""
        if ten_van_ban:
            vb_id = f"VB_{so_hieu}" if so_hieu else f"VB_{hash(ten_van_ban)}"
            add_entity(vb_id, ten_van_ban, f"Văn bản: {ten_van_ban}", "VAN_BAN")

        # 2. Thực thể Chương
        chuong_id = ""
        if chuong_so and vb_id:
            chuong_id = f"{vb_id}_CH_{chuong_so}"
            add_entity(chuong_id, f"Chương {chuong_so}", chuong_ten, "CHUONG")
            add_relationship(chuong_id, vb_id, "THUOC_VAN_BAN", "Nằm trong văn bản")

        # 3. Thực thể Điều
        dieu_id = ""
        if dieu_so and vb_id:
            dieu_id = f"{vb_id}_DIEU_{dieu_so}"
            add_entity(dieu_id, f"Điều {dieu_so}", dieu_ten, "DIEU_LUAT")
            
            if chuong_id:
                add_relationship(dieu_id, chuong_id, "THUOC_CHUONG", "Nằm trong chương")
            else:
                add_relationship(dieu_id, vb_id, "THUOC_VAN_BAN", "Nằm trong văn bản")

        # 4. Thực thể Chunk (Đoạn trích)
        if chunk_id:
            c_id = f"CHUNK_{chunk_id}"
            add_entity(c_id, f"Đoạn {chunk_id}", p.get("noi_dung_chunk", "")[:100] + "...", "DOAN_TRICH")
            
            if dieu_id:
                add_relationship(c_id, dieu_id, "THUOC_DIEU", "Trích từ điều luật")
            elif chuong_id:
                add_relationship(c_id, chuong_id, "THUOC_CHUONG", "Trích từ chương")
            elif vb_id:
                add_relationship(c_id, vb_id, "THUOC_VAN_BAN", "Trích từ văn bản")

    # Format output
    output_data = {
        "entities": list(entities_dict.values()),
        "relationships": [
            {
                "source_entity_id": r[0],
                "target_entity_id": r[1],
                "relationship_type": r[2],
                "description": r[3],
                "weight": 1.0
            } for r in relationships_set
        ]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Trích xuất thành công {len(output_data['entities'])} entities và {len(output_data['relationships'])} relationships.")
    logger.info(f"Đã lưu Graph data tại: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Ưu tiên lấy file đã có Hierarchical / Parent-Child Chunking
    jsonl_path = os.path.join(project_root, "nlp_pipeline", "data", "law_chunks.jsonl")
    if not os.path.exists(jsonl_path): # Dự phòng nếu chưa chạy lệnh tổ chức folder
         jsonl_path = os.path.join(project_root, "law_crawler", "data", "law_chunks.jsonl")
         
    output_path = os.path.join(project_root, "data", "kg_data.json")
    
    extract_entities_and_relationships(jsonl_path, output_path)
