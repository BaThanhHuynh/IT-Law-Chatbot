import os
import sys
import json
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Thêm đường dẫn project vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from app.core.config import Config
from app.core.logger import logger

KG_DATA_PATH = os.path.join(project_dir, "datab", "kg_data.json")
MODEL_NAME = r"C:\law_v2_model_20260505_1418" # Cập nhật mô hình mới

def migrate_to_neo4j():
    """Đọc dữ liệu từ file JSON, nhúng vector và ghi vào Neo4j."""
    
    if not os.path.exists(KG_DATA_PATH):
        logger.warning(f"Không tìm thấy file dữ liệu Graph tại {KG_DATA_PATH}")
        logger.info("Đang tạo file mẫu kg_data.json...")
        sample_data = {
            "entities": [
                {"entity_id": "dieu_1", "name": "Điều 1", "description": "Phạm vi điều chỉnh", "entity_type": "DIEU_LUAT"},
                {"entity_id": "luat_cntt", "name": "Luật CNTT 2006", "description": "Luật Công nghệ thông tin", "entity_type": "VAN_BAN"}
            ],
            "relationships": [
                {"source_entity_id": "dieu_1", "target_entity_id": "luat_cntt", "relationship_type": "THUOC", "description": "Thuộc văn bản", "weight": 1.0}
            ]
        }
        os.makedirs(os.path.dirname(KG_DATA_PATH), exist_ok=True)
        with open(KG_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Đã tạo file mẫu. Hãy cập nhật dữ liệu vào {KG_DATA_PATH} rồi chạy lại script.")
        return

    with open(KG_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    
    logger.info(f"Đã tải {len(entities)} entities và {len(relationships)} relationships từ JSON.")

    # KHỞI TẠO MÔ HÌNH EMBEDDING
    logger.info(f"Đang tải mô hình embedding: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình embedding: {e}")
        return

    logger.info(f"Đang kết nối đến Neo4j tại {Config.NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
    except Exception as e:
        logger.error(f"Lỗi kết nối Neo4j: {e}")
        return

    with driver.session() as session:
        # ĐÃ VÔ HIỆU HÓA LỆNH XÓA ĐỂ GIỮ LẠI DỮ LIỆU CŨ
        # logger.info("Đang xóa dữ liệu cũ trong Neo4j...")
        # session.run("MATCH (n) DETACH DELETE n")
        
        # Thêm Entities (Nodes) và nhúng Vector
        logger.info("Đang tạo Nodes và tính toán Embeddings...")
        for entity in entities:
            label = entity["entity_type"]
            name = entity["name"]
            description = entity.get("description", "")
            
            # Tạo nội dung để nhúng (kết hợp tên và mô tả)
            text_to_embed = f"{name}. {description}"
            # Chuyển đổi thành vector (đưa về dạng list để lưu vào Neo4j)
            embedding_vector = model.encode(text_to_embed, normalize_embeddings=True).tolist()

            query = f"""
            MERGE (n:`{label}` {{entity_id: $entity_id}})
            SET n.name = $name,
                n.description = $description,
                n.embedding = $embedding,
                n:Entity
            """
            session.run(query, {
                "entity_id": entity["entity_id"],
                "name": name,
                "description": description,
                "embedding": embedding_vector
            })
            
        # Thêm Relationships (Edges)
        logger.info("Đang tạo Relationships...")
        for rel in relationships:
            rel_type = rel["relationship_type"].replace(" ", "_").upper()
            query = f"""
            MATCH (source {{entity_id: $source_id}})
            MATCH (target {{entity_id: $target_id}})
            MERGE (source)-[r:`{rel_type}`]->(target)
            SET r.description = $description,
                r.weight = $weight
            """
            session.run(query, {
                "source_id": rel["source_entity_id"],
                "target_id": rel["target_entity_id"],
                "description": rel.get("description", ""),
                "weight": rel.get("weight", 1.0)
            })

        logger.info("Đang tạo indexes...")
        try:
            # Index văn bản thông thường
            session.run("CREATE TEXT INDEX entity_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            
            # Index vector cho Neo4j (kích thước 384 tương ứng mô hình)
            session.run("""
            CREATE VECTOR INDEX entity_embedding_idx IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 384,
              `vector.similarity_function`: 'cosine'
            }}
            """)
        except Exception as e:
            logger.warning(f"Lưu ý khi tạo index: {e}")

    driver.close()
    logger.info("Đồng bộ và nhúng vector vào Neo4j hoàn tất thành công!")

if __name__ == "__main__":
    migrate_to_neo4j()