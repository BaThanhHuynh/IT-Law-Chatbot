import os
import sys
import json
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Thêm đường dẫn project vào đầu sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir in sys.path:
    sys.path.remove(project_dir)
sys.path.insert(0, project_dir)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.core.config import Config
from app.core.logger import logger

KG_DATA_PATH = os.path.join(project_dir, "law_crawler", "data", "kg_data.json")
MODEL_NAME = os.path.join(project_dir, "models", "law_v2_model_20260505_1418")


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
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình embedding: {e}")
        return

    neo4j_uri = Config.NEO4J_URI
    # Nếu chạy bên ngoài container Docker (Windows Host) nhưng URI là host.docker.internal thì chuyển thành localhost
    if "host.docker.internal" in neo4j_uri and not os.path.exists("/.dockerenv"):
        neo4j_uri = neo4j_uri.replace("host.docker.internal", "localhost")

    logger.info(f"Đang kết nối đến Neo4j tại {neo4j_uri}...")
    try:
        driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
    except Exception as e:
        logger.error(f"Lỗi kết nối Neo4j: {e}")
        return

    with driver.session() as session:
        # Thêm Entities (Nodes) và nhúng Vector theo Batch
        logger.info(f"Đang tạo {len(entities)} Nodes và tính toán Embeddings...")
        batch_size = 64
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            texts_to_embed = [f"{e['name']}. {e.get('description', '')}" for e in batch]
            with torch.no_grad():
                embeddings = model.encode(texts_to_embed, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False).tolist()

            for entity, emb in zip(batch, embeddings):
                label = entity["entity_type"]
                session.run(f"""
                MERGE (n:`{label}` {{entity_id: $entity_id}})
                SET n.name = $name,
                    n.description = $description,
                    n.embedding = $embedding,
                    n:Entity
                """, {
                    "entity_id": entity["entity_id"],
                    "name": entity["name"],
                    "description": entity.get("description", ""),
                    "embedding": emb
                })

        # Thêm Relationships (Edges) theo Batch UNWIND
        logger.info(f"Đang tạo {len(relationships)} Relationships...")
        rel_batch_size = 200
        for i in range(0, len(relationships), rel_batch_size):
            batch = relationships[i:i + rel_batch_size]
            rels_by_type = {}
            for r in batch:
                rtype = r["relationship_type"].replace(" ", "_").upper()
                if rtype not in rels_by_type:
                    rels_by_type[rtype] = []
                rels_by_type[rtype].append({
                    "source_id": r["source_entity_id"],
                    "target_id": r["target_entity_id"],
                    "description": r.get("description", ""),
                    "weight": r.get("weight", 1.0)
                })
            for rtype, r_list in rels_by_type.items():
                query = f"""
                UNWIND $batch AS rel
                MATCH (source {{entity_id: rel.source_id}})
                MATCH (target {{entity_id: rel.target_id}})
                MERGE (source)-[r:`{rtype}`]->(target)
                SET r.description = rel.description,
                    r.weight = rel.weight
                """
                session.run(query, {"batch": r_list})


        logger.info("Đang tạo indexes...")
        try:
            # Index văn bản thông thường
            session.run("CREATE TEXT INDEX entity_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            
            # Xóa index cũ nếu tồn tại để tránh xung đột kích thước (dimensions)
            try:
                session.run("DROP INDEX entity_embedding_idx IF EXISTS")
            except Exception as e:
                logger.info(f"Không thể xóa index cũ (có thể chưa tồn tại): {e}")

            # Index vector cho Neo4j với kích thước 768 tương ứng mô hình
            session.run("""
            CREATE VECTOR INDEX entity_embedding_idx IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 768,
              `vector.similarity_function`: 'cosine'
            }}
            """)
        except Exception as e:
            logger.warning(f"Lưu ý khi tạo index: {e}")

    driver.close()
    logger.info("Đồng bộ và nhúng vector vào Neo4j hoàn tất thành công!")

if __name__ == "__main__":
    migrate_to_neo4j()