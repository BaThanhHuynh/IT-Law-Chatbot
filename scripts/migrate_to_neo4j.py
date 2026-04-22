import os
import sys
from neo4j import GraphDatabase

# Thêm đường dẫn project vào sys.path để import từ config/db
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from config import Config
from db import fetch_all

def migrate_to_neo4j():
    """Đọc dữ liệu từ bảng kg_entities và kg_relationships trong MySQL và ghi vào Neo4j."""
    print("Đang kết nối đến MySQL...")
    entities = fetch_all("SELECT * FROM kg_entities")
    relationships = fetch_all("SELECT * FROM kg_relationships")
    print(f"Đã tải {len(entities)} entities và {len(relationships)} relationships từ MySQL.")

    print(f"Đang kết nối đến Neo4j tại {Config.NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
    except Exception as e:
        print(f"Lỗi kết nối Neo4j: {e}")
        print("Hãy chắc chắn rằng bạn đã khởi chạy Neo4j database (ví dụ: dùng docker-compose up -d neo4j).")
        return

    with driver.session() as session:
        # Xóa dữ liệu cũ
        print("Đang xóa dữ liệu cũ trong Neo4j...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # Thêm Entities (Nodes)
        print("Đang tạo Nodes...")
        for entity in entities:
            # entity_type sẽ làm Label, thêm label chung là Entity
            label = entity["entity_type"]
            query = f"""
            MERGE (n:`{label}` {{entity_id: $entity_id}})
            SET n.name = $name,
                n.description = $description,
                n.chunk_id = $chunk_id,
                n:Entity
            """
            session.run(query, {
                "entity_id": entity["entity_id"],
                "name": entity["name"],
                "description": entity.get("description", ""),
                "chunk_id": entity.get("chunk_id", None)
            })
            
        # Thêm Relationships (Edges)
        print("Đang tạo Relationships...")
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

        # Tạo Index cho tìm kiếm nhanh
        print("Đang tạo vector/text indexes...")
        try:
            session.run("CREATE TEXT INDEX entity_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            session.run("CREATE TEXT INDEX entity_desc_idx IF NOT EXISTS FOR (n:Entity) ON (n.description)")
        except Exception as e:
            print(f"Lưu ý khi tạo index: {e}")

    driver.close()
    print("Migration hoàn tất thành công!")

if __name__ == "__main__":
    migrate_to_neo4j()
