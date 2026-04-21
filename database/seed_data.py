"""
Seed data script: Load law data from law_crawler's law_chunks.jsonl,
insert into MySQL, generate embeddings, and build knowledge graph.

Run: python database/seed_data.py
"""
import sys
import os
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mysql.connector
from config import Config

# Path to law_chunks.jsonl
JSONL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "law_crawler", "data", "law_chunks.jsonl"
)


def init_database():
    """Create database and tables from schema.sql."""
    print("[DB] Initializing database...")

    conn = mysql.connector.connect(
        host=Config.MYSQL_HOST,
        port=Config.MYSQL_PORT,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
    )
    cursor = conn.cursor()

    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    for statement in schema_sql.split(";"):
        statement = statement.strip()
        if statement:
            try:
                cursor.execute(statement)
            except mysql.connector.Error as e:
                if "already exists" not in str(e).lower():
                    print(f"  Warning: {e}")

    conn.commit()
    cursor.close()
    conn.close()
    print("[DB] Database initialized successfully.")


def load_jsonl(path: str) -> list:
    """Load law_chunks.jsonl into list of dicts."""
    print(f"[Load] Reading {path}...")
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[Load] Loaded {len(chunks):,} chunks")
    return chunks


def seed_law_data():
    """Main seeding function."""
    from db import execute_query, fetch_one, fetch_all

    # Check if JSONL file exists
    if not os.path.exists(JSONL_PATH):
        print(f"[ERROR] File not found: {JSONL_PATH}")
        print("  Please ensure law_crawler/data/law_chunks.jsonl exists.")
        sys.exit(1)

    # Load chunks from JSONL
    chunks = load_jsonl(JSONL_PATH)

    # Filter out repealed chunks (optional: keep them for reference)
    active_chunks = [c for c in chunks if not c.get("payload", {}).get("is_repealed", False)]
    print(f"[Filter] {len(active_chunks):,} active chunks (excluded {len(chunks) - len(active_chunks)} repealed)")

    # ── Step 1: Clear existing data ──────────────────────────────────────
    print("[Seed] Clearing existing data...")
    for table in ["messages", "conversations", "kg_relationships", "kg_entities",
                   "chunk_embeddings", "document_chunks", "law_documents"]:
        try:
            execute_query(f"DELETE FROM {table}")
        except Exception:
            pass

    # ── Step 2: Extract unique documents and insert ──────────────────────
    print("[Seed] Inserting law documents...")
    doc_map = {}  # so_hieu -> db_id
    seen_docs = {}

    for chunk in active_chunks:
        payload = chunk.get("payload", {})
        so_hieu = payload.get("so_hieu", "").strip()
        if not so_hieu or so_hieu in seen_docs:
            continue
        seen_docs[so_hieu] = payload

    for so_hieu, payload in seen_docs.items():
        doc_id = execute_query(
            """INSERT INTO law_documents
               (ten_van_ban, so_hieu, loai_van_ban, co_quan_ban_hanh,
                ngay_ban_hanh, ngay_hieu_luc, ngay_het_hieu_luc,
                trang_thai, nhom, source_file)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                payload.get("ten_van_ban", ""),
                so_hieu,
                payload.get("loai_van_ban", ""),
                "",  # co_quan_ban_hanh not in JSONL payload
                "",  # ngay_ban_hanh not in JSONL payload
                payload.get("ngay_hieu_luc", ""),
                payload.get("ngay_het_hieu_luc", ""),
                payload.get("trang_thai", ""),
                payload.get("nhom", ""),
                payload.get("source_file", ""),
            )
        )
        doc_map[so_hieu] = doc_id

    print(f"[Seed] Inserted {len(doc_map)} law documents")

    # ── Step 3: Insert chunks ────────────────────────────────────────────
    print("[Seed] Inserting document chunks...")
    chunk_db_ids = []  # (db_id, context_text) for embedding generation

    for i, chunk in enumerate(active_chunks):
        payload = chunk.get("payload", {})
        so_hieu = payload.get("so_hieu", "").strip()
        doc_id = doc_map.get(so_hieu)

        if not doc_id:
            continue

        db_chunk_id = execute_query(
            """INSERT INTO document_chunks
               (document_id, chunk_id, chunk_sub_index, chunk_total_sub,
                chunk_tier, content, context_text,
                chuong_so, chuong_ten, muc_so, muc_ten,
                dieu_so, dieu_ten, is_repealed, is_truncated, do_dai_chunk)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                doc_id,
                chunk.get("id", ""),
                payload.get("chunk_sub_index", 0),
                payload.get("chunk_total_sub", 1),
                payload.get("chunk_tier", 1),
                payload.get("noi_dung_chunk", ""),
                chunk.get("text", ""),  # context_text = text field in JSONL
                payload.get("chuong_so", ""),
                payload.get("chuong_ten", ""),
                payload.get("muc_so", ""),
                payload.get("muc_ten", ""),
                payload.get("dieu_so", ""),
                payload.get("dieu_ten", ""),
                payload.get("is_repealed", False),
                payload.get("is_truncated", False),
                len(payload.get("noi_dung_chunk", "")),
            )
        )
        chunk_db_ids.append((db_chunk_id, chunk.get("text", "")))

        if (i + 1) % 1000 == 0:
            print(f"  ... inserted {i + 1:,}/{len(active_chunks):,} chunks")

    print(f"[Seed] Inserted {len(chunk_db_ids):,} chunks")

    # ── Step 4: Generate embeddings ──────────────────────────────────────
    print("[Seed] Generating embeddings (this takes several minutes for ~9000 chunks)...")
    from rag.embeddings import get_embeddings_batch, serialize_embedding

    BATCH_SIZE = 256
    texts = [ctx for _, ctx in chunk_db_ids]
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]
        batch_ids = [chunk_db_ids[j][0] for j in range(start, end)]

        embeddings = get_embeddings_batch(batch_texts)

        for db_id, embedding in zip(batch_ids, embeddings):
            execute_query(
                "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (%s, %s)",
                (db_id, serialize_embedding(embedding)),
            )

        print(f"  ... batch {batch_idx + 1}/{total_batches} done ({end:,}/{len(texts):,} embeddings)")

    print(f"[Seed] Generated {len(chunk_db_ids):,} embeddings")

    # ── Step 5: Build Knowledge Graph ────────────────────────────────────
    print("[Seed] Building knowledge graph...")
    build_knowledge_graph(active_chunks, doc_map, chunk_db_ids)

    # ── Summary ──────────────────────────────────────────────────────────
    doc_count = fetch_one("SELECT COUNT(*) as cnt FROM law_documents")["cnt"]
    chunk_count = fetch_one("SELECT COUNT(*) as cnt FROM document_chunks")["cnt"]
    emb_count = fetch_one("SELECT COUNT(*) as cnt FROM chunk_embeddings")["cnt"]
    entity_count = fetch_one("SELECT COUNT(*) as cnt FROM kg_entities")["cnt"]
    rel_count = fetch_one("SELECT COUNT(*) as cnt FROM kg_relationships")["cnt"]

    print(f"\n{'='*60}")
    print(f"  ✅ Seed data completed!")
    print(f"  📄 Law documents:     {doc_count}")
    print(f"  📝 Chunks:            {chunk_count:,}")
    print(f"  🔢 Embeddings:        {emb_count:,}")
    print(f"  🔵 KG Entities:       {entity_count:,}")
    print(f"  🔗 KG Relationships:  {rel_count:,}")
    print(f"{'='*60}")


def build_knowledge_graph(chunks: list, doc_map: dict, chunk_db_ids: list):
    """Build knowledge graph from law chunks."""
    from db import execute_query, fetch_one
    import re

    entities = {}    # entity_id -> entity data
    relationships = []  # list of (source, target, type, desc)

    # ── Create document-level entities ───────────────────────────────────
    for so_hieu, payload in {c.get("payload", {}).get("so_hieu", ""): c.get("payload", {})
                              for c in chunks if c.get("payload", {}).get("so_hieu", "")}.items():
        doc_eid = f"doc_{so_hieu.replace('/', '_')}"
        entities[doc_eid] = {
            "entity_id": doc_eid,
            "name": payload.get("ten_van_ban", so_hieu),
            "entity_type": "VAN_BAN",
            "description": f"{payload.get('loai_van_ban', '')} số {so_hieu}",
            "properties": json.dumps({
                "so_hieu": so_hieu,
                "loai_van_ban": payload.get("loai_van_ban", ""),
                "trang_thai": payload.get("trang_thai", ""),
                "nhom": payload.get("nhom", ""),
            }, ensure_ascii=False),
        }

    # ── Create chapter & article entities + relationships ────────────────
    seen_chuong = set()
    seen_dieu = set()

    for chunk in chunks:
        payload = chunk.get("payload", {})
        so_hieu = payload.get("so_hieu", "").strip()
        if not so_hieu:
            continue

        doc_eid = f"doc_{so_hieu.replace('/', '_')}"
        chuong_so = payload.get("chuong_so", "").strip()
        chuong_ten = payload.get("chuong_ten", "").strip()
        dieu_so = payload.get("dieu_so", "").strip()
        dieu_ten = payload.get("dieu_ten", "").strip()
        content = payload.get("noi_dung_chunk", "")

        # Chapter entity
        if chuong_so:
            chuong_eid = f"chuong_{so_hieu.replace('/', '_')}_{chuong_so}"
            if chuong_eid not in seen_chuong:
                seen_chuong.add(chuong_eid)
                entities[chuong_eid] = {
                    "entity_id": chuong_eid,
                    "name": f"Chương {chuong_so}" + (f". {chuong_ten}" if chuong_ten else ""),
                    "entity_type": "CHUONG",
                    "description": f"Chương {chuong_so} của {payload.get('ten_van_ban', '')}",
                }
                relationships.append((chuong_eid, doc_eid, "THUOC", f"Chương {chuong_so} thuộc {so_hieu}"))

        # Article entity
        if dieu_so:
            dieu_eid = f"dieu_{so_hieu.replace('/', '_')}_{dieu_so}"
            if dieu_eid not in seen_dieu:
                seen_dieu.add(dieu_eid)
                entities[dieu_eid] = {
                    "entity_id": dieu_eid,
                    "name": f"Điều {dieu_so}" + (f". {dieu_ten}" if dieu_ten else ""),
                    "entity_type": "DIEU_LUAT",
                    "description": content[:500] if content else "",
                }
                # Article belongs to chapter
                if chuong_so:
                    chuong_eid = f"chuong_{so_hieu.replace('/', '_')}_{chuong_so}"
                    relationships.append((dieu_eid, chuong_eid, "THUOC", f"Điều {dieu_so} thuộc Chương {chuong_so}"))
                else:
                    relationships.append((dieu_eid, doc_eid, "THUOC", f"Điều {dieu_so} thuộc {so_hieu}"))

            # Extract cross-references: "theo Điều X", "tại Điều Y"
            ref_matches = re.findall(r'(?:theo|tại|quy định tại)\s+Điều\s+(\d+)', content)
            for ref_num in ref_matches:
                if ref_num != dieu_so:
                    # Try same document first
                    ref_eid = f"dieu_{so_hieu.replace('/', '_')}_{ref_num}"
                    relationships.append((dieu_eid, ref_eid, "THAM_CHIEU",
                                         f"Điều {dieu_so} tham chiếu Điều {ref_num}"))

            # Extract key concepts from content
            _extract_concepts(content, dieu_eid, dieu_so, entities, relationships)

    # ── Extract inter-document relationships ─────────────────────────────
    # Detect references to other laws: "Luật X số Y"
    for chunk in chunks:
        payload = chunk.get("payload", {})
        content = payload.get("noi_dung_chunk", "")
        so_hieu = payload.get("so_hieu", "").strip()
        dieu_so = payload.get("dieu_so", "").strip()
        if not content or not so_hieu or not dieu_so:
            continue

        dieu_eid = f"dieu_{so_hieu.replace('/', '_')}_{dieu_so}"

        # Find references to other documents by so_hieu pattern
        cross_refs = re.findall(r'(\d+/\d+/(?:QH\d+|NĐ-CP))', content)
        for ref_so_hieu in cross_refs:
            if ref_so_hieu != so_hieu:
                ref_doc_eid = f"doc_{ref_so_hieu.replace('/', '_')}"
                if ref_doc_eid in entities:
                    relationships.append((dieu_eid, ref_doc_eid, "THAM_CHIEU",
                                         f"Tham chiếu đến {ref_so_hieu}"))

    # ── Save to DB ───────────────────────────────────────────────────────
    print(f"  Saving {len(entities):,} entities...")
    for eid, entity in entities.items():
        try:
            execute_query(
                """INSERT INTO kg_entities (entity_id, name, entity_type, description, properties)
                   VALUES (%s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE name=VALUES(name)""",
                (entity["entity_id"], entity["name"], entity["entity_type"],
                 entity.get("description", ""), entity.get("properties")),
            )
        except Exception as e:
            pass  # Skip duplicates

    print(f"  Saving relationships...")
    saved_rels = 0
    for source, target, rel_type, desc in relationships:
        # Verify both entities exist
        src_exists = fetch_one("SELECT id FROM kg_entities WHERE entity_id = %s", (source,))
        tgt_exists = fetch_one("SELECT id FROM kg_entities WHERE entity_id = %s", (target,))
        if src_exists and tgt_exists:
            try:
                execute_query(
                    """INSERT IGNORE INTO kg_relationships
                       (source_entity_id, target_entity_id, relationship_type, description)
                       VALUES (%s, %s, %s, %s)""",
                    (source, target, rel_type, desc),
                )
                saved_rels += 1
            except Exception:
                pass

    print(f"  Saved {saved_rels:,} relationships")


def _extract_concepts(content: str, dieu_eid: str, dieu_so: str,
                      entities: dict, relationships: list):
    """Extract key legal concepts from article content."""
    import re

    # Prohibited actions
    if re.search(r'nghiêm cấm|không được|cấm', content, re.IGNORECASE):
        prohibitions = re.findall(
            r'(?:nghiêm cấm|không được phép|cấm)\s+(.+?)(?:\.|;)',
            content, re.IGNORECASE
        )
        for i, action in enumerate(prohibitions[:5]):  # limit
            action = action.strip()
            if len(action) > 10 and len(action) < 200:
                action_id = f"hv_{dieu_eid}_{i}"
                entities[action_id] = {
                    "entity_id": action_id,
                    "name": action[:200],
                    "entity_type": "HANH_VI",
                    "description": f"Hành vi bị nghiêm cấm theo Điều {dieu_so}",
                }
                relationships.append((dieu_eid, action_id, "NGHIEM_CAM",
                                     f"Nghiêm cấm: {action[:100]}"))

    # Defined terms (in quotes)
    quoted_terms = re.findall(r'"([^"]{3,80})"', content)
    for term in quoted_terms[:5]:
        term_id = re.sub(r'[^\w]', '_', term.lower().strip())[:50]
        concept_eid = f"kn_{term_id}"
        if concept_eid not in entities:
            entities[concept_eid] = {
                "entity_id": concept_eid,
                "name": term,
                "entity_type": "KHAI_NIEM",
                "description": f"Khái niệm pháp lý trong Điều {dieu_so}",
            }
        relationships.append((dieu_eid, concept_eid, "DINH_NGHIA",
                             f"Điều {dieu_so} định nghĩa: {term}"))

    # Key subjects
    subject_patterns = [
        (r'tổ chức, cá nhân', "tổ chức, cá nhân"),
        (r'cơ quan nhà nước', "cơ quan nhà nước"),
        (r'Bộ Thông tin và Truyền thông', "Bộ TT&TT"),
        (r'Chính phủ', "Chính phủ"),
        (r'người tiêu dùng', "người tiêu dùng"),
        (r'doanh nghiệp', "doanh nghiệp"),
    ]
    for pattern, name in subject_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            subject_eid = f"ct_{re.sub(r'[^a-z0-9]', '_', name.lower())}"
            if subject_eid not in entities:
                entities[subject_eid] = {
                    "entity_id": subject_eid,
                    "name": name,
                    "entity_type": "CHU_THE",
                    "description": "Chủ thể pháp luật",
                }
            relationships.append((dieu_eid, subject_eid, "AP_DUNG",
                                 f"Điều {dieu_so} áp dụng cho {name}"))


if __name__ == "__main__":
    init_database()
    seed_law_data()
