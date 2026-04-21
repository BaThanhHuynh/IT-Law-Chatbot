"""
Entity and relationship extractor for Vietnamese IT Law text.
Extracts knowledge graph triples from law articles.
"""
import re
from db import execute_query, fetch_all, fetch_one


# Entity types
ENTITY_TYPES = {
    "DIEU_LUAT": "Điều luật",
    "CHUONG": "Chương",
    "KHAI_NIEM": "Khái niệm",
    "CHU_THE": "Chủ thể",
    "HANH_VI": "Hành vi",
    "QUYEN": "Quyền",
    "NGHIA_VU": "Nghĩa vụ",
    "HINH_PHAT": "Hình phạt",
}

# Relationship types
REL_TYPES = {
    "THUOC": "thuộc về",
    "LIEN_QUAN": "liên quan đến",
    "AP_DUNG": "áp dụng cho",
    "NGHIEM_CAM": "nghiêm cấm",
    "DINH_NGHIA": "định nghĩa",
    "BAO_VE": "bảo vệ",
    "DIEU_CHINH": "điều chỉnh",
    "THAM_CHIEU": "tham chiếu đến",
}


def extract_entities_from_article(content: str, article_num: str = None, chunk_id: int = None) -> list:
    """Extract entities from a law article text."""
    entities = []

    # 1. Extract article entity
    if article_num:
        title_match = re.match(r'Điều\s+\d+[\.:]\s*(.*?)(?:\n|$)', content)
        title = title_match.group(1).strip() if title_match else ""
        entities.append({
            "entity_id": f"dieu_{article_num}",
            "name": f"Điều {article_num}" + (f". {title}" if title else ""),
            "entity_type": "DIEU_LUAT",
            "description": content[:500],
            "chunk_id": chunk_id,
        })

    # 2. Extract concepts (khái niệm) - words in quotes or defined terms
    concept_patterns = [
        r'"([^"]+)"',                     # Quoted terms
        r'"([^"]+)"',                     # Vietnamese quotes
        r'(?:là|được hiểu là|có nghĩa là)\s+(.+?)(?:\.|;|$)',  # Definitions
    ]
    for pattern in concept_patterns:
        for match in re.finditer(pattern, content):
            term = match.group(1).strip()
            if len(term) > 2 and len(term) < 100:
                concept_id = re.sub(r'\s+', '_', term.lower())
                concept_id = re.sub(r'[^\w]', '', concept_id)
                entities.append({
                    "entity_id": f"kn_{concept_id}",
                    "name": term,
                    "entity_type": "KHAI_NIEM",
                    "description": f"Khái niệm được đề cập trong Điều {article_num}" if article_num else "",
                    "chunk_id": chunk_id,
                })

    # 3. Extract subjects (chủ thể)
    subject_patterns = [
        r'(?:tổ chức|cá nhân|cơ quan|doanh nghiệp|người)',
        r'(?:Nhà nước|Chính phủ|Bộ|Sở)',
    ]
    found_subjects = set()
    for pattern in subject_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            subject = match.group(0).strip()
            if subject.lower() not in found_subjects:
                found_subjects.add(subject.lower())
                subject_id = re.sub(r'\s+', '_', subject.lower())
                entities.append({
                    "entity_id": f"ct_{subject_id}",
                    "name": subject,
                    "entity_type": "CHU_THE",
                    "description": f"Chủ thể pháp luật",
                    "chunk_id": chunk_id,
                })

    # 4. Extract prohibited actions (hành vi bị nghiêm cấm)
    if re.search(r'nghiêm cấm|không được|cấm', content, re.IGNORECASE):
        prohib_matches = re.findall(
            r'(?:nghiêm cấm|không được phép|cấm)\s+(.+?)(?:\.|;|\n)',
            content, re.IGNORECASE
        )
        for i, action in enumerate(prohib_matches):
            action = action.strip()
            if len(action) > 5:
                action_id = re.sub(r'\s+', '_', action[:50].lower())
                action_id = re.sub(r'[^\w]', '', action_id)
                entities.append({
                    "entity_id": f"hv_{action_id}",
                    "name": action[:200],
                    "entity_type": "HANH_VI",
                    "description": f"Hành vi bị nghiêm cấm theo Điều {article_num}" if article_num else "",
                    "chunk_id": chunk_id,
                })

    return entities


def extract_relationships(entities: list, content: str, article_num: str = None) -> list:
    """Extract relationships between entities."""
    relationships = []
    article_entity_id = f"dieu_{article_num}" if article_num else None

    for entity in entities:
        eid = entity["entity_id"]
        etype = entity["entity_type"]

        # All entities in an article belong to (THUOC) that article
        if article_entity_id and eid != article_entity_id:
            relationships.append({
                "source_entity_id": eid,
                "target_entity_id": article_entity_id,
                "relationship_type": "THUOC",
                "description": f"{entity['name']} thuộc {article_entity_id}",
            })

        # Prohibited actions → NGHIEM_CAM relationship
        if etype == "HANH_VI" and article_entity_id:
            for subj in entities:
                if subj["entity_type"] == "CHU_THE":
                    relationships.append({
                        "source_entity_id": article_entity_id,
                        "target_entity_id": eid,
                        "relationship_type": "NGHIEM_CAM",
                        "description": f"Luật nghiêm cấm: {entity['name'][:100]}",
                    })
                    break

        # Concepts → DINH_NGHIA relationship
        if etype == "KHAI_NIEM" and article_entity_id:
            relationships.append({
                "source_entity_id": article_entity_id,
                "target_entity_id": eid,
                "relationship_type": "DINH_NGHIA",
                "description": f"Điều {article_num} định nghĩa: {entity['name']}",
            })

    # Cross-article references (Điều X)
    if article_entity_id:
        ref_matches = re.findall(r'(?:theo|tại|quy định tại)\s+Điều\s+(\d+)', content)
        for ref_num in ref_matches:
            if ref_num != article_num:
                ref_id = f"dieu_{ref_num}"
                relationships.append({
                    "source_entity_id": article_entity_id,
                    "target_entity_id": ref_id,
                    "relationship_type": "THAM_CHIEU",
                    "description": f"Điều {article_num} tham chiếu đến Điều {ref_num}",
                })

    return relationships


def save_entities_to_db(entities: list):
    """Save extracted entities to MySQL, skip duplicates."""
    for entity in entities:
        existing = fetch_one(
            "SELECT id FROM kg_entities WHERE entity_id = %s",
            (entity["entity_id"],)
        )
        if not existing:
            execute_query(
                """INSERT INTO kg_entities (entity_id, name, entity_type, description, chunk_id)
                   VALUES (%s, %s, %s, %s, %s)""",
                (entity["entity_id"], entity["name"], entity["entity_type"],
                 entity.get("description", ""), entity.get("chunk_id"))
            )


def save_relationships_to_db(relationships: list):
    """Save extracted relationships to MySQL, skip duplicates."""
    for rel in relationships:
        # Verify both entities exist
        source = fetch_one("SELECT id FROM kg_entities WHERE entity_id = %s", (rel["source_entity_id"],))
        target = fetch_one("SELECT id FROM kg_entities WHERE entity_id = %s", (rel["target_entity_id"],))

        if source and target:
            existing = fetch_one(
                """SELECT id FROM kg_relationships
                   WHERE source_entity_id = %s AND target_entity_id = %s AND relationship_type = %s""",
                (rel["source_entity_id"], rel["target_entity_id"], rel["relationship_type"])
            )
            if not existing:
                execute_query(
                    """INSERT INTO kg_relationships
                       (source_entity_id, target_entity_id, relationship_type, description)
                       VALUES (%s, %s, %s, %s)""",
                    (rel["source_entity_id"], rel["target_entity_id"],
                     rel["relationship_type"], rel.get("description", ""))
                )
