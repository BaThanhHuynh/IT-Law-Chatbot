"""
Knowledge Graph engine for hybrid search (vector + graph traversal).
Loads graph from MySQL and enables multi-hop reasoning.
"""
from collections import defaultdict
from db import fetch_all
from rag.retriever import vector_search
from rag.embeddings import get_embedding, cosine_similarity, deserialize_embedding


class KnowledgeGraph:
    """In-memory knowledge graph loaded from MySQL."""

    def __init__(self):
        self.entities = {}          # entity_id -> entity data
        self.adjacency = defaultdict(list)  # entity_id -> [(target_id, rel_type, desc)]
        self.reverse_adj = defaultdict(list)  # reverse adjacency
        self._loaded = False

    def load_from_db(self):
        """Load the full knowledge graph from MySQL."""
        # Load entities
        entity_rows = fetch_all("SELECT * FROM kg_entities")
        for row in entity_rows:
            self.entities[row["entity_id"]] = {
                "id": row["id"],
                "entity_id": row["entity_id"],
                "name": row["name"],
                "entity_type": row["entity_type"],
                "description": row.get("description", ""),
                "chunk_id": row.get("chunk_id"),
            }

        # Load relationships
        rel_rows = fetch_all("SELECT * FROM kg_relationships")
        for row in rel_rows:
            self.adjacency[row["source_entity_id"]].append({
                "target": row["target_entity_id"],
                "type": row["relationship_type"],
                "description": row.get("description", ""),
                "weight": row.get("weight", 1.0),
            })
            self.reverse_adj[row["target_entity_id"]].append({
                "target": row["source_entity_id"],
                "type": row["relationship_type"],
                "description": row.get("description", ""),
                "weight": row.get("weight", 1.0),
            })

        self._loaded = True
        print(f"[KG] Loaded {len(self.entities)} entities and {len(rel_rows)} relationships.")

    def ensure_loaded(self):
        """Ensure graph is loaded."""
        if not self._loaded:
            self.load_from_db()

    def get_entity(self, entity_id: str) -> dict:
        """Get entity by ID."""
        self.ensure_loaded()
        return self.entities.get(entity_id)

    def get_related_entities(self, entity_id: str, depth: int = 2) -> list:
        """
        Multi-hop traversal: get all entities within N hops.
        Returns list of (entity, relationship_type, distance) tuples.
        """
        self.ensure_loaded()
        visited = set()
        results = []
        queue = [(entity_id, 0, None)]

        while queue:
            current_id, current_depth, rel_type = queue.pop(0)

            if current_id in visited or current_depth > depth:
                continue
            visited.add(current_id)

            entity = self.entities.get(current_id)
            if entity and current_depth > 0:
                results.append({
                    "entity": entity,
                    "relationship": rel_type,
                    "distance": current_depth,
                })

            if current_depth < depth:
                # Forward edges
                for edge in self.adjacency.get(current_id, []):
                    if edge["target"] not in visited:
                        queue.append((edge["target"], current_depth + 1, edge["type"]))
                # Reverse edges
                for edge in self.reverse_adj.get(current_id, []):
                    if edge["target"] not in visited:
                        queue.append((edge["target"], current_depth + 1, edge["type"]))

        return results

    def search_entities(self, query: str, top_k: int = 5) -> list:
        """Search entities by name similarity."""
        self.ensure_loaded()
        query_lower = query.lower()

        scored = []
        for eid, entity in self.entities.items():
            name_lower = entity["name"].lower()
            desc_lower = (entity.get("description") or "").lower()

            # Simple keyword match scoring
            score = 0
            for word in query_lower.split():
                if len(word) < 2:
                    continue
                if word in name_lower:
                    score += 3
                if word in desc_lower:
                    score += 1

            if score > 0:
                scored.append({"entity": entity, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_graph_context(self, entity_ids: list, depth: int = 1) -> str:
        """
        Build context string from graph traversal starting from given entities.
        """
        self.ensure_loaded()
        context_parts = []
        seen = set()

        for eid in entity_ids:
            entity = self.get_entity(eid)
            if not entity:
                continue

            if eid not in seen:
                seen.add(eid)
                context_parts.append(
                    f"[Entity: {entity['name']}] (Loại: {entity['entity_type']})\n"
                    f"  {entity.get('description', '')[:200]}"
                )

            related = self.get_related_entities(eid, depth=depth)
            for r in related:
                rel_entity = r["entity"]
                if rel_entity["entity_id"] not in seen:
                    seen.add(rel_entity["entity_id"])
                    context_parts.append(
                        f"  → [{r['relationship']}] {rel_entity['name']} "
                        f"(Loại: {rel_entity['entity_type']})"
                    )

        return "\n".join(context_parts)

    def get_graph_data_for_visualization(self, entity_ids: list = None, depth: int = 1) -> dict:
        """
        Get graph data (nodes + edges) for frontend visualization.
        """
        self.ensure_loaded()
        nodes = []
        edges = []
        node_ids = set()

        if entity_ids is None:
            # Return all VAN_BAN + DIEU_LUAT entities with their direct connections
            entity_ids = [
                eid for eid, e in self.entities.items()
                if e["entity_type"] in ("VAN_BAN", "DIEU_LUAT", "CHUONG")
            ][:50]  # Limit for performance

        for eid in entity_ids:
            entity = self.get_entity(eid)
            if entity and eid not in node_ids:
                node_ids.add(eid)
                nodes.append({
                    "id": eid,
                    "label": entity["name"][:50],
                    "type": entity["entity_type"],
                })

            for edge in self.adjacency.get(eid, []):
                target = self.get_entity(edge["target"])
                if target and edge["target"] not in node_ids:
                    node_ids.add(edge["target"])
                    nodes.append({
                        "id": edge["target"],
                        "label": target["name"][:50],
                        "type": target["entity_type"],
                    })
                if target:
                    edges.append({
                        "source": eid,
                        "target": edge["target"],
                        "type": edge["type"],
                        "label": edge.get("description", "")[:30],
                    })

        return {"nodes": nodes, "edges": edges}


# Singleton instance
_kg_instance = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get singleton KnowledgeGraph instance."""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance


def hybrid_search(query: str, top_k: int = 5) -> dict:
    """
    Hybrid search: combine vector search + knowledge graph traversal.
    Returns both RAG chunks and graph context.
    """
    # 1. Vector search for relevant chunks
    vector_results = vector_search(query, top_k=top_k)

    # 2. Knowledge graph search for related entities
    kg = get_knowledge_graph()
    kg_results = kg.search_entities(query, top_k=3)

    # 3. Expand graph context from matched entities
    matched_entity_ids = [r["entity"]["entity_id"] for r in kg_results]

    # Also find entities linked to vector search results
    for vr in vector_results[:3]:
        article = vr.get("article", "")
        if article:
            import re
            art_match = re.search(r'(\d+)', article)
            if art_match:
                article_entity_id = f"dieu_{art_match.group(1)}"
                if article_entity_id not in matched_entity_ids:
                    matched_entity_ids.append(article_entity_id)

    graph_context = kg.get_graph_context(matched_entity_ids, depth=2)
    graph_data = kg.get_graph_data_for_visualization(matched_entity_ids, depth=1)

    return {
        "vector_results": vector_results,
        "graph_context": graph_context,
        "graph_data": graph_data,
        "matched_entities": kg_results,
    }
