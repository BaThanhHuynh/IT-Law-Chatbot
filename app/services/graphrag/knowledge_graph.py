"""
Knowledge Graph engine for hybrid search (vector + graph traversal).
Uses LangChain Neo4jGraph and Cypher queries.

Architecture:
  hybrid_search() is the SINGLE ENTRY POINT called by the chatbot engine.
  It orchestrates: multi-query vector search + KG entity search + graph traversal.
"""
import re
from app.core.config import Config
from app.core.logger import logger
from app.services.rag.retriever import multi_query_search
from app.services.rag.query_expansion import expand_abbreviations, get_domain_static_queries
from langchain_neo4j import Neo4jGraph


class KnowledgeGraph:
    """Knowledge graph loaded from Neo4j."""

    def __init__(self):
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = Neo4jGraph(
                url=Config.NEO4J_URI,
                username=Config.NEO4J_USERNAME,
                password=Config.NEO4J_PASSWORD,
                enhanced_schema=False,
                refresh_schema=False
            )
        return self._graph

    def search_entities(self, query: str, top_k: int = 5, min_score: float = 0.35) -> list:
        """
        Search entities by name/description using keyword match in Neo4j,
        then re-rank using real cosine similarity from the embedding model.
        
        Applies query expansion to handle abbreviations (CNTT, SHTT, etc.)
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            min_score: Minimum cosine similarity threshold to filter irrelevant entities
        """
        from app.services.rag.embeddings import get_embedding, cosine_similarity

        # Apply abbreviation expansion for broader keyword matching
        expanded_query = expand_abbreviations(query)
        
        # Combine words from both original and expanded queries for keyword search
        all_words = set()
        for text in [query, expanded_query]:
            all_words.update(w.lower() for w in text.split() if len(w) >= 2)
        words = list(all_words)
        
        if not words:
            return []

        # Step 1: Keyword-based candidate retrieval from Neo4j (cast a wider net)
        cypher = """
        MATCH (n:Entity)
        WHERE any(word in $words WHERE toLower(n.name) CONTAINS word OR toLower(n.description) CONTAINS word)
        RETURN n.entity_id AS entity_id, n.name AS name, n.description AS description, labels(n) AS labels
        LIMIT $top_k
        """
        
        results = self.graph.query(cypher, params={"words": words, "top_k": top_k * 3})
        
        if not results:
            return []

        # Step 2: Compute real cosine similarity between EXPANDED query and each entity
        # Use expanded query for embedding to match entity descriptions better
        query_embedding = get_embedding(expanded_query)

        scored = []
        for r in results:
            labels = [l for l in r.get("labels", []) if l != "Entity"]
            entity_type = labels[0] if labels else "UNKNOWN"
            
            # Build entity text = name + description for semantic comparison
            entity_text = f"{r.get('name', '')}. {r.get('description', '')}"
            entity_embedding = get_embedding(entity_text)
            
            # Real cosine similarity score
            sim_score = cosine_similarity(query_embedding, entity_embedding)
            
            # Filter out entities below the minimum relevance threshold
            if sim_score < min_score:
                continue

            scored.append({
                "entity": {
                    "entity_id": r["entity_id"],
                    "name": r["name"],
                    "description": r["description"],
                    "entity_type": entity_type
                },
                "score": round(sim_score, 3)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_graph_context(self, entity_ids: list, depth: int = 1) -> str:
        """
        Build context string from graph traversal starting from given entities.
        """
        if not entity_ids:
            return ""

        # Query up to 2 hops from the starting entities
        # Note: Cypher paths length can be dynamic but for simple RAG, depth 1-2 is enough.
        cypher = f"""
        MATCH (start:Entity)-[r*1..{depth}]-(target:Entity)
        WHERE start.entity_id IN $entity_ids
        RETURN start.name AS start_name, 
               [l IN labels(start) WHERE l <> 'Entity'][0] AS start_type,
               start.description AS start_desc,
               target.name AS target_name, 
               [l IN labels(target) WHERE l <> 'Entity'][0] AS target_type,
               type(r[-1]) AS rel_type
        LIMIT 50
        """
        
        results = self.graph.query(cypher, params={"entity_ids": entity_ids})
        
        if not results:
            # Maybe just fetch the nodes themselves if no relationships
            cypher_nodes = """
            MATCH (n:Entity) WHERE n.entity_id IN $entity_ids
            RETURN n.name AS name, [l IN labels(n) WHERE l <> 'Entity'][0] AS type, n.description AS desc
            """
            nodes = self.graph.query(cypher_nodes, params={"entity_ids": entity_ids})
            context = ""
            for n in nodes:
                context += f"[Entity: {n.get('name')}] (Loại: {n.get('type')})\n  {n.get('desc', '')[:200]}\n"
            return context

        context_parts = []
        seen_starts = set()
        seen_edges = set()

        for r in results:
            start_name = r["start_name"]
            if start_name not in seen_starts:
                seen_starts.add(start_name)
                context_parts.append(
                    f"[Entity: {start_name}] (Loại: {r['start_type']})\n"
                    f"  {r.get('start_desc', '')[:200]}"
                )
            
            edge_key = f"{start_name}-{r['rel_type']}-{r['target_name']}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                context_parts.append(
                    f"  → [{r['rel_type']}] {r['target_name']} (Loại: {r['target_type']})"
                )

        return "\n".join(context_parts)

    def get_graph_data_for_visualization(self, entity_ids: list = None, depth: int = 1) -> dict:
        """
        Get graph data (nodes + edges) for frontend visualization.
        """
        nodes_dict = {}
        edges_list = []

        if not entity_ids:
            # Default visualization: Fetch a subset of the graph
            cypher = """
            MATCH (n)-[r]->(m)
            WHERE ('VAN_BAN' IN labels(n) OR 'DIEU_LUAT' IN labels(n) OR 'CHUONG' IN labels(n))
            RETURN n.entity_id AS source_id, n.name AS source_name, [l IN labels(n) WHERE l <> 'Entity'][0] AS source_type,
                   m.entity_id AS target_id, m.name AS target_name, [l IN labels(m) WHERE l <> 'Entity'][0] AS target_type,
                   type(r) AS rel_type, r.description AS rel_desc
            LIMIT 50
            """
            results = self.graph.query(cypher)
        else:
            cypher = f"""
            MATCH (n)-[r*1..{depth}]-(m)
            WHERE n.entity_id IN $entity_ids
            WITH n, r[-1] as last_rel, m
            RETURN n.entity_id AS source_id, n.name AS source_name, [l IN labels(n) WHERE l <> 'Entity'][0] AS source_type,
                   m.entity_id AS target_id, m.name AS target_name, [l IN labels(m) WHERE l <> 'Entity'][0] AS target_type,
                   type(last_rel) AS rel_type, last_rel.description AS rel_desc
            LIMIT 100
            """
            results = self.graph.query(cypher, params={"entity_ids": entity_ids})

        for r in results:
            sid = r["source_id"]
            if sid not in nodes_dict:
                nodes_dict[sid] = {"id": sid, "label": r.get("source_name", "")[:50], "type": r.get("source_type")}
                
            tid = r["target_id"]
            if tid not in nodes_dict:
                nodes_dict[tid] = {"id": tid, "label": r.get("target_name", "")[:50], "type": r.get("target_type")}
                
            edges_list.append({
                "source": sid,
                "target": tid,
                "type": r["rel_type"],
                "label": r.get("rel_desc", "")[:30]
            })

        return {"nodes": list(nodes_dict.values()), "edges": edges_list}

# Singleton instance
_kg_instance = None

def get_knowledge_graph() -> KnowledgeGraph:
    """Get singleton KnowledgeGraph instance."""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance

def _build_entity_id(so_hieu: str, dieu_so: str) -> str:
    """Build Neo4j entity_id from so_hieu and dieu_so.
    
    VD: so_hieu='24/2018/QH14', dieu_so='7' → 'dieu_24_2018_QH14_7'
    """
    if not so_hieu or not dieu_so:
        return ""
    sanitized = re.sub(r'[/\-\s]', '_', so_hieu.strip())
    return f"dieu_{sanitized}_{dieu_so}"


def _expand_via_graph(vector_results: list, kg: "KnowledgeGraph", top_extra: int = 3) -> list:
    """
    Graph Re-ranking: sau khi RAG tìm được chunks, duyệt đồ thị để tìm các điều luật
    liên quan (tham chiếu, liên kết) và bổ sung vào kết quả.
    
    Chiến lược 2 lớp:
    Lớp 1 (Direct): Duyệt trực tiếp 1-2 hop từ entity_id của các điều đã tìm được
    Lớp 2 (Concept Bridge): Tìm các điều từ LUẬT KHÁC cùng AP_DUNG cho 1 khái niệm
           VD: Điều 7 ATTTM --AP_DUNG--> "phần mềm độc hại" <--AP_DUNG-- Điều 83 NĐ15
    """
    from app.services.rag.retriever import get_qdrant_client
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    # ── Bước 1: Xây dựng entity_id ĐÚNG FORMAT từ RAG results ────────
    article_entity_ids = []
    source_so_hieus = set()  # Track which laws we already have
    for vr in vector_results[:5]:
        dieu_so = vr.get("dieu_so", "")
        so_hieu = vr.get("so_hieu", "")
        if dieu_so and so_hieu:
            eid = _build_entity_id(so_hieu, dieu_so)
            if eid and eid not in article_entity_ids:
                article_entity_ids.append(eid)
                source_so_hieus.add(so_hieu)

    if not article_entity_ids:
        logger.info(f"[GraphExpand] No valid entity_ids from RAG results, skipping")
        return vector_results

    logger.info(f"[GraphExpand] Built entity_ids from RAG: {article_entity_ids[:5]}")

    # ── Bước 2: Lớp 1 — Duyệt trực tiếp (THUOC, LIEN_QUAN...) ──────
    related_entity_ids = []
    try:
        cypher_direct = """
        MATCH (start:Entity)-[r*1..2]-(related:Entity)
        WHERE start.entity_id IN $entity_ids
          AND NOT related.entity_id IN $entity_ids
          AND related.entity_id STARTS WITH 'dieu_'
        RETURN DISTINCT related.entity_id AS entity_id, related.name AS name
        LIMIT 10
        """
        results = kg.graph.query(cypher_direct, params={"entity_ids": article_entity_ids})
        related_entity_ids = [r["entity_id"] for r in results if r.get("entity_id")]
        logger.info(f"[GraphExpand] Layer 1 (Direct): Found {len(related_entity_ids)} related articles")
    except Exception as e:
        logger.warning(f"[GraphExpand] Layer 1 failed: {e}")

    # ── Bước 3: Lớp 2 — Concept Bridge (AP_DUNG xuyên luật) ─────────
    # Tìm các điều từ VĂN BẢN KHÁC cùng áp dụng cho 1 khái niệm
    try:
        cypher_bridge = """
        MATCH (start:Entity)-[:AP_DUNG]->(concept)<-[:AP_DUNG]-(related:Entity)
        WHERE start.entity_id IN $entity_ids
          AND NOT related.entity_id IN $entity_ids
          AND NOT related.entity_id IN $already_found
          AND related.entity_id STARTS WITH 'dieu_'
        RETURN DISTINCT related.entity_id AS entity_id, related.name AS name,
               concept.name AS via_concept
        LIMIT 10
        """
        bridge_results = kg.graph.query(cypher_bridge, params={
            "entity_ids": article_entity_ids,
            "already_found": related_entity_ids
        })
        
        for r in bridge_results:
            if r.get("entity_id") and r["entity_id"] not in related_entity_ids:
                related_entity_ids.append(r["entity_id"])
                logger.info(f"[GraphExpand] Layer 2 (Bridge): {r['entity_id']} via concept '{r.get('via_concept', '?')}'")
        
        logger.info(f"[GraphExpand] Layer 2 (Bridge): Found {len(bridge_results)} cross-law articles")
    except Exception as e:
        logger.warning(f"[GraphExpand] Layer 2 failed: {e}")

    if not related_entity_ids:
        logger.info(f"[GraphExpand] No related articles found in graph")
        return vector_results

    # ── Bước 4: Trích dieu_so từ entity_id ────────────────────────────
    # dieu_24_2018_QH14_7 → "7",  dieu_15_2020_ND_CP_83 → "83"
    related_dieu_so = []
    for eid in related_entity_ids:
        # Lấy số cuối cùng trong entity_id (đó là dieu_so)
        parts = eid.split('_')
        if len(parts) >= 2:
            dieu_so = parts[-1]
            if dieu_so.isdigit() and dieu_so not in related_dieu_so:
                related_dieu_so.append(dieu_so)

    if not related_dieu_so:
        return vector_results

    # ── Bước 5: Truy vấn Qdrant lấy nội dung các điều liên quan ──────
    try:
        qdrant = get_qdrant_client()
        existing_keys = {(r.get("doc_title", ""), r.get("dieu_so", "")) for r in vector_results}

        qdrant_results, _ = qdrant.scroll(
            collection_name=Config.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="dieu_so",
                        match=MatchAny(any=related_dieu_so)
                    )
                ]
            ),
            limit=top_extra * 3,
            with_payload=True,
        )

        added = 0
        for point in qdrant_results:
            p = point.payload
            key = (p.get("ten_van_ban", ""), p.get("dieu_so", ""))
            if key in existing_keys:
                continue
            existing_keys.add(key)

            article_str = f"Điều {p.get('dieu_so', '')}"
            if p.get("dieu_ten"):
                article_str += f". {p['dieu_ten']}"

            vector_results.append({
                "chunk_id": p.get("chunk_id"),
                "content": p.get("full_dieu_text") or p.get("noi_dung_chunk") or p.get("content", ""),
                "context_text": p.get("context_text", ""),
                "dieu_so": p.get("dieu_so", ""),
                "dieu_ten": p.get("dieu_ten", ""),
                "article": article_str,
                "doc_title": p.get("ten_van_ban", ""),
                "so_hieu": p.get("so_hieu", ""),
                "score": 0.0,
                "_source": "graph_expand",
            })
            added += 1
            if added >= top_extra:
                break

        logger.info(f"[GraphExpand] Injected {added} extra articles from graph into RAG context")
    except Exception as e:
        logger.warning(f"[GraphExpand] Qdrant fetch for related articles failed: {e}")

    return vector_results


def hybrid_search(query: str, sub_queries: list = None, entities: str = None, top_k: int = 5) -> dict:
    """
    Hybrid search: combine multi-query vector search + knowledge graph traversal.
    Returns both RAG chunks and graph context.
    
    This is the SINGLE ENTRY POINT for the chatbot pipeline.
    
    Strategy:
    - Vector search: Multi-query (original + LLM variants + expanded abbreviations)
    - Graph expand: Traverse graph to find related articles missed by vector search
    - Graph search:  Expanded entities (keyword match with abbreviation expansion)
    - Graph traversal: Expand context from matched entities + vector result entities
    
    Args:
        query: Original user query
        sub_queries: LLM-generated query variants (if None, falls back to single query)
        entities: Extracted entity keywords for graph keyword matching
        top_k: Number of results per search
    """
    # ── 1. Multi-query Vector Search ──────────────────────────────────
    # Layer 1: LLM-generated sub_queries (semantic diversity)
    all_queries = list(sub_queries) if sub_queries else [query]
    
    # Layer 2: Abbreviation-expanded version (lexical coverage)
    expanded = expand_abbreviations(query)
    if expanded != query and expanded not in all_queries:
        all_queries.append(expanded)
    
    # Layer 3: Domain static queries (rule-based, always consistent)
    static_queries = get_domain_static_queries(query)
    for sq in static_queries:
        if sq not in all_queries:
            all_queries.append(sq)
    
    logger.info(f"[HybridSearch] Total queries: {len(all_queries)} "
                f"(LLM:{len(sub_queries) if sub_queries else 1} "
                f"+ abbr:{1 if expanded != query else 0} "
                f"+ static:{len(static_queries)})")
    
    vector_results = multi_query_search(all_queries, top_k=top_k)

    # ── 1.5. Graph Re-ranking ─────────────────────────────────────────
    # Sau khi RAG tìm được top-k, duyệt đồ thị để bổ sung các điều luật
    # có liên hệ (tham chiếu, liên kết) mà vector search có thể bỏ sót.
    # VD: RAG tìm được "Điều 7 cấm hành vi" → Graph kéo thêm "Điều 84 mức phạt"
    kg = get_knowledge_graph()
    try:
        vector_results = _expand_via_graph(vector_results, kg, top_extra=3)
    except Exception as e:
        logger.warning(f"[GraphExpand] Skipped due to error: {e}")

    # ── 2. Knowledge Graph Entity Search ──────────────────────────────
    graph_search_term = entities if entities else query
    try:
        kg_results = kg.search_entities(graph_search_term, top_k=3)
    except Exception as e:
        logger.error(f"[KG Error] Entity search failed: {e}")
        kg_results = []

    # ── 3. Expand Graph Context ───────────────────────────────────────
    matched_entity_ids = [r["entity"]["entity_id"] for r in kg_results]

    # Also bridge vector results → graph entities (by article number)
    for vr in vector_results[:3]:
        article = vr.get("article", "")
        if article:
            art_match = re.search(r'(\d+)', article)
            if art_match:
                article_entity_id = f"dieu_{art_match.group(1)}"
                if article_entity_id not in matched_entity_ids:
                    matched_entity_ids.append(article_entity_id)

    try:
        if matched_entity_ids:
            graph_context = kg.get_graph_context(matched_entity_ids, depth=2)
            graph_data = kg.get_graph_data_for_visualization(matched_entity_ids, depth=1)
        else:
            graph_context = ""
            graph_data = {"nodes": [], "edges": []}
    except Exception as e:
        logger.error(f"[KG Error] Graph context failed: {e}")
        graph_context = ""
        graph_data = {"nodes": [], "edges": []}

    return {
        "vector_results": vector_results,
        "graph_context": graph_context,
        "graph_data": graph_data,
        "matched_entities": kg_results,
    }

