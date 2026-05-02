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


def hybrid_search(query: str, sub_queries: list = None, entities: str = None, top_k: int = 5) -> dict:
    """
    Hybrid search: combine multi-query vector search + knowledge graph traversal.
    Returns both RAG chunks and graph context.
    
    This is the SINGLE ENTRY POINT for the chatbot pipeline.
    
    Strategy:
    - Vector search: Multi-query (original + LLM variants + expanded abbreviations)
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
    # These guarantee specific laws are always retrieved for known topics
    # regardless of LLM variant quality (fixes non-deterministic retrieval)
    static_queries = get_domain_static_queries(query)
    for sq in static_queries:
        if sq not in all_queries:
            all_queries.append(sq)
    
    logger.info(f"[HybridSearch] Total queries: {len(all_queries)} "
                f"(LLM:{len(sub_queries) if sub_queries else 1} "
                f"+ abbr:{1 if expanded != query else 0} "
                f"+ static:{len(static_queries)})")
    
    vector_results = multi_query_search(all_queries, top_k=top_k)

    # ── 2. Knowledge Graph Entity Search ──────────────────────────────
    graph_search_term = entities if entities else query
    kg = get_knowledge_graph()
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
