"""
Knowledge Graph engine for hybrid search (vector + graph traversal).
Uses LangChain Neo4jGraph and Cypher queries.

Architecture:
  hybrid_search() is the SINGLE ENTRY POINT called by the chatbot engine.
  It orchestrates: multi-query vector search + KG entity search + graph traversal.
"""
import re
from concurrent.futures import ThreadPoolExecutor
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
        Search entities by name/description using Neo4j Vector Index (entity_embedding_idx),
        falling back to keyword match with RAM-based cosine similarity if vector search fails.
        
        Applies query expansion to handle abbreviations (CNTT, SHTT, etc.)
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            min_score: Minimum cosine similarity threshold to filter irrelevant entities
        """
        from app.services.rag.embeddings import get_embedding, cosine_similarity

        # Apply abbreviation expansion for broader matching
        expanded_query = expand_abbreviations(query)
        query_embedding = get_embedding(expanded_query)

        # Step 1: Attempt Neo4j Vector Search
        cypher_vector = """
        CALL db.index.vector.queryNodes('entity_embedding_idx', $top_k, $query_embedding)
        YIELD node, score
        RETURN node.entity_id AS entity_id, node.name AS name, 
               node.description AS description, labels(node) AS labels, score
        """
        try:
            results = self.graph.query(cypher_vector, params={
                "query_embedding": query_embedding.tolist(),
                "top_k": top_k
            })
            
            if results:
                scored = []
                for r in results:
                    # Neo4j vector search cosine score is mapped, filter by min_score
                    score = r.get("score", 0.0)
                    if score < min_score:
                        continue
                    
                    labels = [l for l in r.get("labels", []) if l != "Entity"]
                    entity_type = labels[0] if labels else "UNKNOWN"
                    scored.append({
                        "entity": {
                            "entity_id": r["entity_id"],
                            "name": r["name"],
                            "description": r["description"],
                            "entity_type": entity_type
                        },
                        "score": round(score, 3)
                    })
                return scored
        except Exception as e:
            logger.warning(f"Neo4j Vector Search failed, falling back to keyword search: {e}")

        # Step 2: Fallback to keyword-based candidate retrieval + RAM-based cosine similarity
        all_words = set()
        for text in [query, expanded_query]:
            all_words.update(w.lower() for w in text.split() if len(w) >= 2)
        words = list(all_words)
        
        if not words:
            return []

        cypher_keyword = """
        MATCH (n:Entity)
        WHERE any(word in $words WHERE toLower(n.name) CONTAINS word OR toLower(n.description) CONTAINS word)
        RETURN n.entity_id AS entity_id, n.name AS name, n.description AS description, labels(n) AS labels
        LIMIT $limit
        """
        
        results = self.graph.query(cypher_keyword, params={"words": words, "limit": top_k * 3})
        if not results:
            return []

        # Batch embed all candidate entity texts to optimize performance
        from app.services.rag.embeddings import get_embeddings_batch
        texts_to_embed = [f"{r.get('name', '')}. {r.get('description', '')}" for r in results]
        try:
            embeddings = get_embeddings_batch(texts_to_embed)
        except Exception as e:
            logger.warning(f"[KG Search] Batch embedding failed, falling back to sequential: {e}")
            from app.services.rag.embeddings import get_embedding
            embeddings = [get_embedding(t) for t in texts_to_embed]

        scored = []
        for r, entity_embedding in zip(results, embeddings):
            labels = [l for l in r.get("labels", []) if l != "Entity"]
            entity_type = labels[0] if labels else "UNKNOWN"
            sim_score = cosine_similarity(query_embedding, entity_embedding)
            
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
        Only includes relationship types meaningful for legal reasoning.
        Format is optimized for LLM consumption.
        """
        if not entity_ids:
            return ""

        # Only traverse meaningful legal relationship types
        USEFUL_REL_TYPES = {
            "THUOC", "LIEN_QUAN", "AP_DUNG", "THAM_CHIEU", "QUY_DINH", "XU_PHAT", "CAM",
            "DINH_NGHIA", "NGHIEM_CAM", "THUOC_CHUONG", "THUOC_DIEU", "THUOC_VAN_BAN"
        }

        cypher = f"""
        MATCH (start:Entity)-[r*1..{depth}]-(target:Entity)
        WHERE start.entity_id IN $entity_ids
          AND type(r[-1]) IN $rel_types
        RETURN start.name AS start_name, 
               [l IN labels(start) WHERE l <> 'Entity'][0] AS start_type,
               start.description AS start_desc,
               target.name AS target_name, 
               [l IN labels(target) WHERE l <> 'Entity'][0] AS target_type,
               type(r[-1]) AS rel_type
        LIMIT 40
        """
        
        try:
            results = self.graph.query(cypher, params={
                "entity_ids": entity_ids,
                "rel_types": list(USEFUL_REL_TYPES)
            })
        except Exception:
            # Fallback without rel_types filter if Neo4j version doesn't support IN for rel types
            cypher_fallback = f"""
            MATCH (start:Entity)-[r*1..{depth}]-(target:Entity)
            WHERE start.entity_id IN $entity_ids
            RETURN start.name AS start_name, 
                   [l IN labels(start) WHERE l <> 'Entity'][0] AS start_type,
                   start.description AS start_desc,
                   target.name AS target_name, 
                   [l IN labels(target) WHERE l <> 'Entity'][0] AS target_type,
                   type(r[-1]) AS rel_type
            LIMIT 40
            """
            results = self.graph.query(cypher_fallback, params={"entity_ids": entity_ids})
        
        if not results:
            # Fallback: just fetch the nodes themselves if no relationships found
            cypher_nodes = """
            MATCH (n:Entity) WHERE n.entity_id IN $entity_ids
            RETURN n.name AS name, [l IN labels(n) WHERE l <> 'Entity'][0] AS type, n.description AS desc
            """
            nodes = self.graph.query(cypher_nodes, params={"entity_ids": entity_ids})
            context = ""
            for n in nodes:
                context += f"🔗 [{n.get('type', 'Entity')}] {n.get('name')}\n   {n.get('desc', '')[:200]}\n"
            return context

        # Build structured relationship map — group by start entity
        entity_rels: dict = {}   # start_name -> list of (rel_type, target_name, target_type)
        entity_desc: dict = {}   # start_name -> (start_type, start_desc)

        for r in results:
            start_name = r["start_name"]
            rel_type = r["rel_type"]
            # Skip uninformative relationship types
            if rel_type not in USEFUL_REL_TYPES:
                continue
            if start_name not in entity_rels:
                entity_rels[start_name] = []
                entity_desc[start_name] = (r.get("start_type", ""), r.get("start_desc", ""))
            edge_key = f"{rel_type}-{r['target_name']}"
            if edge_key not in [f"{rr[0]}-{rr[1]}" for rr in entity_rels[start_name]]:
                entity_rels[start_name].append((rel_type, r["target_name"], r.get("target_type", "")))

        if not entity_rels:
            return ""

        # Format for LLM: clear, hierarchical, role-labeled
        rel_label = {
            "THUOC": "thuộc",
            "LIEN_QUAN": "liên quan đến",
            "AP_DUNG": "áp dụng cho",
            "THAM_CHIEU": "tham chiếu đến",
            "QUY_DINH": "quy định về",
            "XU_PHAT": "xử phạt hành vi",
            "CAM": "cấm hành vi",
            "DINH_NGHIA": "định nghĩa",
            "NGHIEM_CAM": "nghiêm cấm hành vi",
            "THUOC_CHUONG": "thuộc chương",
            "THUOC_DIEU": "thuộc điều",
            "THUOC_VAN_BAN": "thuộc văn bản",
        }

        context_parts = ["=== Mối quan hệ pháp lý (Knowledge Graph) ==="]
        for start_name, rels in entity_rels.items():
            start_type, start_desc = entity_desc[start_name]
            context_parts.append(f"\n🔗 {start_name} [{start_type}]")
            if start_desc:
                context_parts.append(f"   Mô tả: {start_desc[:150]}")
            for rel_type, target_name, target_type in rels:
                verb = rel_label.get(rel_type, rel_type)
                context_parts.append(f"   → {verb}: {target_name} [{target_type}]")

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
    # In Neo4j, hyphens '-' are preserved in entity IDs (e.g. NĐ-CP). Replace only '/' and spaces.
    sanitized = re.sub(r'[/\s]', '_', so_hieu.strip())
    return f"dieu_{sanitized}_{dieu_so}"


# Minimum similarity score for graph-expanded chunks to be injected into RAG context.
# Chunks below this threshold are dropped to avoid diluting the LLM context.
MIN_GRAPH_INJECT_SCORE = 0.28


def _score_graph_chunks(query_embedding, candidates: list) -> list:
    """
    Score graph-candidate chunks by cosine similarity with the query embedding.
    Returns only chunks with similarity >= MIN_GRAPH_INJECT_SCORE, sorted desc.

    Args:
        query_embedding: Pre-computed query embedding (np.ndarray)
        candidates: List of chunk dicts (from Qdrant scroll)
    Returns:
        Filtered and scored list of chunk dicts with 'score' set to real similarity
    """
    from app.services.rag.embeddings import get_embeddings_batch, cosine_similarity

    if not candidates:
        return []

    valid_candidates = []
    texts_to_embed = []
    for chunk in candidates:
        content = chunk.get("content", "")
        if content:
            valid_candidates.append(chunk)
            texts_to_embed.append(content[:500])

    if not texts_to_embed:
        return []

    try:
        # Batch embed all candidate texts to maximize GPU/CPU vectorization
        embeddings = get_embeddings_batch(texts_to_embed)
        
        scored = []
        for chunk, chunk_emb in zip(valid_candidates, embeddings):
            sim = cosine_similarity(query_embedding, chunk_emb)
            if sim >= MIN_GRAPH_INJECT_SCORE:
                chunk["score"] = round(sim, 3)
                scored.append(chunk)
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored
    except Exception as e:
        logger.warning(f"[GraphExpand] Batch scoring failed: {e}")
        # Fallback to sequential if batch fails
        from app.services.rag.embeddings import get_embedding
        scored = []
        for chunk in candidates:
            content = chunk.get("content", "")
            if not content:
                continue
            try:
                chunk_emb = get_embedding(content[:500])
                sim = cosine_similarity(query_embedding, chunk_emb)
                if sim >= MIN_GRAPH_INJECT_SCORE:
                    chunk["score"] = round(sim, 3)
                    scored.append(chunk)
            except Exception as ex:
                logger.warning(f"[GraphExpand] Fallback scoring failed: {ex}")
                continue
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored


def _expand_via_graph(vector_results: list, kg: "KnowledgeGraph", query: str = "", top_k: int = 5, top_extra: int = 2) -> list:
    """
    Graph Re-ranking: sau khi RAG tìm được chunks, duyệt đồ thị để tìm các điều luật
    liên quan (tham chiếu, liên kết) và bổ sung vào kết quả.

    KEY IMPROVEMENT: Trước khi inject, tính cosine similarity giữa query và chunk.
    Chỉ inject khi similarity >= MIN_GRAPH_INJECT_SCORE để tránh pha loãng context.
    Gán score thực (không hardcode 0.0) để hệ thống phân biệt được độ liên quan.

    Chiến lược 2 lớp:
    Lớp 1 (Direct): Duyệt trực tiếp 1-2 hop từ entity_id của các điều đã tìm được
    Lớp 2 (Concept Bridge): Tìm các điều từ LUẬT KHÁC cùng AP_DUNG cho 1 khái niệm
           VD: Điều 7 ATTTM --AP_DUNG--> "phần mềm độc hại" <--AP_DUNG-- Điều 83 NĐ15
    """
    from app.services.rag.retriever import get_qdrant_client
    from app.services.rag.embeddings import get_embedding
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    # ── Bước 1: Xây dựng entity_id ĐÚNG FORMAT từ RAG results ────────
    # Dùng top_k vector results để mở rộng hướng expand phù hợp với số kết quả tìm kiếm
    article_entity_ids = []
    source_so_hieus = set()  # Track which laws we already have
    for vr in vector_results[:top_k]:
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

    logger.info(f"[GraphExpand] Built entity_ids from RAG top-{top_k}: {article_entity_ids}")

    # Pre-compute query embedding once for scoring all candidates
    query_embedding = None
    if query:
        try:
            query_embedding = get_embedding(query)
        except Exception as e:
            logger.warning(f"[GraphExpand] Failed to embed query for scoring: {e}")

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

    # ── Bước 5: Truy vấn Qdrant lấy candidates từ graph ──────────────
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
            limit=top_extra * 5,   # Fetch more candidates for scoring
            with_payload=True,
        )

        # Build candidate list first
        candidates = []
        for point in qdrant_results:
            p = point.payload
            key = (p.get("ten_van_ban", ""), p.get("dieu_so", ""))
            if key in existing_keys:
                continue

            article_str = f"Điều {p.get('dieu_so', '')}"
            if p.get("dieu_ten"):
                article_str += f". {p['dieu_ten']}"

            content = p.get("full_dieu_text") or p.get("noi_dung_chunk") or p.get("content", "")
            candidates.append({
                "chunk_id": p.get("chunk_id"),
                "content": content,
                "context_text": p.get("context_text", ""),
                "dieu_so": p.get("dieu_so", ""),
                "dieu_ten": p.get("dieu_ten", ""),
                "article": article_str,
                "doc_title": p.get("ten_van_ban", ""),
                "so_hieu": p.get("so_hieu", ""),
                "score": 0.0,
                "_source": "graph_expand",
            })

        # ── Bước 6 (MỚI): Lọc relevance bằng cosine similarity ────────
        # Chỉ inject chunk có đủ độ liên quan với query
        if query_embedding is not None and candidates:
            scored_candidates = _score_graph_chunks(query_embedding, candidates)
            logger.info(
                f"[GraphExpand] Relevance filter: {len(candidates)} candidates → "
                f"{len(scored_candidates)} passed (threshold={MIN_GRAPH_INJECT_SCORE})"
            )
        else:
            # No query embedding available — skip scoring, take all (safe fallback)
            scored_candidates = candidates
            logger.warning("[GraphExpand] No query embedding for scoring — using unfiltered candidates")

        added = 0
        for chunk in scored_candidates:
            key = (chunk.get("doc_title", ""), chunk.get("dieu_so", ""))
            if key not in existing_keys:
                existing_keys.add(key)
                vector_results.append(chunk)
                added += 1
                logger.info(
                    f"[GraphExpand] ✅ Injected: {chunk['doc_title']} Điều {chunk['dieu_so']} "
                    f"(score={chunk['score']:.3f})"
                )
            if added >= top_extra:
                break

        logger.info(f"[GraphExpand] Final: injected {added} relevant articles from graph")
    except Exception as e:
        logger.warning(f"[GraphExpand] Qdrant fetch for related articles failed: {e}")

    return vector_results


# ── Thread-safe wrappers: preserve the original try/except fallbacks so a
#    failure in one parallel branch never breaks the others. ────────────────
def _safe_search_entities(kg: "KnowledgeGraph", term: str, top_k: int = 5) -> list:
    try:
        return kg.search_entities(term, top_k=top_k)
    except Exception as e:
        logger.error(f"[KG Error] Entity search failed: {e}")
        return []


def _safe_expand_via_graph(vector_results: list, kg: "KnowledgeGraph", query: str, top_k: int = 5) -> list:
    try:
        return _expand_via_graph(vector_results, kg, query=query, top_k=top_k, top_extra=2)
    except Exception as e:
        logger.warning(f"[GraphExpand] Skipped due to error: {e}")
        return vector_results


def _safe_graph_context(kg: "KnowledgeGraph", entity_ids: list) -> str:
    try:
        return kg.get_graph_context(entity_ids, depth=2)
    except Exception as e:
        logger.error(f"[KG Error] Graph context failed: {e}")
        return ""


def _safe_graph_viz(kg: "KnowledgeGraph", entity_ids: list) -> dict:
    try:
        return kg.get_graph_data_for_visualization(entity_ids, depth=1)
    except Exception as e:
        logger.error(f"[KG Error] Graph viz failed: {e}")
        return {"nodes": [], "edges": []}


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
    
    # ── 1. Multi-query Vector Search ∥ KG Entity Search ───────────────
    # These two are independent (entity search only needs the query text), so
    # we overlap the Qdrant/embedding work with the Neo4j vector lookup.
    kg = get_knowledge_graph()
    graph_search_term = entities if entities else query
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_vector = ex.submit(multi_query_search, all_queries, top_k)
        f_entities = ex.submit(_safe_search_entities, kg, graph_search_term, top_k)
        vector_results = f_vector.result()
        kg_results = f_entities.result()

    # NOTE: Không hardcode fallback inject điều luật cụ thể ở đây.
    # Việc tìm đúng điều luật được đảm bảo qua 3 lớp:
    #   1. LLM-generated sub_queries (đa dạng ngữ nghĩa)
    #   2. Static queries trong query_expansion.py (trỏ đúng Điều/Khoản theo chủ đề)
    #   3. _expand_via_graph() bên dưới (bổ sung điều luật liên quan từ graph)

    # ── 2. Build matched entity ids (KG hits + RAG top-k) ─────────────
    # Computed from the front of vector_results, so it is unaffected by the
    # graph-expanded chunks that _expand_via_graph appends to the tail.
    matched_entity_ids = [r["entity"]["entity_id"] for r in kg_results]
    for vr in vector_results[:top_k]:
        so_hieu = vr.get("so_hieu", "")
        dieu_so = vr.get("dieu_so", "")
        if so_hieu and dieu_so:
            article_entity_id = _build_entity_id(so_hieu, dieu_so)
            if article_entity_id and article_entity_id not in matched_entity_ids:
                matched_entity_ids.append(article_entity_id)

    # ── 3. Graph expand ∥ graph context ∥ viz data ────────────────────
    # All three are mutually independent: expand only appends to the tail of
    # vector_results, while context/viz read the already-fixed entity ids.
    # Running them concurrently collapses three serial Neo4j round-trips into
    # one, which is the single biggest cut to time-to-first-token.
    graph_context = ""
    graph_data = {"nodes": [], "edges": []}
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_expand = ex.submit(_safe_expand_via_graph, vector_results, kg, query, top_k)
        f_context = ex.submit(_safe_graph_context, kg, matched_entity_ids) if matched_entity_ids else None
        f_viz = ex.submit(_safe_graph_viz, kg, matched_entity_ids) if matched_entity_ids else None
        vector_results = f_expand.result()
        if f_context:
            graph_context = f_context.result()
        if f_viz:
            graph_data = f_viz.result()

    return {
        "vector_results": vector_results,
        "graph_context": graph_context,
        "graph_data": graph_data,
        "matched_entities": kg_results,
    }

