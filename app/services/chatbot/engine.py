import json
import uuid
import os
from datetime import datetime
import google.generativeai as genai

from app.core.config import Config
from app.core.logger import logger
from app.services.rag.retriever import get_context_from_results
from app.services.graphrag.knowledge_graph import hybrid_search
from app.services.chatbot.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, TITLE_PROMPT, INTENT_CLASSIFICATION_PROMPT, ENTITY_EXTRACTION_PROMPT, MULTI_QUERY_PROMPT, QUERY_REWRITE_PROMPT

# Configure Gemini
_model = None

def get_llm():
    """Get or initialize Gemini model."""
    global _model
    if _model is None:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        _model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
        logger.info("[LLM] Gemini model initialized.")
    return _model



def classify_intent(query: str) -> str:
    """Phân loại ý định người dùng (CHATCHIT hoặc LUAT)"""
    try:
        model = get_llm()
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        response = model.generate_content(prompt)
        intent = response.text.strip().upper()
        if "LUAT" in intent:
            return "LUAT"
        return "CHATCHIT"
    except Exception as e:
        logger.error(f"[Error] Intent classification failed: {e}")
        return "LUAT"  # Fallback to LUAT
def rewrite_query(query: str, history_context: str) -> str:
    """Viết lại câu hỏi sử dụng ngữ cảnh từ lịch sử."""
    if not history_context.strip():
        return query
    try:
        model = get_llm()
        prompt = QUERY_REWRITE_PROMPT.format(history_context=history_context, query=query)
        response = model.generate_content(prompt)
        rewritten = response.text.strip()
        # Fallback if model refuses or returns empty
        if not rewritten or len(rewritten) < 2:
            return query
        return rewritten
    except Exception as e:
        logger.error(f"[Error] Query rewrite failed: {e}")
        return query


def extract_entities(query: str) -> str:
    """Trích xuất từ khóa pháp lý từ câu hỏi."""
    try:
        model = get_llm()
        prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"[Error] Entity extraction failed: {e}")
        return query  # Fallback to original query


def generate_sub_queries(query: str) -> list:
    """
    Generate 3 alternative query formulations using LLM for multi-query retrieval.
    Each variant targets a different legal angle:
      - Variant 1: Specialized domain law (Luật CNTT, Luật ANM, Nghị định...)
      - Variant 2: Rights & protection measures
      - Variant 3: Prohibited acts & penalties
    Returns list: [original_query, variant_1, variant_2, variant_3]
    """
    try:
        model = get_llm()
        prompt = MULTI_QUERY_PROMPT.format(query=query)
        response = model.generate_content(prompt)
        
        # Parse response: expect 3 lines
        variants = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
        
        # Always include original query first, then add all 3 variants
        all_queries = [query] + variants[:3]
        
        logger.info(f"[MultiQuery] Generated {len(all_queries)} queries:")
        for i, q in enumerate(all_queries):
            logger.info(f"  [{i}] {q[:80]}")
        
        return all_queries
    except Exception as e:
        logger.error(f"[Error] Multi-query generation failed: {e}")
        return [query]  # Fallback to single original query



def generate_response(query: str, conversation_id: str = None) -> dict:
    """
    Main chatbot pipeline:
    1. Intent classification
    2. Entity extraction + Multi-query generation
    3. Hybrid search (multi-query vector + KG entity + graph traversal)
    4. Build prompt with context
    5. Call Gemini API
    6. Save to conversation history
    7. Return response with sources
    """
    # 1. Create conversation if needed
    if not conversation_id:
        conversation_id = create_conversation(query)

    # 1.5 Contextual Query Rewriting
    raw_history = get_conversation_history(conversation_id, limit=4) # 2 recent turns
    history_context = ""
    for msg in raw_history:
        role_name = "User" if msg["role"] == "user" else "Bot"
        history_context += f"{role_name}: {msg['content']}\n"
    
    rewritten_query = query
    if history_context.strip():
        rewritten_query = rewrite_query(query, history_context)
        if rewritten_query != query:
            logger.info(f"[{conversation_id}] Query rewritten:\n  Original: {query}\n  Rewritten: {rewritten_query}")

    # 2. Save user message (original query)
    save_message(conversation_id, "user", query)

    # 3. Classify intent (using rewritten query)
    intent = classify_intent(rewritten_query)
    logger.info(f"[{conversation_id}] Query classified as: {intent}")

    if intent == "CHATCHIT":
        # CHATCHIT MODE: Bypass RAG and Graph search
        search_results = {"vector_results": []}
        graph_data = {"nodes": [], "edges": []}
        
        system_prompt = "Bạn là trợ lý ảo chuyên tư vấn pháp luật CNTT tại Việt Nam. Người dùng đang trò chuyện hoặc chào hỏi thông thường. Hãy đáp lại thân thiện, ngắn gọn và chủ động giới thiệu rằng bạn có thể giải đáp các thắc mắc về Luật An ninh mạng, Giao dịch điện tử, Viễn thông, v.v."
        prompt = query
    else:
        # LUAT MODE: Full hybrid search pipeline
        try:
            # Step 1: Extract entities for KG keyword matching
            extracted_entities = extract_entities(rewritten_query)
            logger.info(f"[{conversation_id}] Extracted entities: {extracted_entities}")

            # Step 2: Generate multi-query variants via LLM
            sub_queries = generate_sub_queries(rewritten_query)

            # Step 3: Hybrid search
            search_results = hybrid_search(
                query=rewritten_query,
                sub_queries=sub_queries,
                entities=extracted_entities,
                top_k=Config.TOP_K_RESULTS,
            )

            graph_data = search_results.get("graph_data", {"nodes": [], "edges": []})
            rag_context = get_context_from_results(search_results["vector_results"])
            graph_context = search_results.get("graph_context", "")

        except Exception as e:
            logger.error(f"[Error] Search failed: {e}")
            rag_context = "Không thể truy xuất dữ liệu."
            graph_context = ""
            graph_data = {"nodes": [], "edges": []}
            search_results = {"vector_results": []}

        # Build RAG prompt
        system_prompt = SYSTEM_PROMPT
        prompt = RAG_PROMPT_TEMPLATE.format(
            rag_context=rag_context,
            graph_context=graph_context,
            query=rewritten_query,
        )

    # 4. Get conversation history for context
    history = get_conversation_history(conversation_id, limit=6)
    chat_history = []
    for msg in history[:-1]:  # Exclude the current user message
        chat_history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]],
        })

    # 5. Generate Response using Gemini API
    try:
        # Check for MOCK mode
        if query.strip().lower().startswith("/mock"):
            logger.info("MOCK mode activated. Bypassing Gemini API.")
            answer = "Dữ liệu tìm thấy từ CSDL (Mock Mode):\n\n" + rag_context
        else:
            model = get_llm()
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(
                f"{system_prompt}\n\n{prompt}",
            )
            answer = response.text
    except Exception as e:
        logger.error(f"[Error] LLM generation failed: {e}")
        answer = f"Lỗi API key Gemini"

    # 6. Build sources list (calibrate raw scores to user-friendly confidence)
    from app.services.rag.embeddings import calibrate_score
    sources = []
    
    # Sources from Vector DB (Qdrant) — diversity-aware selection
    # Strategy: pick best chunk per unique doc_title (max 1 per document),
    # then fill remaining slots with next-best unique articles from same docs.
    # This ensures ALL cited documents appear in Sources, not just top-score repeated.
    MAX_SOURCES = 4
    seen_doc_titles = set()   # track which doc_titles already have a source
    seen_articles = set()     # (doc_title, article) -> avoid exact duplicate articles

    # Pass 1: one best chunk per unique document
    for r in search_results.get("vector_results", []):
        doc = r.get("doc_title", "")
        article_key = (doc, r.get("article", ""))
        if doc not in seen_doc_titles and article_key not in seen_articles:
            seen_doc_titles.add(doc)
            seen_articles.add(article_key)
            sources.append({
                "article": r.get("article", ""),
                "content": r.get("content", "")[:200],
                "score": calibrate_score(r.get("score", 0)),
                "doc_title": doc,
            })
        if len(sources) >= MAX_SOURCES:
            break

    # Pass 2: fill remaining slots with next-best unique articles (different article number)
    if len(sources) < MAX_SOURCES:
        for r in search_results.get("vector_results", []):
            article_key = (r.get("doc_title", ""), r.get("article", ""))
            if article_key not in seen_articles:
                seen_articles.add(article_key)
                sources.append({
                    "article": r.get("article", ""),
                    "content": r.get("content", "")[:200],
                    "score": calibrate_score(r.get("score", 0)),
                    "doc_title": r.get("doc_title", ""),
                })
            if len(sources) >= MAX_SOURCES:
                break

    # Sources from Graph DB (Neo4j) — real cosine similarity scores
    for r in search_results.get("matched_entities", [])[:2]:
        entity = r.get("entity", {})
        real_score = r.get("score", 0)
        # Only include relevant entities (score >= 0.35) of legal types
        if real_score >= 0.35 and entity.get("entity_type") in ["DIEU_LUAT", "VAN_BAN"]:
            sources.append({
                "article": entity.get("name", ""),
                "content": entity.get("description", "")[:200],
                "score": calibrate_score(real_score),
                "doc_title": "Mạng Lưới Tri Thức (GraphRAG)",
            })

    # 7. Save assistant message
    save_message(conversation_id, "assistant", answer, sources)

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": sources,
        "graph_data": graph_data,
    }


def _init_qdrant_history():
    """Initialize Qdrant collection for chat history if it doesn't exist."""
    from qdrant_client.models import VectorParams, Distance
    from app.services.rag.retriever import get_qdrant_client
    
    client = get_qdrant_client()
    try:
        if not client.collection_exists(Config.QDRANT_HISTORY_COLLECTION):
            client.create_collection(
                collection_name=Config.QDRANT_HISTORY_COLLECTION,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
            logger.info(f"[Qdrant] Created collection {Config.QDRANT_HISTORY_COLLECTION} for chat history.")
    except Exception as e:
        logger.error(f"[Qdrant] Error initializing history collection: {e}")

def create_conversation(first_query: str = "") -> str:
    """Create a new conversation and return its ID."""
    from qdrant_client.models import PointStruct
    from app.services.rag.retriever import get_qdrant_client
    
    _init_qdrant_history()
    client = get_qdrant_client()
    conv_id = str(uuid.uuid4())
    title = "Cuộc hội thoại mới"

    # Try to generate a smart title
    if first_query:
        try:
            model = get_llm()
            title_prompt = TITLE_PROMPT.format(query=first_query)
            response = model.generate_content(title_prompt)
            title = response.text.strip()[:100]
        except Exception:
            title = first_query[:50] + "..." if len(first_query) > 50 else first_query

    now = datetime.now().isoformat()
    
    try:
        client.upsert(
            collection_name=Config.QDRANT_HISTORY_COLLECTION,
            points=[
                PointStruct(
                    id=conv_id,
                    vector=[0.0],
                    payload={
                        "type": "conversation",
                        "id": conv_id,
                        "title": title,
                        "created_at": now,
                        "updated_at": now
                    }
                )
            ]
        )
    except Exception as e:
        logger.error(f"[Error] Failed to create conversation {conv_id}: {e}")
        
    return conv_id


def save_message(conversation_id: str, role: str, content: str, sources: list = None):
    """Save a message to the conversation history in Qdrant."""
    from qdrant_client.models import PointStruct
    from app.services.rag.retriever import get_qdrant_client
    
    _init_qdrant_history()
    client = get_qdrant_client()
    msg_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    try:
        client.upsert(
            collection_name=Config.QDRANT_HISTORY_COLLECTION,
            points=[
                PointStruct(
                    id=msg_id,
                    vector=[0.0],
                    payload={
                        "type": "message",
                        "conversation_id": conversation_id,
                        "role": role,
                        "content": content,
                        "sources": sources,
                        "created_at": now
                    }
                )
            ]
        )
        
        # Update conversation's updated_at
        records = client.retrieve(
            collection_name=Config.QDRANT_HISTORY_COLLECTION,
            ids=[conversation_id]
        )
        if records:
            conv_payload = records[0].payload
            conv_payload["updated_at"] = now
            client.upsert(
                collection_name=Config.QDRANT_HISTORY_COLLECTION,
                points=[
                    PointStruct(
                        id=conversation_id,
                        vector=[0.0],
                        payload=conv_payload
                    )
                ]
            )
    except Exception as e:
        logger.error(f"[Error] Failed to save message for {conversation_id}: {e}")


def get_conversation_history(conversation_id: str, limit: int = 20) -> list:
    """Get conversation messages from Qdrant."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from app.services.rag.retriever import get_qdrant_client
    
    _init_qdrant_history()
    client = get_qdrant_client()
    try:
        records, _ = client.scroll(
            collection_name=Config.QDRANT_HISTORY_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="type", match=MatchValue(value="message")),
                    FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))
                ]
            ),
            limit=1000
        )
        msgs = [r.payload for r in records]
        msgs.sort(key=lambda x: x["created_at"])
        return msgs[-limit:] if limit else msgs
    except Exception as e:
        logger.error(f"[Error] get_conversation_history failed: {e}")
        return []


def get_all_conversations() -> list:
    """Get all conversations sorted by recent from Qdrant."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from app.services.rag.retriever import get_qdrant_client
    
    _init_qdrant_history()
    client = get_qdrant_client()
    try:
        records, _ = client.scroll(
            collection_name=Config.QDRANT_HISTORY_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="type", match=MatchValue(value="conversation"))
                ]
            ),
            limit=10000
        )
        convs = [r.payload for r in records]
        convs.sort(key=lambda x: x["updated_at"], reverse=True)
        return convs
    except Exception as e:
        logger.error(f"[Error] get_all_conversations failed: {e}")
        return []
