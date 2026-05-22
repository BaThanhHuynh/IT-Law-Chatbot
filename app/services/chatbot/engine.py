import json
import uuid
import os
import time
import concurrent.futures
from functools import wraps
from datetime import datetime
from google import genai
from google.genai import types

from app.core.config import Config
from app.core.logger import logger
from app.services.rag.retriever import get_context_from_results
from app.services.graphrag.knowledge_graph import hybrid_search
from app.services.chatbot.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, TITLE_PROMPT, INTENT_CLASSIFICATION_PROMPT, ENTITY_EXTRACTION_PROMPT, MULTI_QUERY_PROMPT, QUERY_REWRITE_PROMPT

# Configure Gemini
_model = None

def get_llm():
    """Get or initialize Gemini client."""
    global _model
    if _model is None:
        _model = genai.Client(api_key=Config.GEMINI_API_KEY)
        logger.info("[LLM] Gemini client initialized.")
    return _model

def retry_on_503(max_retries=3, backoff_factor=2):
    """Decorator to retry LLM calls on 503 UNAVAILABLE or 500 errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = 2
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str or "500" in error_str or "UNAVAILABLE" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            logger.warning(f"[{func.__name__}] LLM API overload (503/429). Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(delay)
                            delay *= backoff_factor
                            continue
                    # If it's not a retryable error or max retries reached, raise it
                    raise e
        return wrapper
    return decorator


@retry_on_503()
def classify_intent(query: str) -> str:
    """Phân loại ý định người dùng (CHATCHIT hoặc LUAT)"""
    try:
        model = get_llm()
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        response = model.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        intent = response.text.strip().upper()
        if "LUAT" in intent:
            return "LUAT"
        return "CHATCHIT"
    except Exception as e:
        logger.error(f"[Error] Intent classification failed: {e}")
        return "LUAT"  # Fallback to LUAT
@retry_on_503()
def rewrite_query(query: str, history_context: str) -> str:
    """Viết lại câu hỏi sử dụng ngữ cảnh từ lịch sử."""
    if not history_context.strip():
        return query
    try:
        model = get_llm()
        prompt = QUERY_REWRITE_PROMPT.format(history_context=history_context, query=query)
        response = model.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        rewritten = response.text.strip()
        # Fallback if model refuses or returns empty
        if not rewritten or len(rewritten) < 2:
            return query
        return rewritten
    except Exception as e:
        logger.error(f"[Error] Query rewrite failed: {e}")
        return query


@retry_on_503()
def generate_title(query: str) -> str:
    """Trích xuất từ khóa pháp lý từ câu hỏi."""
    try:
        model = get_llm()
        prompt = TITLE_PROMPT.format(query=query)
        response = model.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        try:
            text = response.text.strip()
            return text if text else "Câu hỏi mới"
        except ValueError:
            return "Câu hỏi mới"
    except Exception as e:
        logger.error(f"[Error] Title generation failed: {e}")
        return "Câu hỏi mới"


@retry_on_503()
def extract_entities(query: str) -> str:
    """Trích xuất từ khóa pháp lý từ câu hỏi."""
    try:
        model = get_llm()
        prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
        response = model.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        try:
            text = response.text.strip()
            return text if text else query
        except ValueError:
            return query
    except Exception as e:
        logger.error(f"[Error] Entity extraction failed: {e}")
        return query  # Fallback to original query


@retry_on_503()
def generate_sub_queries(query: str, num_queries: int = 3) -> list:
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
        response = model.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        
        try:
            text = response.text.strip()
        except ValueError:
            text = ""
            
        # Parse response: expect 3 lines
        variants = [line.strip() for line in text.split("\n") if line.strip()]
        
        # Always include original query first, then add all 3 variants
        all_queries = [query] + variants[:3]
        
        logger.info(f"[MultiQuery] Generated {len(all_queries)} queries:")
        for i, q in enumerate(all_queries):
            logger.info(f"  [{i}] {q[:200]}")
        
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

    # 2. Save user message (original query)
    save_message(conversation_id, "user", query)

    # 3. Phân loại intent — Regex trước, LLM sau (tiết kiệm 2-3s cho CHATCHIT)
    import re as _re
    _CHATCHIT_PATTERNS = _re.compile(
        r'^(xin\s*chào|chào\s*bạn|hello|hi\b|hey\b|ok\b|cảm\s*ơn|cám\s*ơn|thanks|thank\s*you'
        r'|tạm\s*biệt|bye|bạn\s*là\s*ai|bạn\s*tên\s*(gì|j)|bạn\s*có\s*thể\s*(giúp|làm)'
        r'|tôi\s*hiểu\s*rồi|được\s*rồi|ừ|uh|à|oke|okie|good|tốt|tuyệt|hay|wow'
        r'|giúp\s*tôi\s*được\s*không|hỗ\s*trợ\s*tôi|có\s*thể\s*giúp'
        r'|thắc\s*mắc.*bạn\s*giúp)',
        _re.IGNORECASE
    )

    query_stripped = query.strip().rstrip('?!.,')
    if _CHATCHIT_PATTERNS.search(query_stripped) and len(query_stripped) < 120:
        intent = "CHATCHIT"
        logger.info(f"[{conversation_id}] Query fast-classified as: CHATCHIT (regex)")

    # ── TÓM TẮT: Phát hiện yêu cầu tóm tắt nội dung vừa trả lời ──
    elif _re.search(
        r'(tóm\s*tắt|tóm\s*gọn|rút\s*gọn|ngắn\s*gọn\s*lại|nói\s*ngắn|giải\s*thích\s*ngắn'
        r'|tổng\s*kết|summary|tl;\s*dr|cho\s*tôi\s*bản\s*tóm|nói\s*lại\s*ngắn)',
        query_stripped, _re.IGNORECASE
    ) and len(query_stripped) < 200:
        intent = "TOMTAT"
        logger.info(f"[{conversation_id}] Query fast-classified as: TOMTAT (regex)")

        # Lấy tin nhắn assistant gần nhất
        recent_history = get_conversation_history(conversation_id, limit=4)
        last_bot_msg = ""
        for msg in reversed(recent_history):
            if msg["role"] == "assistant" and len(msg.get("content", "")) > 50:
                last_bot_msg = msg["content"]
                break

        if last_bot_msg:
            try:
                model = get_llm()
                summary_response = model.models.generate_content(
                    model="gemini-3.1-flash-lite-preview",
                    contents=f"Hãy tóm tắt nội dung sau thành 3-5 ý chính, mỗi ý 1-2 câu ngắn gọn, dùng bullet point:\n\n{last_bot_msg[:3000]}",
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "Bạn là trợ lý pháp luật CNTT. Tóm tắt nội dung luật thành 3-5 ý chính ngắn gọn. "
                            "Giữ lại số điều, khoản quan trọng. Dùng bullet point (•). Không thêm thông tin mới."
                        ),
                    )
                )
                answer = summary_response.text
            except Exception as e:
                logger.error(f"[Error] TOMTAT LLM failed: {e}")
                answer = "Xin lỗi, tôi không thể tóm tắt lúc này. Vui lòng thử lại."
        else:
            answer = "Chưa có nội dung luật nào để tóm tắt. Bạn hãy hỏi một câu hỏi về luật trước nhé!"

        save_message(conversation_id, "assistant", answer, [])
        return {
            "conversation_id": conversation_id,
            "answer": answer,
            "sources": [],
            "graph_data": {"nodes": [], "edges": []},
        }

    else:
        intent = classify_intent(query)
        logger.info(f"[{conversation_id}] Query classified as: {intent}")

    if intent == "CHATCHIT":
        # ═══ CHATCHIT MODE: Đường tắt tối đa ═══
        # Bỏ qua: rewrite, entities, sub-queries, RAG, Graph, chat history nặng
        search_results = {"vector_results": []}
        graph_data = {"nodes": [], "edges": []}

        chatchit_system = (
            "Bạn là trợ lý tư vấn pháp luật CNTT. "
            "Khi người dùng chào hỏi hoặc hỏi chuyện thông thường, hãy trả lời THẬT NGẮN GỌN: "
            "tối đa 2-3 câu, không liệt kê, không đề cập điều khoản luật. "
            "Chỉ cần xác nhận lời cảm ơn hoặc chào lại thân thiện."
        )

        # Gọi LLM trực tiếp (không tạo chat session, không load history đầy đủ)
        try:
            model = get_llm()
            chatchit_response = model.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=chatchit_system,
                )
            )
            answer = chatchit_response.text
        except Exception as e:
            logger.error(f"[Error] CHATCHIT LLM failed: {e}")
            answer = "Chào bạn! Tôi sẵn sàng hỗ trợ bạn về pháp luật CNTT. Bạn cần tư vấn gì?"

        # Save and return immediately — skip source building
        save_message(conversation_id, "assistant", answer, [])
        return {
            "conversation_id": conversation_id,
            "answer": answer,
            "sources": [],
            "graph_data": graph_data,
        }

    # ═══ LUAT MODE: Full pipeline ═══

    # 3.1 Contextual Query Rewriting (chỉ khi LUAT)
    raw_history = get_conversation_history(conversation_id, limit=4)
    history_context = ""
    for msg in raw_history:
        role_name = "User" if msg["role"] == "user" else "Bot"
        history_context += f"{role_name}: {msg['content']}\n"

    rewritten_query = query
    if history_context.strip():
        rewritten_query = rewrite_query(query, history_context)
        if rewritten_query != query:
            logger.info(f"[{conversation_id}] Query rewritten:\n  Original: {query}\n  Rewritten: {rewritten_query}")

    # 3.2 Chạy song song entity + sub-queries
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_entities = executor.submit(extract_entities, rewritten_query)
        future_sub_queries = executor.submit(generate_sub_queries, rewritten_query)
        extracted_entities = future_entities.result()
        sub_queries = future_sub_queries.result()

    # 3.3 Hybrid search (Vector + Graph)
    try:
        logger.info(f"[{conversation_id}] Extracted entities: {extracted_entities}")

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

    # ── Auto-inject Điều 4 NĐ 15/2020 (quy tắc phạt cá nhân = 1/2 tổ chức) ──
    import re as _re_ctx
    _PENALTY_KEYWORDS = _re_ctx.compile(
        r'(xử\s*phạt|mức\s*phạt|phạt\s*tiền|vi\s*phạm\s*hành\s*chính|chế\s*tài|bị\s*phạt'
        r'|xử\s*lý|hình\s*thức.*phạt|xử\s*phạt\s*bổ\s*sung|khắc\s*phục)',
        _re_ctx.IGNORECASE
    )
    if _PENALTY_KEYWORDS.search(rewritten_query) or _PENALTY_KEYWORDS.search(query):
        dieu4_context = (
            "\n\n--- Đoạn BỔ SUNG [Nghị định 15/2020/NĐ-CP - Điều 4. Quy định về mức phạt tiền] (QUY TẮC NỀN TẢNG) ---\n"
            "3. Mức phạt tiền quy định tại Chương II của Nghị định này là mức phạt tiền đối với tổ chức. "
            "Đối với cùng một hành vi vi phạm hành chính thì mức phạt tiền đối với cá nhân bằng 1/2 mức phạt tiền đối với tổ chức.\n"
            "4. Thẩm quyền xử phạt vi phạm hành chính quy định tại Chương III của Nghị định này là thẩm quyền áp dụng "
            "đối với một hành vi vi phạm hành chính của cá nhân."
        )
        rag_context += dieu4_context
        logger.info(f"[{conversation_id}] Auto-injected Điều 4 NĐ 15/2020 (penalty ÷2 rule)")

    # ── Thêm structured markers [KHOẢN] [ĐIỂM] để LLM dễ phân biệt ──
    rag_context = _re_ctx.sub(r'(?m)^(\d+)\.\s+', r'\n**[KHOẢN \1]** ', rag_context)

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
        chat_history.append(
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # 5. Generate Response using Gemini API
    try:
        # Check for MOCK mode
        if query.strip().lower().startswith("/mock"):
            logger.info("MOCK mode activated. Bypassing Gemini API.")
            answer = "Dữ liệu tìm thấy từ CSDL (Mock Mode):\n\n" + rag_context
        else:
            model = get_llm()
            chat = model.chats.create(
                model="gemini-3.1-flash-lite-preview",
                history=chat_history,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                )
            )
            
            # Inline retry logic for the main response generation
            max_retries = 3
            delay = 2
            for attempt in range(max_retries):
                try:
                    response = chat.send_message(prompt)
                    answer = response.text
                    break
                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str or "500" in error_str or "UNAVAILABLE" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            logger.warning(f"[generate_response] Gemini API overload. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(delay)
                            delay *= 2
                            continue
                    raise e
    except Exception as e:
        logger.error(f"[Error] LLM generation failed: {e}")
        answer = f"Hiện tại máy chủ đang quá tải. Xin vui lòng thử lại sau giây lát."

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
                "full_content": r.get("content", ""),
                "score": calibrate_score(r.get("score", 0)),
                "doc_title": doc,
                "so_hieu": r.get("so_hieu", ""),
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
                    "full_content": r.get("content", ""),
                    "score": calibrate_score(r.get("score", 0)),
                    "doc_title": r.get("doc_title", ""),
                    "so_hieu": r.get("so_hieu", ""),
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
                "full_content": entity.get("description", ""),
                "score": calibrate_score(real_score),
                "doc_title": "Mạng Lưới Tri Thức (GraphRAG)",
                "so_hieu": "",
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
            response = model.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=title_prompt
            )
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
