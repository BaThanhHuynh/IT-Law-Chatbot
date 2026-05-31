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
            model=Config.GEMINI_MODEL,
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
            model=Config.GEMINI_MODEL,
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
            model=Config.GEMINI_MODEL,
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
            model=Config.GEMINI_MODEL,
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
            model=Config.GEMINI_MODEL,
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
                    model=Config.GEMINI_MODEL,
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
        # Fetch history once (limit=6) and reuse for both the rewrite context
        # and the final chat history — avoids a second Qdrant scroll round-trip.
        history = get_conversation_history(conversation_id, limit=6)
        
        # Exclude the current user message (last element) from history context
        prev_history = history[:-1]
        
        # Use the last 2 history messages for context
        raw_history = prev_history[-2:]
        history_context = ""
        for msg in raw_history:
            role_name = "User" if msg["role"] == "user" else "Bot"
            history_context += f"{role_name}: {msg['content']}\n"

        # If we have history, rewrite the query first to get the correct context for classification & extraction
        if history_context.strip():
            rewritten_query = rewrite_query(query, history_context)
        else:
            rewritten_query = query

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_intent = executor.submit(classify_intent, rewritten_query)
            future_entities = executor.submit(extract_entities, rewritten_query)
            future_sub_queries = executor.submit(generate_sub_queries, rewritten_query)

            intent = future_intent.result()
            extracted_entities = future_entities.result()
            sub_queries = future_sub_queries.result()

        logger.info(f"[{conversation_id}] Query classified as: {intent}")

    if intent == "CHATCHIT":
        # ═══ CHATCHIT MODE: Đường tắt tối đa ═══
        # Bỏ qua: rewrite, entities, sub-queries, RAG, Graph, chat history nặng
        search_results = {"vector_results": []}
        graph_data = {"nodes": [], "edges": []}

        chatchit_system = (
            "Bạn là trợ lý tư vấn pháp luật CNTT Việt Nam.\n"
            "Nhiệm vụ của bạn chỉ là chào hỏi, cảm ơn, xã giao và tư vấn các vấn đề liên quan đến pháp luật hoặc công nghệ thông tin.\n"
            "QUAN TRỌNG: Nếu người dùng hỏi bất kỳ câu hỏi nào ngoài lề (như kiến thức phổ thông, toán học, dịch thuật, khoa học, chính trị, địa lý...) không liên quan đến pháp luật hoặc CNTT, hãy lịch sự từ chối trả lời và hướng người dùng quay lại chủ đề tư vấn luật CNTT.\n"
            "Yêu cầu chung: Trả lời THẬT NGĂN GỌN (tối đa 2 câu), không liệt kê, không giải thích dài dòng."
        )

        # Gọi LLM trực tiếp (không tạo chat session, không load history đầy đủ)
        try:
            model = get_llm()
            chatchit_response = model.models.generate_content(
                model=Config.GEMINI_MODEL,
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
    # Cả 4 LLM call đã được chạy song song phía trên.

    if rewritten_query != query:
        logger.info(f"[{conversation_id}] Query rewritten:\n  Original: {query}\n  Rewritten: {rewritten_query}")
        # sub_queries sinh từ câu gốc → thêm câu đã rewrite để giữ recall cho câu hỏi nối tiếp
        if rewritten_query not in sub_queries:
            sub_queries = [rewritten_query] + sub_queries

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

    # 4. Reuse the conversation history fetched above (no second Qdrant scroll).
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
                model=Config.GEMINI_MODEL,
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
    all_candidates = []
    seen_articles = set()     # (doc_title, article) -> avoid exact duplicate articles

    # Gather all unique articles from vector search results
    for r in search_results.get("vector_results", []):
        if r.get("score", 0) < 0.35:
            continue
        doc = r.get("doc_title", "")
        article_key = (doc, r.get("article", ""))
        if article_key not in seen_articles:
            seen_articles.add(article_key)
            all_candidates.append({
                "article": r.get("article", ""),
                "content": r.get("content", "")[:200],
                "full_content": r.get("content", ""),
                "score": calibrate_score(r.get("score", 0)),
                "doc_title": doc,
                "so_hieu": r.get("so_hieu", ""),
                "dieu_so": r.get("dieu_so", ""),
            })

    # Gather sources from Graph DB (Neo4j)
    for r in search_results.get("matched_entities", [])[:2]:
        entity = r.get("entity", {})
        real_score = r.get("score", 0)
        # Only include relevant entities (score >= 0.35) of legal types
        if real_score >= 0.35 and entity.get("entity_type") in ["DIEU_LUAT", "VAN_BAN"]:
            article_key = ("Mạng Lưới Tri Thức (GraphRAG)", entity.get("name", ""))
            if article_key not in seen_articles:
                seen_articles.add(article_key)
                all_candidates.append({
                    "article": entity.get("name", ""),
                    "content": entity.get("description", "")[:200],
                    "full_content": entity.get("description", ""),
                    "score": calibrate_score(real_score),
                    "doc_title": "Mạng Lưới Tri Thức (GraphRAG)",
                    "so_hieu": "",
                    "dieu_so": "",
                })

    # Filter candidates by actual citations in the answer
    filtered_candidates = _filter_sources_by_citations(answer, all_candidates)

    # Apply diversity-aware top-k selection on filtered candidates
    MAX_SOURCES = Config.TOP_K_RESULTS
    sources = []
    seen_doc_titles = set()
    seen_articles_final = set()

    # Pass 1: one best chunk per unique document among filtered candidates
    for s in filtered_candidates:
        doc = s.get("doc_title", "")
        article_key = (doc, s.get("article", ""))
        if doc not in seen_doc_titles and article_key not in seen_articles_final:
            seen_doc_titles.add(doc)
            seen_articles_final.add(article_key)
            sources.append(s)
        if len(sources) >= MAX_SOURCES:
            break

    # Pass 2: fill remaining slots with next-best unique articles among filtered candidates
    if len(sources) < MAX_SOURCES:
        for s in filtered_candidates:
            article_key = (s.get("doc_title", ""), s.get("article", ""))
            if article_key not in seen_articles_final:
                seen_articles_final.add(article_key)
                sources.append(s)
            if len(sources) >= MAX_SOURCES:
                break

    # 7. Save assistant message
    save_message(conversation_id, "assistant", answer, sources)

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": sources,
        "graph_data": graph_data,
    }



def _filter_sources_by_citations(answer: str, sources: list) -> list:
    """
    Lọc danh sách nguồn trích dẫn, chỉ giữ lại những nguồn được LLM trích dẫn 
    hoặc nhắc tới trong văn bản câu trả lời (answer).
    """
    import re
    filtered = []
    answer_lower = answer.lower()
    
    # Chuẩn hóa để nhận diện viết tắt luật phổ biến
    normalized_answer = answer_lower.replace("anm", "an ninh mạng").replace("attt", "an toàn thông tin").replace("shtt", "sở hữu trí tuệ")

    for s in sources:
        doc_title = s.get("doc_title", "")
        article = s.get("article", "")
        dieu_so = s.get("dieu_so", "")
        so_hieu = s.get("so_hieu", "")
        
        # Try to parse dieu_so from article name if it's missing (e.g. "Điều 94" -> "94")
        if not dieu_so and article:
            dieu_match = re.search(r'[Đđ]i[eề]u\s*(\d+)', article)
            if dieu_match:
                dieu_so = dieu_match.group(1)
        
        # Luôn giữ nguồn GraphRAG nếu Điều tương ứng được nhắc tới
        if "GraphRAG" in doc_title:
            article_num = re.search(r"\d+", article)
            if article_num and f"điều {article_num.group()}" in normalized_answer:
                filtered.append(s)
            continue
            
        # Tìm từ khóa cốt lõi của văn bản luật
        keywords = []
        doc_lower = doc_title.lower()
        if "an ninh mạng" in doc_lower:
            keywords.extend(["an ninh mạng", "luật anm"])
        if "an toàn thông tin" in doc_lower:
            keywords.extend(["an toàn thông tin", "luật attt"])
        if "dữ liệu cá nhân" in doc_lower:
            keywords.extend(["dữ liệu cá nhân", "bảo vệ dữ liệu", "nghị định 13", "13/2023"])
        if "sở hữu trí tuệ" in doc_lower or "shtt" in doc_lower:
            keywords.extend(["sở hữu trí tuệ", "shtt", "quyền tác giả", "tác phẩm"])
        if "giao dịch điện tử" in doc_lower:
            keywords.extend(["giao dịch điện tử", "hợp đồng điện tử", "chữ ký số"])
        if "viễn thông" in doc_lower:
            # Chỉ thêm "viễn thông" nếu không phải Nghị định 15
            if "xử phạt" not in doc_lower:
                keywords.extend(["viễn thông"])
        if "thương mại điện tử" in doc_lower:
            keywords.extend(["thương mại điện tử", "tmdt", "sàn thương mại", "52/2013"])
        if "xử phạt" in doc_lower or "bưu chính" in doc_lower:
            keywords.extend(["xử phạt", "nghị định 15", "15/2020"])
        
        # Nếu không khớp nhóm nào, tách từ trong title nhưng loại bỏ các từ chung chung
        if not keywords:
            generic_words = {"điều", "luật", "nghị", "định", "thông", "tư", "quyết", "quy", "chế", "về", "của", "và", "cho", "tại", "quyền", "chi", "tiết"}
            keywords.extend([w for w in re.findall(r'\w+', doc_lower) if len(w) > 3 and w not in generic_words])
            
        # Kiểm tra sự xuất hiện của ký hiệu/số hiệu văn bản hoặc từ khóa cốt lõi
        has_doc_ref = (so_hieu and so_hieu.lower() in answer_lower) or any(k in normalized_answer for k in keywords)
        
        # Kiểm tra sự xuất hiện của số điều
        if dieu_so:
            article_num_str = f"điều {dieu_so}"
            has_article = article_num_str in normalized_answer
            # Chỉ giữ nguồn nếu câu trả lời chứa cả tên văn bản/số hiệu và số điều của nó
            if has_doc_ref and has_article:
                filtered.append(s)
        else:
            # Nếu không có số điều (chỉ trích dẫn cấp văn bản), giữ nguồn dựa vào tên văn bản/số hiệu
            if has_doc_ref:
                filtered.append(s)
            
    # Fallback: nếu lọc hết thì giữ lại toàn bộ nguồn để tránh trống
    return filtered if filtered else sources


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

    # Gộp/Tối ưu: Tránh cuộc gọi LLM phụ gây lag thời gian phản hồi đầu tiên.
    # Lấy trực tiếp câu hỏi làm tiêu đề (cắt ngắn nếu quá dài).
    if first_query:
        first_line = first_query.strip().split('\n')[0]
        title = first_line[:50] + "..." if len(first_line) > 50 else first_line

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


def generate_response_stream(query: str, conversation_id: str = None):
    """
    Main chatbot streaming pipeline.
    Yields events in dictionary format:
      1. Initial metadata: {"event": "metadata", "conversation_id": ..., "graph_data": ...}
      2. Content chunks: {"event": "text", "text": token}
      3. Citations & final results: {"event": "sources", "sources": ...}
      4. Done signal: {"event": "done"}
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
    intent = "LUAT"
    
    if _CHATCHIT_PATTERNS.search(query_stripped) and len(query_stripped) < 120:
        intent = "CHATCHIT"
        logger.info(f"[{conversation_id}] (Stream) Query fast-classified as: CHATCHIT (regex)")
    elif _re.search(
        r'(tóm\s*tắt|tóm\s*gọn|rút\s*gọn|ngắn\s*gọn\s*lại|nói\s*ngắn|giải\s*thích\s*ngắn'
        r'|tổng\s*kết|summary|tl;\s*dr|cho\s*tôi\s*bản\s*tóm|nói\s*lại\s*ngắn)',
        query_stripped, _re.IGNORECASE
    ) and len(query_stripped) < 200:
        intent = "TOMTAT"
        logger.info(f"[{conversation_id}] (Stream) Query fast-classified as: TOMTAT (regex)")

    if intent == "TOMTAT":
        # Stream summary response
        # Send initial metadata
        yield {"event": "metadata", "conversation_id": conversation_id, "graph_data": {"nodes": [], "edges": []}}
        
        # Get last bot message
        recent_history = get_conversation_history(conversation_id, limit=4)
        last_bot_msg = ""
        for msg in reversed(recent_history):
            if msg["role"] == "assistant" and len(msg.get("content", "")) > 50:
                last_bot_msg = msg["content"]
                break

        if last_bot_msg:
            try:
                model = get_llm()
                summary_response = model.models.generate_content_stream(
                    model=Config.GEMINI_MODEL,
                    contents=f"Hãy tóm tắt nội dung sau thành 3-5 ý chính, mỗi ý 1-2 câu ngắn gọn, dùng bullet point:\n\n{last_bot_msg[:3000]}",
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "Bạn là trợ lý pháp luật CNTT. Tóm tắt nội dung luật thành 3-5 ý chính ngắn gọn. "
                            "Giữ lại số điều, khoản quan trọng. Dùng bullet point (•). Không thêm thông tin mới."
                        ),
                    )
                )
                full_answer = ""
                for chunk in summary_response:
                    text = chunk.text or ""
                    full_answer += text
                    yield {"event": "text", "text": text}
                save_message(conversation_id, "assistant", full_answer, [])
            except Exception as e:
                logger.error(f"[Error] TOMTAT Stream LLM failed: {e}")
                err_msg = "Xin lỗi, tôi không thể tóm tắt lúc này. Vui lòng thử lại."
                yield {"event": "text", "text": err_msg}
                save_message(conversation_id, "assistant", err_msg, [])
        else:
            err_msg = "Chưa có nội dung luật nào để tóm tắt. Bạn hãy hỏi một câu hỏi về luật trước nhé!"
            yield {"event": "text", "text": err_msg}
            save_message(conversation_id, "assistant", err_msg, [])
            
        yield {"event": "sources", "sources": []}
        yield {"event": "done"}
        return

    if intent == "LUAT":
        # Fetch history once (limit=6) and reuse for both the rewrite context
        # and the final chat history — avoids a second Qdrant scroll round-trip.
        history = get_conversation_history(conversation_id, limit=6)
        
        # Exclude the current user message (last element) from history context
        prev_history = history[:-1]
        
        # Use the last 2 history messages for context
        raw_history = prev_history[-2:]
        history_context = ""
        for msg in raw_history:
            role_name = "User" if msg["role"] == "user" else "Bot"
            history_context += f"{role_name}: {msg['content']}\n"

        # If we have history, rewrite the query first to get the correct context for classification & extraction
        if history_context.strip():
            rewritten_query = rewrite_query(query, history_context)
        else:
            rewritten_query = query

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_intent = executor.submit(classify_intent, rewritten_query)
            future_entities = executor.submit(extract_entities, rewritten_query)
            future_sub_queries = executor.submit(generate_sub_queries, rewritten_query)

            intent = future_intent.result()
            extracted_entities = future_entities.result()
            sub_queries = future_sub_queries.result()

        logger.info(f"[{conversation_id}] (Stream) Query classified as: {intent}")

    if intent == "CHATCHIT":
        yield {"event": "metadata", "conversation_id": conversation_id, "graph_data": {"nodes": [], "edges": []}}
        
        chatchit_system = (
            "Bạn là trợ lý tư vấn pháp luật CNTT. "
            "Nếu người dùng chào hỏi mở đầu, hãy chào lại thân thiện và hỏi xem họ cần tư vấn vấn đề pháp lý gì. "
            "Nếu người dùng cảm ơn hoặc tạm biệt, hãy đáp lại lịch sự. "
            "Yêu cầu chung: Trả lời THẬT NGẮN GỌN (tối đa 2 câu), không liệt kê, không giải thích dài dòng."
        )

        try:
            model = get_llm()
            chatchit_response = model.models.generate_content_stream(
                model=Config.GEMINI_MODEL,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=chatchit_system,
                )
            )
            full_answer = ""
            for chunk in chatchit_response:
                text = chunk.text or ""
                full_answer += text
                yield {"event": "text", "text": text}
            save_message(conversation_id, "assistant", full_answer, [])
        except Exception as e:
            logger.error(f"[Error] CHATCHIT Stream LLM failed: {e}")
            err_msg = "Chào bạn! Tôi sẵn sàng hỗ trợ bạn về pháp luật CNTT. Bạn cần tư vấn gì?"
            yield {"event": "text", "text": err_msg}
            save_message(conversation_id, "assistant", err_msg, [])
            
        yield {"event": "sources", "sources": []}
        yield {"event": "done"}
        return

    # ═══ LUAT MODE: Full pipeline ═══
    # Cả 4 LLM call đã được chạy song song phía trên.

    if rewritten_query != query:
        logger.info(f"[{conversation_id}] (Stream) Query rewritten:\n  Original: {query}\n  Rewritten: {rewritten_query}")
        if rewritten_query not in sub_queries:
            sub_queries = [rewritten_query] + sub_queries

    # Import hybrid_search inside function to avoid circular import issues
    from app.services.graphrag.knowledge_graph import hybrid_search
    try:
        logger.info(f"[{conversation_id}] (Stream) Extracted entities: {extracted_entities}")

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
        logger.info(f"[{conversation_id}] (Stream) Auto-injected Điều 4 NĐ 15/2020 (penalty ÷2 rule)")

    rag_context = _re_ctx.sub(r'(?m)^(\d+)\.\s+', r'\n**[KHOẢN \1]** ', rag_context)

    system_prompt = SYSTEM_PROMPT
    prompt = RAG_PROMPT_TEMPLATE.format(
        rag_context=rag_context,
        graph_context=graph_context,
        query=rewritten_query,
    )

    # Reuse the history fetched above (no second Qdrant scroll).
    chat_history = []
    for msg in history[:-1]:  # Exclude the current user message
        chat_history.append(
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # Yield metadata first (contains conversation_id and graph_data)
    yield {"event": "metadata", "conversation_id": conversation_id, "graph_data": graph_data}

    # Stream main response
    full_answer = ""
    try:
        if query.strip().lower().startswith("/mock"):
            logger.info("MOCK mode activated. Bypassing Gemini API.")
            full_answer = "Dữ liệu tìm thấy từ CSDL (Mock Mode):\n\n" + rag_context
            yield {"event": "text", "text": full_answer}
        else:
            model = get_llm()
            chat = model.chats.create(
                model=Config.GEMINI_MODEL,
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
                    response = chat.send_message_stream(prompt)
                    for chunk in response:
                        text = chunk.text or ""
                        full_answer += text
                        yield {"event": "text", "text": text}
                    break
                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str or "500" in error_str or "UNAVAILABLE" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            logger.warning(f"[generate_response_stream] Gemini API overload. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(delay)
                            delay *= 2
                            continue
                    raise e
    except Exception as e:
        logger.error(f"[Error] LLM stream generation failed: {e}")
        err_msg = "Hiện tại máy chủ đang quá tải. Xin vui lòng thử lại sau giây lát."
        full_answer += err_msg
        yield {"event": "text", "text": err_msg}

    # Now build and filter sources after full_answer is fully generated
    from app.services.rag.embeddings import calibrate_score
    all_candidates = []
    seen_articles = set()

    for r in search_results.get("vector_results", []):
        if r.get("score", 0) < 0.35:
            continue
        doc = r.get("doc_title", "")
        article_key = (doc, r.get("article", ""))
        if article_key not in seen_articles:
            seen_articles.add(article_key)
            all_candidates.append({
                "article": r.get("article", ""),
                "content": r.get("content", "")[:200],
                "full_content": r.get("content", ""),
                "score": calibrate_score(r.get("score", 0)),
                "doc_title": doc,
                "so_hieu": r.get("so_hieu", ""),
                "dieu_so": r.get("dieu_so", ""),
            })

    for r in search_results.get("matched_entities", [])[:2]:
        entity = r.get("entity", {})
        real_score = r.get("score", 0)
        if real_score >= 0.35 and entity.get("entity_type") in ["DIEU_LUAT", "VAN_BAN"]:
            article_key = ("Mạng Lưới Tri Thức (GraphRAG)", entity.get("name", ""))
            if article_key not in seen_articles:
                seen_articles.add(article_key)
                all_candidates.append({
                    "article": entity.get("name", ""),
                    "content": entity.get("description", "")[:200],
                    "full_content": entity.get("description", ""),
                    "score": calibrate_score(real_score),
                    "doc_title": "Mạng Lưới Tri Thức (GraphRAG)",
                    "so_hieu": "",
                    "dieu_so": "",
                })

    filtered_candidates = _filter_sources_by_citations(full_answer, all_candidates)

    MAX_SOURCES = Config.TOP_K_RESULTS
    sources = []
    seen_doc_titles = set()
    seen_articles_final = set()

    for s in filtered_candidates:
        doc = s.get("doc_title", "")
        article_key = (doc, s.get("article", ""))
        if doc not in seen_doc_titles and article_key not in seen_articles_final:
            seen_doc_titles.add(doc)
            seen_articles_final.add(article_key)
            sources.append(s)
        if len(sources) >= MAX_SOURCES:
            break

    if len(sources) < MAX_SOURCES:
        for s in filtered_candidates:
            article_key = (s.get("doc_title", ""), s.get("article", ""))
            if article_key not in seen_articles_final:
                seen_articles_final.add(article_key)
                sources.append(s)
            if len(sources) >= MAX_SOURCES:
                break

    # Save to history
    save_message(conversation_id, "assistant", full_answer, sources)

    # Yield sources & done
    yield {"event": "sources", "sources": sources}
    yield {"event": "done"}

