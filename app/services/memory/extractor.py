"""
Memory Extractor using LLM to extract facts, profile updates, and session state.
Runs asynchronously in background worker threads to achieve zero latency impact.
"""
import re
import json
import concurrent.futures
from typing import Optional, Dict, Any, List
from app.core.config import Config
from app.core.logger import logger
from app.services.memory.prompts import MEMORY_EXTRACTION_PROMPT
from app.services.memory.short_term import ShortTermMemory
from app.services.memory.long_term import LongTermMemory


class MemoryExtractor:
    """Extracts facts from interaction turns and updates Short-term & Long-term memories asynchronously."""

    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="mem_worker"
    )

    _SKIP_EXTRACTION_PATTERNS = re.compile(
        r'^(xin\s*chào|chào\s*bạn|hello|hi|ok|cảm\s*ơn|cám\s*ơn|thanks|bye|tạm\s*biệt|'
        r'bạn\s*là\s*ai|bạn\s*tên\s*gì|giúp\s*tôi\s*được\s*không|ừ|uh|okie|tốt|hay)$',
        re.IGNORECASE
    )

    @classmethod
    def dispatch_extract_and_store(
        cls,
        query: str,
        answer: str,
        conversation_id: str,
        user_id: str = "default_user",
        sources: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Non-blocking dispatch: Submit extraction task to background thread pool.
        Returns immediately (0ms added latency to user response).
        """
        if not Config.ENABLE_MEMORY:
            return

        from app.services.memory import get_short_term_memory, get_long_term_memory
        short_term = get_short_term_memory()
        long_term = get_long_term_memory()

        # Immediate synchronous heuristic update for short-term citations (0ms)
        short_term.extract_citations_and_update(
            conversation_id=conversation_id,
            query=query,
            answer=answer,
            sources=sources,
            user_id=user_id
        )

        # Submit deeper LLM fact extraction to background worker
        cls._executor.submit(
            cls._extract_and_store_worker,
            query=query,
            answer=answer,
            conversation_id=conversation_id,
            user_id=user_id,
            short_term=short_term,
            long_term=long_term,
        )

    @classmethod
    def _extract_and_store_worker(
        cls,
        query: str,
        answer: str,
        conversation_id: str,
        user_id: str,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
    ):
        """Worker function executing inside background thread."""
        query_stripped = query.strip()
        answer_stripped = answer.strip()

        # Heuristic filters: skip trivial interactions or greetings
        if len(query_stripped) < 8 or len(answer_stripped) < 30:
            return

        if cls._SKIP_EXTRACTION_PATTERNS.search(query_stripped.rstrip('?!., ')):
            return

        # Skip if answer indicates system error
        if "quá tải" in answer or "lỗi hệ thống" in answer:
            return

        try:
            from app.services.chatbot.engine import get_llm

            # 1. Fetch relevant existing memories to avoid redundant fact generation
            existing = long_term.search_memories(query_stripped, user_id=user_id, limit=3)
            existing_str = "\n".join([f"- {m['fact']}" for m in existing]) if existing else "Chưa có thông tin."

            prompt = MEMORY_EXTRACTION_PROMPT.format(
                query=query_stripped[:1500],
                answer=answer_stripped[:2000],
                existing_memories=existing_str,
            )

            model = get_llm()
            chat = model.chats.create(model=Config.GEMINI_MODEL)
            response = chat.send_message(prompt)
            raw_text = response.text.strip() if response and response.text else ""

            # Clean markdown code blocks
            clean_json = re.sub(r"^```(?:json)?\s*", "", raw_text)
            clean_json = re.sub(r"\s*```$", "", clean_json)

            try:
                data = json.loads(clean_json)
            except Exception:
                # Try regex extraction of JSON object if wrapper text exists
                match = re.search(r'(\{.*\})', clean_json, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(1))
                    except Exception:
                        logger.warning(f"[MemoryExtractor] Could not parse JSON from LLM: {raw_text[:120]}")
                        return
                else:
                    logger.warning(f"[MemoryExtractor] No JSON found in LLM response: {raw_text[:120]}")
                    return

            # 2. Update Short-Term Working Memory
            session_updates = data.get("session_updates", {})
            if session_updates:
                short_term.update_state(
                    conversation_id=conversation_id,
                    focused_laws=session_updates.get("focused_laws"),
                    focused_articles=session_updates.get("focused_articles"),
                    user_role=session_updates.get("user_role"),
                    topic=session_updates.get("topic"),
                    last_query=query_stripped,
                    user_id=user_id,
                )

            # 3. Store extracted facts in Long-Term Memory
            facts: List[str] = data.get("facts", [])
            user_role = session_updates.get("user_role") if isinstance(session_updates, dict) else None

            # If user_role is detected, add it as a profile memory fact if not already present
            if user_role and user_role not in ["cá nhân", "tổ chức"]:
                role_fact = f"Người dùng có vai trò là {user_role} trong lĩnh vực CNTT."
                long_term.add_memory(
                    fact=role_fact,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_type="user_profile",
                    entities=[user_role],
                )

            for fact in facts:
                if isinstance(fact, str) and len(fact.strip()) >= 10:
                    entities = []
                    if isinstance(session_updates, dict):
                        entities = (session_updates.get("focused_laws") or []) + (session_updates.get("focused_articles") or [])
                    long_term.add_memory(
                        fact=fact.strip(),
                        user_id=user_id,
                        conversation_id=conversation_id,
                        memory_type="legal_context",
                        entities=entities,
                    )

            if facts or session_updates:
                logger.info(f"[MemoryExtractor] Background extraction saved {len(facts)} facts for user '{user_id}'.")

        except Exception as e:
            logger.error(f"[MemoryExtractor] Background extraction error: {e}", exc_info=False)
