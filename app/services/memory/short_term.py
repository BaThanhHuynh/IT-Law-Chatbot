"""
Short-Term Working Memory (Session Context) for tracking active dialogue state and enabling Fast-Path processing.
Inspired by Mem0 working memory architecture.
"""
import re
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from app.core.logger import logger


@dataclass
class SessionState:
    conversation_id: str
    user_id: str = "default_user"
    focused_laws: List[str] = field(default_factory=list)
    focused_articles: List[str] = field(default_factory=list)
    user_role: Optional[str] = None  # "cá nhân", "tổ chức", "doanh nghiệp CNTT", "sàn TMĐT", ...
    last_topic: Optional[str] = None
    last_query: Optional[str] = None
    last_answer_snippet: Optional[str] = None
    turn_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_context_string(self) -> str:
        """Render a concise context string for injection into prompts."""
        parts = []
        if self.user_role:
            parts.append(f"Vai trò/Đối tượng áp dụng: {self.user_role}")
        if self.focused_laws:
            parts.append(f"Văn bản luật đang tập trung: {', '.join(self.focused_laws)}")
        if self.focused_articles:
            parts.append(f"Điều khoản trọng tâm: {', '.join(self.focused_articles)}")
        if self.last_topic:
            parts.append(f"Chủ đề gần nhất: {self.last_topic}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "focused_laws": self.focused_laws,
            "focused_articles": self.focused_articles,
            "user_role": self.user_role,
            "last_topic": self.last_topic,
            "last_query": self.last_query,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ShortTermMemory:
    """Manages active session context across turns in a conversation to optimize LLM calls."""

    # Regex patterns matching typical Vietnamese legal follow-up questions
    _FOLLOW_UP_PATTERNS = re.compile(
        r'^(mức phạt|phạt bao nhiêu|thẩm quyền|ai phạt|ai có quyền|khoản \d+|điều này|điều đó|còn gì nữa|'
        r'đối với cá nhân|đối với tổ chức|trường hợp này|thế nào|như thế nào|'
        r'vậy thì|có bị|bị phạt thế nào|hình phạt bổ sung|biện pháp khắc phục|'
        r'thời hiệu|thời hạn|thủ tục|hồ sơ|bao lâu|điều kiện|trách nhiệm|nghĩa vụ|'
        r'nếu vi phạm|có bị phạt không|có bị thu hồi|có phải bồi thường|'
        r'chi tiết hơn|giải thích rõ hơn|nói rõ hơn|tại sao|còn trường hợp|cho ví dụ)',
        re.IGNORECASE
    )

    _ROLE_PATTERNS = [
        (re.compile(r'\b(doanh nghiệp|công ty|tổ chức|pháp nhân)\b', re.I), "tổ chức / doanh nghiệp"),
        (re.compile(r'\b(cá nhân|người dùng|tôi|lập trình viên|dev)\b', re.I), "cá nhân"),
        (re.compile(r'\b(sàn thương mại điện tử|website thương mại điện tử|sàn tmđt|shopee|lazada|tiki)\b', re.I), "doanh nghiệp sàn thương mại điện tử"),
        (re.compile(r'\b(xuyên biên giới|dịch vụ qua biên giới)\b', re.I), "doanh nghiệp cung cấp dịch vụ xuyên biên giới"),
    ]

    def __init__(self, max_sessions: int = 1000):
        self._sessions: Dict[str, SessionState] = {}
        self._max_sessions = max_sessions

    def get_state(self, conversation_id: str) -> Optional[SessionState]:
        return self._sessions.get(conversation_id)

    def get_or_create_state(self, conversation_id: str, user_id: str = "default_user") -> SessionState:
        if conversation_id not in self._sessions:
            if len(self._sessions) >= self._max_sessions:
                # Evict oldest session (FIFO)
                first_key = next(iter(self._sessions))
                del self._sessions[first_key]
            self._sessions[conversation_id] = SessionState(
                conversation_id=conversation_id,
                user_id=user_id
            )
        state = self._sessions[conversation_id]
        if user_id and state.user_id == "default_user" and user_id != "default_user":
            state.user_id = user_id
        return state

    def update_state(
        self,
        conversation_id: str,
        focused_laws: Optional[List[str]] = None,
        focused_articles: Optional[List[str]] = None,
        user_role: Optional[str] = None,
        topic: Optional[str] = None,
        last_query: Optional[str] = None,
        last_answer_snippet: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        state = self.get_or_create_state(conversation_id, user_id or "default_user")
        state.turn_count += 1
        state.updated_at = datetime.now().isoformat()
        
        if focused_laws:
            for law in focused_laws:
                if law and law.strip() and law not in state.focused_laws:
                    state.focused_laws.append(law.strip())
            state.focused_laws = state.focused_laws[-3:]

        if focused_articles:
            for art in focused_articles:
                if art and art.strip() and art not in state.focused_articles:
                    state.focused_articles.append(art.strip())
            state.focused_articles = state.focused_articles[-3:]

        if user_role:
            state.user_role = user_role

        if topic:
            state.last_topic = topic

        if last_query:
            state.last_query = last_query

        if last_answer_snippet:
            state.last_answer_snippet = last_answer_snippet[:300]

    def extract_citations_and_update(
        self,
        conversation_id: str,
        query: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        user_id: str = "default_user"
    ):
        """
        Fast heuristic extractor: Updates session state from sources and query text
        immediately without needing an LLM call.
        """
        state = self.get_or_create_state(conversation_id, user_id)
        state.turn_count += 1
        state.last_query = query
        state.last_answer_snippet = answer[:300] if answer else None
        state.updated_at = datetime.now().isoformat()

        # 1. Extract role from user query
        for pattern, role_label in self._ROLE_PATTERNS:
            if pattern.search(query):
                state.user_role = role_label
                break

        # 2. Extract focused laws & articles from sources
        if sources:
            for s in sources:
                doc = s.get("doc_title") or s.get("ten_van_ban") or ""
                article = s.get("article") or s.get("dieu_so") or ""
                if doc and doc not in state.focused_laws and "GraphRAG" not in doc:
                    state.focused_laws.append(doc)
                if article and str(article) not in state.focused_articles:
                    # Clean article text e.g. "Điều 84" or "84"
                    art_str = f"Điều {article}" if str(article).isdigit() else str(article)
                    state.focused_articles.append(art_str)

        # 3. Regex extract from bot answer if sources were empty
        if not state.focused_articles:
            dieu_matches = re.findall(r'[Đđ]i[eề]u\s*(\d+)', answer)
            for d in dieu_matches[:2]:
                art_str = f"Điều {d}"
                if art_str not in state.focused_articles:
                    state.focused_articles.append(art_str)

        if not state.focused_laws:
            nd_matches = re.findall(r'(Nghị định \d+/\d+/[A-ZĐ-]+|Luật [A-ZÀ-Ỹa-zà-ỹ\s]+ \d{4})', answer)
            for nd in nd_matches[:2]:
                clean_nd = nd.strip()
                if clean_nd not in state.focused_laws:
                    state.focused_laws.append(clean_nd)

        state.focused_laws = state.focused_laws[-3:]
        state.focused_articles = state.focused_articles[-3:]

    def is_follow_up_query(self, query: str, conversation_id: str) -> bool:
        """
        Check if the query is a short follow-up query that can be resolved via Fast-Path
        using Short-Term memory rather than an expensive LLM rewrite call.
        """
        state = self.get_state(conversation_id)
        if not state:
            return False

        if not state.focused_laws and not state.focused_articles and not state.last_topic and not state.last_query:
            return False

        q_clean = query.strip()
        words = q_clean.split()
        word_count = len(words)

        # Explicit follow-up pattern match
        if self._FOLLOW_UP_PATTERNS.search(q_clean):
            return True

        # Short question (< 12 words) that lacks an explicit law or article mention
        has_explicit_law = bool(re.search(r'(luật|nghị định|thông tư|bộ luật)\s+[A-Z0-9]', q_clean, re.I))
        if word_count <= 12 and not has_explicit_law:
            return True

        return False

    def build_fast_path_query(self, query: str, conversation_id: str) -> str:
        """
        Synthesize an enriched query directly from short-term memory state
        without requiring an expensive LLM rewrite call.
        """
        state = self.get_state(conversation_id)
        if not state:
            return query

        context_additions = []
        q_lower = query.lower()

        # Check if query already mentions the article
        if state.focused_articles:
            latest_art = state.focused_articles[-1]
            art_num = re.search(r'\d+', latest_art)
            if art_num:
                if f"điều {art_num.group()}" not in q_lower:
                    context_additions.append(latest_art)
            elif latest_art.lower() not in q_lower:
                context_additions.append(latest_art)

        # Check if query already mentions the law
        if state.focused_laws:
            latest_law = state.focused_laws[-1]
            # Check shorthand e.g. "15/2020" or "an ninh mạng"
            law_keywords = [w for w in latest_law.lower().split() if len(w) > 3 and w not in ["nghị", "định", "luật"]]
            if not any(kw in q_lower for kw in law_keywords):
                context_additions.append(latest_law)

        # Check role context
        if state.user_role and state.user_role.lower() not in q_lower:
            if any(k in q_lower for k in ["mức phạt", "bị phạt", "xử phạt", "đối với", "áp dụng"]):
                context_additions.append(f"đối với {state.user_role}")

        if context_additions:
            enriched = f"{query} ({', '.join(context_additions)})"
            logger.info(f"[FastPath Memory] Short-term enriched: '{query}' -> '{enriched}'")
            return enriched

        return query

    def clear(self, conversation_id: str):
        """Clear memory for a specific conversation."""
        if conversation_id in self._sessions:
            del self._sessions[conversation_id]
