"""
Chatbot engine: orchestrates RAG retrieval, graph context, and LLM generation.
"""
import json
import uuid
import google.generativeai as genai

from config import Config
from db import execute_query, fetch_all, fetch_one
from rag.retriever import get_context_from_results
from graphrag.knowledge_graph import hybrid_search
from chatbot.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, TITLE_PROMPT

# Configure Gemini
_model = None

def get_llm():
    """Get or initialize Gemini model."""
    global _model
    if _model is None:
        genai.configure(api_key=Config.GEMINI_API_KEY)
        _model = genai.GenerativeModel("gemini-2.5-flash")
        print("[LLM] Gemini model initialized.")
    return _model


def generate_response(query: str, conversation_id: str = None) -> dict:
    """
    Main chatbot pipeline:
    1. Hybrid retrieval (vector + graph)
    2. Build prompt with context
    3. Call Gemini API
    4. Save to conversation history
    5. Return response with sources
    """
    # 1. Create conversation if needed
    if not conversation_id:
        conversation_id = create_conversation(query)

    # 2. Save user message
    save_message(conversation_id, "user", query)

    # 3. Hybrid search (vector + knowledge graph)
    try:
        search_results = hybrid_search(query, top_k=5)
        rag_context = get_context_from_results(search_results["vector_results"])
        graph_context = search_results.get("graph_context", "Không có thông tin từ Knowledge Graph.")
        graph_data = search_results.get("graph_data", {"nodes": [], "edges": []})
    except Exception as e:
        print(f"[Error] Search failed: {e}")
        rag_context = "Không thể truy xuất dữ liệu."
        graph_context = ""
        graph_data = {"nodes": [], "edges": []}
        search_results = {"vector_results": []}

    # 4. Build prompt
    prompt = RAG_PROMPT_TEMPLATE.format(
        rag_context=rag_context,
        graph_context=graph_context,
        query=query,
    )

    # 5. Get conversation history for context
    history = get_conversation_history(conversation_id, limit=6)
    chat_history = []
    for msg in history[:-1]:  # Exclude the current user message
        chat_history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]],
        })

    # 6. Call Gemini API directly
    try:
        model = get_llm()
        chat = model.start_chat(history=chat_history)
        response = chat.send_message(
            f"{SYSTEM_PROMPT}\n\n{prompt}",
        )
        answer = response.text
    except Exception as e:
        print(f"[Error] LLM generation failed: {e}")
        answer = f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại. (Lỗi: {str(e)})"

    # 7. Build sources list
    sources = []
    for r in search_results.get("vector_results", [])[:3]:
        sources.append({
            "article": r.get("article", ""),
            "content": r.get("content", "")[:200],
            "score": round(r.get("score", 0), 3),
            "doc_title": r.get("doc_title", ""),
        })

    # 8. Save assistant message
    save_message(conversation_id, "assistant", answer, sources)

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": sources,
        "graph_data": graph_data,
    }


def create_conversation(first_query: str = "") -> str:
    """Create a new conversation and return its ID."""
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

    execute_query(
        "INSERT INTO conversations (id, title) VALUES (%s, %s)",
        (conv_id, title),
    )
    return conv_id


def save_message(conversation_id: str, role: str, content: str, sources: list = None):
    """Save a message to the conversation history."""
    execute_query(
        "INSERT INTO messages (conversation_id, role, content, sources) VALUES (%s, %s, %s, %s)",
        (conversation_id, role, content, json.dumps(sources, ensure_ascii=False) if sources else None),
    )
    # Update conversation timestamp
    execute_query(
        "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
        (conversation_id,),
    )


def get_conversation_history(conversation_id: str, limit: int = 20) -> list:
    """Get conversation messages."""
    return fetch_all(
        """SELECT role, content, sources, created_at
           FROM messages
           WHERE conversation_id = %s
           ORDER BY created_at ASC
           LIMIT %s""",
        (conversation_id, limit),
    )


def get_all_conversations() -> list:
    """Get all conversations sorted by recent."""
    return fetch_all(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    )
