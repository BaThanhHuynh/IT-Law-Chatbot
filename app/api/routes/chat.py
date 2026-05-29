import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest, ChatResponse, ConversationListResponse,
    NewConversationResponse, HistoryResponse, KGResponse
)
from app.core.logger import logger
from app.core.security import verify_api_key
from app.services.chatbot.engine import (
    generate_response,
    generate_response_stream,
    create_conversation,
    get_conversation_history,
    get_all_conversations,
)
from app.services.graphrag.knowledge_graph import get_knowledge_graph

chat_router = APIRouter(prefix="/api", dependencies=[Depends(verify_api_key)])


@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get AI response."""
    if not request.message:
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")

    try:
        result = generate_response(request.message, request.conversation_id)
        return {
            "success": True,
            "data": {
                "conversation_id": result["conversation_id"],
                "answer": result["answer"],
                "sources": result["sources"],
                "graph_data": result["graph_data"],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


@chat_router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Send a message and get a streaming AI response via SSE."""
    if not request.message:
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")

    async def event_generator():
        try:
            for chunk in generate_response_stream(request.message, request.conversation_id):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[SSE Error] Stream failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@chat_router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations():
    """Get all conversations."""
    try:
        conversations = get_all_conversations()
        return {"success": True, "data": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.post("/conversations", response_model=NewConversationResponse)
async def new_conversation():
    """Create a new conversation."""
    try:
        conv_id = create_conversation()
        return {"success": True, "data": {"id": conv_id}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/conversations/{conversation_id}", response_model=HistoryResponse)
async def get_conversation(conversation_id: str):
    """Get messages for a conversation."""
    try:
        messages = get_conversation_history(conversation_id)
        for msg in messages:
            if msg.get("sources") and isinstance(msg["sources"], str):
                msg["sources"] = json.loads(msg["sources"])
        return {"success": True, "data": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/knowledge-graph", response_model=KGResponse)
async def get_kg_data(
    entity_ids: Optional[str] = Query(None),
    depth: int = Query(1)
):
    """Get knowledge graph data for visualization."""
    try:
        kg = get_knowledge_graph()

        ids = None
        if entity_ids:
            ids = entity_ids.split(",")

        graph_data = kg.get_graph_data_for_visualization(
            entity_ids=ids,
            depth=depth
        )
        return {"success": True, "data": graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/article-full")
async def get_article_full(
    dieu_so: str = Query(..., description="Số điều (VD: 83)"),
    so_hieu: str = Query("", description="Số hiệu văn bản (VD: 15/2020/NĐ-CP)")
):
    """
    Lấy toàn bộ nội dung gốc của 1 điều luật từ Qdrant.
    Ghép tất cả chunks (noi_dung_chunk) theo thứ tự sub_index.
    """
    from app.services.rag.retriever import get_qdrant_client
    from app.core.config import Config
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    try:
        client = get_qdrant_client()

        # Build filter: dieu_so bắt buộc, so_hieu tuỳ chọn
        conditions = [
            FieldCondition(key="dieu_so", match=MatchValue(value=str(dieu_so)))
        ]
        if so_hieu:
            conditions.append(
                FieldCondition(key="so_hieu", match=MatchValue(value=str(so_hieu)))
            )

        results, _ = client.scroll(
            collection_name=Config.QDRANT_COLLECTION,
            scroll_filter=Filter(must=conditions),
            limit=50,
            with_payload=True,
        )

        if not results:
            return {"success": True, "data": {"full_text": "Không tìm thấy nội dung điều luật.", "chunks": 0}}

        # Sort chunks by sub_index to reconstruct original order
        chunks = sorted(results, key=lambda p: int(p.payload.get("chunk_sub_index", 0)))

        # Reconstruct full text from all chunks
        full_parts = []
        for point in chunks:
            p = point.payload
            chunk_text = p.get("noi_dung_chunk", "")
            if chunk_text and chunk_text not in full_parts:
                full_parts.append(chunk_text)

        full_text = "\n".join(full_parts)

        # Get metadata from first chunk
        meta = chunks[0].payload
        return {
            "success": True,
            "data": {
                "full_text": full_text,
                "dieu_so": meta.get("dieu_so", ""),
                "dieu_ten": meta.get("dieu_ten", ""),
                "ten_van_ban": meta.get("ten_van_ban", ""),
                "so_hieu": meta.get("so_hieu", ""),
                "chuong_so": meta.get("chuong_so", ""),
                "chuong_ten": meta.get("chuong_ten", ""),
                "chunks": len(chunks),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


