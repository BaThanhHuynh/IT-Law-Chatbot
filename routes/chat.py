"""API routes for the IT Law Chatbot."""
import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from chatbot.engine import (
    generate_response,
    create_conversation,
    get_conversation_history,
    get_all_conversations,
)
from graphrag.knowledge_graph import get_knowledge_graph

chat_router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponseData(BaseModel):
    conversation_id: str
    answer: str
    sources: List[Dict[str, Any]]
    graph_data: Dict[str, Any]


class ChatResponse(BaseModel):
    success: bool
    data: ChatResponseData


class ConversationListResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]


class NewConversationResponse(BaseModel):
    success: bool
    data: Dict[str, str]


class HistoryResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]


class KGResponse(BaseModel):
    success: bool
    data: Dict[str, Any]


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


@chat_router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations():
    """Get all conversations."""
    try:
        conversations = get_all_conversations()
        # Convert datetime objects to strings
        for conv in conversations:
            for key in ["created_at", "updated_at"]:
                if conv.get(key):
                    conv[key] = conv[key].isoformat()
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
            if msg.get("created_at"):
                msg["created_at"] = msg["created_at"].isoformat()
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

