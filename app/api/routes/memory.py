"""
Memory management API endpoints for IT Law Chatbot.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from app.api.schemas import (
    MemoryAddRequest, MemoryListResponse, SessionStateResponse
)
from app.core.security import verify_api_key
from app.core.logger import logger
from app.services.memory import get_short_term_memory, get_long_term_memory

memory_router = APIRouter(prefix="/api/memory", tags=["Memory"], dependencies=[Depends(verify_api_key)])


@memory_router.get("/user/{user_id}", response_model=MemoryListResponse)
async def get_user_memories(user_id: str):
    """Retrieve all long-term memories and profile facts stored for a user."""
    try:
        long_term = get_long_term_memory()
        memories = long_term.get_all_memories(user_id=user_id)
        return {"success": True, "data": memories}
    except Exception as e:
        logger.error(f"[API] get_user_memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.post("", response_model=dict)
async def add_user_memory(payload: MemoryAddRequest):
    """Manually add or upsert a memory fact for a user."""
    try:
        long_term = get_long_term_memory()
        result = long_term.add_memory(
            fact=payload.fact,
            user_id=payload.user_id or "default_user",
            conversation_id=payload.conversation_id,
            memory_type=payload.memory_type or "user_profile",
            entities=payload.entities or [],
        )
        if result is None:
            raise HTTPException(status_code=400, detail="Không thể lưu thông tin trí nhớ này.")
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] add_user_memory failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.delete("/{memory_id}", response_model=dict)
async def delete_memory(memory_id: str):
    """Delete a specific memory by its ID."""
    try:
        long_term = get_long_term_memory()
        success = long_term.delete_memory(memory_id)
        return {"success": success, "data": {"id": memory_id, "deleted": success}}
    except Exception as e:
        logger.error(f"[API] delete_memory failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.delete("/user/{user_id}", response_model=dict)
async def clear_user_memories(user_id: str):
    """Clear all long-term memories for a specific user."""
    try:
        long_term = get_long_term_memory()
        success = long_term.clear_user_memories(user_id)
        return {"success": success, "data": {"user_id": user_id, "cleared": success}}
    except Exception as e:
        logger.error(f"[API] clear_user_memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@memory_router.get("/session/{conversation_id}", response_model=SessionStateResponse)
async def get_session_memory(conversation_id: str):
    """Get active short-term working memory state for a conversation."""
    try:
        short_term = get_short_term_memory()
        state = short_term.get_state(conversation_id)
        data = state.to_dict() if state else {
            "conversation_id": conversation_id,
            "turn_count": 0,
            "focused_laws": [],
            "focused_articles": [],
            "user_role": None,
            "last_topic": None,
        }
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"[API] get_session_memory failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
