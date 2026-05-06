"""
FastAPI app entry point for IT Law Chatbot.
"""
import warnings
import uvicorn
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.core.config import Config
from app.core.logger import logger
from app.api.routes.chat import chat_router
from app.services.chatbot.engine import generate_response

warnings.filterwarnings("ignore", category=FutureWarning)


class ChatQuery(BaseModel):
    """Request body cho endpoint /chat."""
    query: str = Field(..., description="Câu hỏi pháp lý của người dùng", min_length=1)
    conversation_id: Optional[str] = Field(None, description="ID cuộc hội thoại (tùy chọn)")


class ChatAnswer(BaseModel):
    """Response trả về cho frontend."""
    answer: str
    conversation_id: str
    sources: list = []
    graph_data: dict = {"nodes": [], "edges": []}


def create_app():
    app = FastAPI(
        title="IT Law Chatbot API",
        description="API for consulting IT laws in Vietnam",
        version="1.0.0"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": "Đã xảy ra lỗi hệ thống. Vui lòng thử lại sau."},
        )

    # ── /chat endpoint (đơn giản dành cho frontend) ──────────────────
    @app.post("/chat", response_model=ChatAnswer, tags=["Chat"])
    async def chat_endpoint(payload: ChatQuery):
        """
        Nhận câu hỏi (query) từ frontend và trả về câu trả lời (answer)
        cùng nguồn tham khảo và dữ liệu Knowledge Graph.

        Body JSON:
            { "query": "Hành vi nào bị nghiêm cấm trên không gian mạng?",
              "conversation_id": "optional-uuid" }

        Response JSON:
            { "answer": "...", "conversation_id": "...",
              "sources": [...], "graph_data": {...} }
        """
        query = (payload.query or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")

        try:
            result = generate_response(query, payload.conversation_id)
            return ChatAnswer(
                answer=result.get("answer", ""),
                conversation_id=result.get("conversation_id", ""),
                sources=result.get("sources", []),
                graph_data=result.get("graph_data", {"nodes": [], "edges": []}),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[/chat] Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

    # Include routers (giữ tương thích với /api/chat cũ)
    app.include_router(chat_router)

    # Serve static files at root (css, js, images, etc.)
    # This must be after the router to not shadow API routes
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  IT Law Chatbot - Tư vấn Luật Công nghệ thông tin")
    print(f"  Server running at http://localhost:{Config.API_PORT}")
    print(f"{'='*60}\n")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=Config.API_PORT,
        reload=Config.API_DEBUG
    )

