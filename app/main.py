"""
FastAPI app entry point for IT Law Chatbot.
"""
import warnings
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import Config
from app.core.logger import logger
from app.core.security import verify_api_key
from app.api.routes.chat import chat_router
from app.api.routes.memory import memory_router
from app.services.chatbot.engine import generate_response
from app.services.memory import get_long_term_memory

warnings.filterwarnings("ignore", category=FutureWarning)

limiter = Limiter(key_func=get_remote_address)
_rate_limit = f"{Config.RATE_LIMIT_PER_MINUTE}/minute"


class ChatQuery(BaseModel):
    """Request body cho endpoint /chat."""
    query: str = Field(..., description="Câu hỏi pháp lý của người dùng", min_length=1, max_length=Config.MAX_QUERY_LENGTH)
    conversation_id: Optional[str] = Field(None, description="ID cuộc hội thoại (tùy chọn)")
    user_id: Optional[str] = Field("default_user", description="ID người dùng để cá nhân hóa bộ nhớ")


class ChatAnswer(BaseModel):
    """Response trả về cho frontend."""
    answer: str
    conversation_id: str
    sources: list = []
    graph_data: dict = {"nodes": [], "edges": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Warmup models and connections
    logger.info("Đang Warmup hệ thống...")
    try:
        from app.services.rag.embeddings import get_model as get_embedding_model
        from app.services.rag.bm25 import get_bm25_service
        from app.services.chatbot.engine import get_llm
        from app.services.graphrag.knowledge_graph import get_knowledge_graph
        from app.services.rag.retriever import get_qdrant_client
        import concurrent.futures
        
        # 1. Nạp song song các Model & Chỉ mục nặng vào RAM
        logger.info("Đang khởi tạo song song các mô hình AI và chỉ mục...")
        
        def load_embeddings():
            logger.info("Đang nạp Dek21 Embedding Model vào RAM...")
            get_embedding_model()
            logger.info("Nạp Dek21 Embedding Model hoàn tất.")
            
        def load_bm25():
            logger.info("Đang nạp BM25 Service & chỉ mục corpus vào RAM...")
            get_bm25_service()
            logger.info("Nạp BM25 Service hoàn tất.")
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(load_embeddings),
                executor.submit(load_bm25)
            ]
            concurrent.futures.wait(futures)
        
        # 2. Tải Gemini
        logger.info("Khởi tạo cấu hình Gemini LLM...")
        get_llm()
        
        # 3. Kết nối CSDL & Bộ nhớ
        logger.info("Khởi tạo kết nối Qdrant, Neo4j & User Memory Collection...")
        get_qdrant_client()
        get_knowledge_graph()
        if Config.ENABLE_MEMORY:
            get_long_term_memory()._init_collection()
        
        logger.info("Warmup hoàn tất! Hệ thống sẵn sàng nhận yêu cầu siêu tốc.")
    except Exception as e:
        logger.error(f"Warmup thất bại: {e}")
        
    yield
    
    # Shutdown logic
    logger.info("Dừng hệ thống...")


def create_app():
    app = FastAPI(
        title="IT Law Chatbot API",
        description="API for consulting IT laws in Vietnam",
        version="1.0.0",
        lifespan=lifespan
    )

    # Rate limiter state
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS — restrict to configured origins (set ALLOWED_ORIGINS in .env)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=Config.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Api-Key"],
    )

    # Disable static files caching during development/updates
    @app.middleware("http")
    async def add_no_cache_headers(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith(".html") or path.endswith(".js") or path.endswith(".css"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": "Đã xảy ra lỗi hệ thống. Vui lòng thử lại sau."},
        )

    # ── /chat endpoint (đơn giản dành cho frontend) ──────────────────
    @app.post("/chat", response_model=ChatAnswer, tags=["Chat"], dependencies=[Depends(verify_api_key)])
    @limiter.limit(_rate_limit)
    async def chat_endpoint(request: Request, payload: ChatQuery):
        """
        Nhận câu hỏi (query) từ frontend và trả về câu trả lời (answer)
        cùng nguồn tham khảo và dữ liệu Knowledge Graph.

        Body JSON:
            { "query": "Hành vi nào bị nghiêm cấm trên không gian mạng?",
              "conversation_id": "optional-uuid",
              "user_id": "default_user" }

        Response JSON:
            { "answer": "...", "conversation_id": "...",
              "sources": [...], "graph_data": {...} }
        """
        query = (payload.query or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")

        try:
            user_id = payload.user_id or "default_user"
            result = generate_response(query, payload.conversation_id, user_id=user_id)
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

    # Include routers
    app.include_router(chat_router)
    app.include_router(memory_router)

    # Serve static files at root (css, js, images, etc.)
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

