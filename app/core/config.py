import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Qdrant Database Service
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = "it_law_chunks"
    QDRANT_HISTORY_COLLECTION = "chat_history"

    # Chat History (JSON)
    CHAT_HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "chat_history.json")

    # Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # API Settings
    API_PORT = int(os.getenv("API_PORT", "5000"))
    API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

    # Security — CORS & Authentication
    # Comma-separated list of allowed origins, e.g. "http://localhost:5000,https://myapp.com"
    ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5000").split(",") if o.strip()]
    # Optional API key for protecting endpoints. Leave empty to disable authentication.
    API_KEY = os.getenv("API_KEY", "")

    # Rate Limiting (requests per minute per IP)
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "embedding_model"))
    EMBEDDING_DIM = 768
    # Max number of embeddings to keep in LRU cache (reduces repeated model inference)
    EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "512"))

    # Reranker model
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # BM25 Corpus Path
    BM25_CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "law_crawler", "data", "law_chunks.jsonl")

    # RAG settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    TOP_K_RESULTS = 5

    # Input validation
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
