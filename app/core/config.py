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
    API_DEBUG = os.getenv("API_DEBUG", "true").lower() == "true"

    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "embedding_model"))
    EMBEDDING_DIM = 768

    # RAG settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    TOP_K_RESULTS = 5
