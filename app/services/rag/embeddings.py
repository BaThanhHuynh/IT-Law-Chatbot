import math
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import Config

_model = None


def get_model():
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {Config.EMBEDDING_MODEL}...")
        _model = SentenceTransformer(Config.EMBEDDING_MODEL)
        print("[Embeddings] Model loaded successfully.")
    return _model


def get_embedding(text: str) -> np.ndarray:
    """Generate embedding vector for a text string."""
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)


def get_embeddings_batch(texts: list) -> list:
    """Generate embeddings for a batch of texts."""
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [np.array(e, dtype=np.float32) for e in embeddings]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def calibrate_score(raw_score: float) -> float:
    """
    Calibrate raw cosine similarity to a user-friendly confidence score.

    Sentence-transformer models with cosine similarity typically return:
    - 0.5-0.7  for highly relevant results
    - 0.35-0.5 for moderately relevant results
    - <0.3     for weakly relevant results

    This sigmoid maps those ranges to a more intuitive display scale:
    - 0.5  raw → ~0.88 display
    - 0.45 raw → ~0.82 display
    - 0.4  raw → ~0.73 display
    - 0.35 raw → ~0.62 display
    - 0.3  raw → ~0.50 display
    """
    calibrated = 1 / (1 + math.exp(-10 * (raw_score - 0.3)))
    return round(min(max(calibrated, 0.01), 0.99), 3)



