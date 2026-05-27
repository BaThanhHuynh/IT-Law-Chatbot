import torch
from sentence_transformers import CrossEncoder
from app.core.config import Config
from app.core.logger import logger

_reranker_service = None

class RerankerService:
    """Service wrapper for lazy-loading BAAI/bge-reranker-v2-m3 and re-ranking documents."""
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self):
        """Lazy-load the cross-encoder reranker model."""
        if self.model is None:
            model_name = Config.RERANKER_MODEL
            logger.info(f"[Reranker] Loading reranker model: {model_name} on device: {self.device}...")
            # CrossEncoder handles device detection, but we pass it explicitly to be safe
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info("[Reranker] Reranker model loaded successfully.")
        return self.model

    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        """
        Re-rank candidate documents against the query.
        
        Args:
            query: User's search query (or expanded query)
            candidates: List of dictionary records (each must have a 'content' key)
            top_k: Number of sorted records to return
            
        Returns:
            List of sorted candidate dictionaries with updated 'score' keys
        """
        if not candidates:
            return []
            
        try:
            model = self.get_model()
            
            # Prepare pairs of (query, document)
            pairs = []
            for c in candidates:
                content = c.get("content", "")
                # Clean or truncate content if too long (bge-reranker-v2-m3 handles up to 8192 tokens, but truncating keeps it fast)
                pairs.append((query, content[:1500]))
                
            logger.info(f"[Reranker] Re-ranking {len(candidates)} candidates against query '{query[:50]}...'")
            
            # Predict scores (higher score means more relevant)
            # bge-reranker returns raw logits or sigmoid values depending on version; we apply a standard sigmoid
            # to map logits (normally in [-8, 4]) to a pseudo-probability in [0.0, 1.0] range
            import numpy as np
            scores = model.predict(pairs)
            
            # Map raw logits using sigmoid function
            probs = 1.0 / (1.0 + np.exp(-np.array(scores, dtype=np.float32)))
            
            # Update candidates with new scores
            for i, p_score in enumerate(probs):
                candidates[i]["score"] = float(p_score)
                # Keep track of original source to assist in logging/debugging
                if "_source" not in candidates[i]:
                    candidates[i]["_source"] = "vector"
                    
            # Sort candidates by score descending
            reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
            
            logger.info(f"[Reranker] Re-ranking completed. Top score: {reranked[0]['score'] if reranked else 0.0}")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"[Reranker Error] Failed to re-rank: {e}")
            # Fallback: return top_k candidates by their original Qdrant/BM25 scores
            return candidates[:top_k]

def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
