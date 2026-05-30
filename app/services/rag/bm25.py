import json
import os
import re
import math
from app.core.config import Config
from app.core.logger import logger

class BM25Okapi:
    """Standard BM25Okapi implementation in pure Python."""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size
        self.doc_freqs = {}
        self.idf = {}
        # Inverted index: term -> list of (doc_idx, term_frequency).
        # Lets get_scores touch only documents that actually contain a query
        # term, instead of scanning the whole corpus once per term.
        self.postings = {}

        # Build index
        for idx, doc in enumerate(corpus):
            frequencies = {}
            for term in doc:
                frequencies[term] = frequencies.get(term, 0) + 1

            for term, tf in frequencies.items():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                self.postings.setdefault(term, []).append((idx, tf))

        # Calculate IDF
        for term, freq in self.doc_freqs.items():
            # Standard BM25 IDF formula with smoothing
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query):
        scores = [0.0] * self.corpus_size
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for term in query:
            idf_val = self.idf.get(term)
            if idf_val is None:
                continue
            # Only iterate documents that contain this term (inverted index).
            for idx, tf in self.postings[term]:
                doc_len = self.doc_lengths[idx]
                denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
                scores[idx] += idf_val * (tf * (k1 + 1) / denominator)
        return scores


_bm25_service = None

class BM25Service:
    """Service wrapper for initializing and querying BM25 over the static law chunks."""
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self._load_corpus()

    def _load_corpus(self):
        try:
            corpus_path = Config.BM25_CORPUS_PATH
            if not os.path.exists(corpus_path):
                logger.error(f"[BM25] Corpus file not found at {corpus_path}")
                return
            
            logger.info(f"[BM25] Loading corpus from {corpus_path}...")
            tokenized_corpus = []
            
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    chunk = json.loads(line)
                    self.chunks.append(chunk)
                    
                    # Extract text for tokenization
                    text = chunk.get("text", "")
                    tokens = self.tokenize(text)
                    tokenized_corpus.append(tokens)
            
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"[BM25] Corpus loaded successfully. Total documents: {len(self.chunks)}")
        except Exception as e:
            logger.error(f"[BM25 Error] Failed to load corpus: {e}")

    def tokenize(self, text: str) -> list:
        # Lowercase and clean punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if w]

    def search(self, query: str, top_k: int = 20) -> list:
        if not self.bm25 or not self.chunks:
            logger.warning("[BM25] Index not initialized. Returning empty search results.")
            return []
            
        tokens = self.tokenize(query)
        if not tokens:
            return []
            
        scores = self.bm25.get_scores(tokens)
        
        # Sort by score and take top_k
        scored_docs = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = []
        for idx, score in scored_docs:
            if score <= 0.0:
                continue
                
            chunk = self.chunks[idx]
            p = chunk.get("payload", {})
            
            # Format to match standard Qdrant results schema in retriever.py
            article = f"Điều {p.get('dieu_so', '')}" + (f". {p.get('dieu_ten', '')}" if p.get('dieu_ten') else "") if p.get('dieu_so') else ""
            chapter = f"Chương {p.get('chuong_so', '')}" + (f". {p.get('chuong_ten', '')}" if p.get('chuong_ten') else "") if p.get('chuong_so') else ""
            
            results.append({
                "chunk_id": chunk.get("id"),
                "content": p.get("full_dieu_text") or p.get("noi_dung_chunk") or chunk.get("text", ""),
                "context_text": p.get("context_text", ""),
                "dieu_so": p.get("dieu_so", ""),
                "dieu_ten": p.get("dieu_ten", ""),
                "chuong_so": p.get("chuong_so", ""),
                "chuong_ten": p.get("chuong_ten", ""),
                "article": article,
                "chapter": chapter,
                "doc_title": p.get("ten_van_ban", ""),
                "so_hieu": p.get("so_hieu", ""),
                "loai_van_ban": p.get("loai_van_ban", ""),
                "trang_thai": p.get("trang_thai", ""),
                "nhom": p.get("nhom", ""),
                # Store the raw score
                "score": float(score),
                "_source": "bm25",
            })
            
        return results

def get_bm25_service() -> BM25Service:
    global _bm25_service
    if _bm25_service is None:
        _bm25_service = BM25Service()
    return _bm25_service
