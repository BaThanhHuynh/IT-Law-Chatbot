"""
Memory subsystem for IT Law Chatbot (inspired by Mem0).
Provides multi-layer user memory (Short-Term Working Memory & Persistent Long-Term Vector Memory).
"""
import threading
from app.services.memory.short_term import ShortTermMemory, SessionState
from app.services.memory.long_term import LongTermMemory
from app.services.memory.extractor import MemoryExtractor

_short_term_instance = None
_short_term_lock = threading.Lock()

_long_term_instance = None
_long_term_lock = threading.Lock()

_extractor_instance = None
_extractor_lock = threading.Lock()


def get_short_term_memory() -> ShortTermMemory:
    """Get or initialize singleton ShortTermMemory."""
    global _short_term_instance
    if _short_term_instance is None:
        with _short_term_lock:
            if _short_term_instance is None:
                _short_term_instance = ShortTermMemory()
    return _short_term_instance


def get_long_term_memory() -> LongTermMemory:
    """Get or initialize singleton LongTermMemory."""
    global _long_term_instance
    if _long_term_instance is None:
        with _long_term_lock:
            if _long_term_instance is None:
                _long_term_instance = LongTermMemory()
    return _long_term_instance


def get_memory_extractor() -> MemoryExtractor:
    """Get or initialize singleton MemoryExtractor."""
    global _extractor_instance
    if _extractor_instance is None:
        with _extractor_lock:
            if _extractor_instance is None:
                _extractor_instance = MemoryExtractor()
    return _extractor_instance


__all__ = [
    "ShortTermMemory",
    "SessionState",
    "LongTermMemory",
    "MemoryExtractor",
    "get_short_term_memory",
    "get_long_term_memory",
    "get_memory_extractor",
]
