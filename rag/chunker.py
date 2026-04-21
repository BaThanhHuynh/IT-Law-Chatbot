import re
from config import Config


def chunk_by_article(text: str) -> list:
    """
    Chunk law text by articles (Điều).
    Each chunk contains one article with its full content.
    """
    # Pattern to match "Điều X." or "Điều X:" at the start of a line
    pattern = r'(?=(?:^|\n)(Điều\s+\d+[\.:]))'
    parts = re.split(pattern, text)

    chunks = []
    current_chunk = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(r'^Điều\s+\d+[\.:]\s*', part):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = part
        else:
            current_chunk += "\n" + part

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_document(text: str, chunk_size: int = None, overlap: int = None) -> list:
    """
    Chunk document by fixed size with overlap.
    Falls back to this when article-based chunking produces chunks that are too large.
    """
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = overlap or Config.CHUNK_OVERLAP

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def smart_chunk(text: str, max_chunk_size: int = 800) -> list:
    """
    Smart chunking: first try article-based, then split large articles.
    Returns list of dicts with content and metadata.
    """
    article_chunks = chunk_by_article(text)
    result = []

    for chunk in article_chunks:
        # Extract article number from chunk
        article_match = re.match(r'(Điều\s+(\d+)[\.:]\s*(.*))', chunk.split('\n')[0])
        article = article_match.group(1).split('.')[0].strip() if article_match else None
        article_num = article_match.group(2) if article_match else None

        if len(chunk) > max_chunk_size:
            sub_chunks = chunk_document(chunk, chunk_size=max_chunk_size, overlap=100)
            for i, sub in enumerate(sub_chunks):
                result.append({
                    "content": sub,
                    "article": article,
                    "article_num": article_num,
                    "part": i + 1,
                })
        else:
            result.append({
                "content": chunk,
                "article": article,
                "article_num": article_num,
                "part": 1,
            })

    return result
