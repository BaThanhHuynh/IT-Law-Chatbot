"""
Query expansion utilities for Vietnamese IT Law domain.
Handles abbreviation expansion and query preprocessing to improve
retrieval across all search paths (vector, graph, entity).
"""

from app.core.logger import logger

# Vietnamese IT Law abbreviation mapping
ABBREVIATION_MAP = {
    "cntt": "công nghệ thông tin",
    "shtt": "sở hữu trí tuệ",
    "anm": "an ninh mạng",
    "attt": "an toàn thông tin",
    "tmdt": "thương mại điện tử",
    "gddt": "giao dịch điện tử",
    "dlcn": "dữ liệu cá nhân",
    "cnts": "công nghệ số",
    "bvqlntd": "bảo vệ quyền lợi người tiêu dùng",
    "iot": "internet vạn vật",
    "ai": "trí tuệ nhân tạo",
    "ip": "sở hữu trí tuệ",
    "drm": "quản lý quyền kỹ thuật số",
    "vpn": "mạng riêng ảo",
    "ddos": "tấn công từ chối dịch vụ",
    "atm": "máy rút tiền tự động",
    "ndt": "người dùng dịch vụ điện tử",
    "csdl": "cơ sở dữ liệu",
    "qtkd": "quản trị kinh doanh",
}


def expand_abbreviations(text: str) -> str:
    """
    Expand Vietnamese IT Law abbreviations in text.
    Case-insensitive matching, preserves original structure.
    
    Example:
        "Quyền SHTT trong CNTT" → "Quyền sở hữu trí tuệ trong công nghệ thông tin"
    """
    words = text.split()
    expanded = []
    
    for word in words:
        # Check if the word (case-insensitive, stripped of punctuation) is an abbreviation
        clean_word = word.lower().strip(".,;:!?()\"\'")
        if clean_word in ABBREVIATION_MAP:
            expanded.append(ABBREVIATION_MAP[clean_word])
        else:
            expanded.append(word)
    
    result = " ".join(expanded)
    
    if result != text:
        logger.info(f"[QueryExpand] '{text}' → '{result}'")
    
    return result


def get_expanded_queries(query: str) -> list:
    """
    Generate expanded query variants from abbreviations.
    Returns [original, expanded] if different, else [original].
    """
    expanded = expand_abbreviations(query)
    if expanded != query:
        return [query, expanded]
    return [query]


# Keyword-triggered static query rules
# Format: (set of trigger keywords, list of static queries to inject)
_DOMAIN_STATIC_RULES = [
    (
        {"shtt", "sở hữu trí tuệ", "bản quyền", "quyền tác giả", "cntt", "công nghệ thông tin"},
        [
            "bảo vệ quyền sở hữu trí tuệ trong lĩnh vực công nghệ thông tin Luật Công nghệ thông tin 2006",
            "quyền sao chép phần mềm bảo hộ chương trình máy tính tác phẩm văn học",
        ]
    ),
    (
        {"an ninh mạng", "anm", "không gian mạng", "mã độc", "tấn công mạng"},
        [
            "hành vi bị nghiêm cấm trên không gian mạng Luật An ninh mạng",
            "phòng ngừa xử lý tấn công mạng an ninh quốc gia",
        ]
    ),
    (
        {"giao dịch điện tử", "gddt", "hợp đồng điện tử", "chữ ký số", "chữ ký điện tử"},
        [
            "giá trị pháp lý hợp đồng điện tử chữ ký số Luật Giao dịch điện tử",
            "điều kiện chứng thực chữ ký điện tử tổ chức cung cấp dịch vụ",
        ]
    ),
    (
        {"dữ liệu cá nhân", "dlcn", "bảo vệ dữ liệu", "thông tin cá nhân", "quyền riêng tư"},
        [
            "bảo vệ dữ liệu cá nhân quyền của chủ thể dữ liệu Luật Bảo vệ dữ liệu cá nhân",
            "xử lý dữ liệu cá nhân nhạy cảm đồng ý của chủ thể",
        ]
    ),
    (
        {"thương mại điện tử", "tmdt", "website thương mại", "sàn thương mại"},
        [
            "điều kiện cấp phép hoạt động website thương mại điện tử Nghị định",
            "quyền nghĩa vụ thương nhân kinh doanh dịch vụ thương mại điện tử",
        ]
    ),
]


def get_domain_static_queries(query: str) -> list:
    """
    Rule-based: detect topic keywords and inject guaranteed static queries
    targeting specific laws. Independent of LLM — always consistent.

    Example:
        query = "Quyền SHTT trong CNTT..." 
        → ["bảo vệ quyền sở hữu trí tuệ trong lĩnh vực công nghệ thông tin Luật CNTT 2006",
           "quyền sao chép phần mềm bảo hộ..."]
    """
    query_lower = query.lower()
    static_queries = []
    
    for trigger_keywords, queries in _DOMAIN_STATIC_RULES:
        # Check if ANY trigger keyword appears in the query
        if any(kw in query_lower for kw in trigger_keywords):
            static_queries.extend(queries)
            logger.info(f"[DomainStatic] Injected {len(queries)} static queries for topic: {list(trigger_keywords)[:3]}")
    
    return static_queries

