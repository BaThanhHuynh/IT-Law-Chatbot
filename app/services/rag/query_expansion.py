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
        logger.info(f"[QueryExpand] '{text}' -> '{result}'")
    
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
    # Rule 5a: Cài đặt phần mềm trái phép / xâm nhập chiếm quyền hệ thống
    # → Điều 80 NĐ 15/2020: Vi phạm quy định về cung cấp, sử dụng trái phép thông tin trên mạng
    # (Khoản 2a: truy cập trái phép chiếm quyền điều khiển thiết bị số — phạt 30-50 triệu)
    (
        {"cài đặt phần mềm", "cài lén", "cài phần mềm", "cài vào máy", "cài vào hệ thống",
         "phần mềm trái phép", "xâm nhập", "truy cập trái phép", "chiếm quyền",
         "chiếm quyền điều khiển", "tấn công hệ thống", "hack"},
        [
            "truy cập trái phép chiếm quyền điều khiển thiết bị số thay đổi xóa thông tin Điều 80 Nghị định 15 2020",
            "hành vi bị cấm tấn công chiếm quyền điều khiển phá hoại hệ thống thông tin Điều 7 Luật An toàn thông tin mạng",
            "xử phạt sử dụng trái phép thông tin trên mạng bẻ khóa mật khẩu Nghị định 15 2020",
        ]
    ),
    # Rule 5b: Phát tán mã độc / phần mềm độc hại
    # → Điều 83 NĐ 15/2020: Tổ chức không phòng ngừa phần mềm độc hại
    # → Điều 7 Khoản 4 Luật ATTTM: Nghiêm cấm phát tán phần mềm độc hại
    (
        {"phần mềm độc hại", "mã độc", "gián điệp", "phát tán", "ransomware", "trojan", "virus máy tính"},
        [
            "phát tán phần mềm độc hại mã độc hành vi bị nghiêm cấm Điều 7 Luật An toàn thông tin mạng",
            "vi phạm quy định phòng ngừa phát hiện ngăn chặn xử lý phần mềm độc hại Điều 83 Nghị định 15 2020",
            "gửi phát tán thư điện tử rác tin nhắn rác phần mềm độc hại xử phạt Nghị định 15 2020",
        ]
    ),
    # Rule 6: Tấn công từ chối dịch vụ / cản trở hệ thống thông tin
    (
        {"tấn công mạng", "ddos", "tấn công từ chối dịch vụ", "cản trở dịch vụ", "làm tê liệt"},
        [
            "tấn công từ chối dịch vụ cản trở hoạt động hệ thống thông tin Điều 80 Nghị định 15 2020",
            "hành vi bị cấm gây ảnh hưởng cản trở hoạt động hệ thống thông tin Điều 7 Luật An toàn thông tin mạng",
        ]
    ),
    # Rule 7: Tạo lập/viết chương trình vi rút máy tính, phần mềm độc hại/gây hại
    # → Điều 77 NĐ 15/2020: Vi phạm quy định về bảo vệ quyền, lợi ích hợp pháp và hỗ trợ người sử dụng sản phẩm, dịch vụ CNTT
    (
        {"viết chương trình", "lập trình chương trình", "tạo ra phần mềm", "viết phần mềm",
         "lập trình phần mềm", "viết virus", "tạo virus", "lập trình virus"},
        [
            "tạo ra hoặc cài đặt hoặc phát tán chương trình vi rút máy tính hoặc phần mềm gây hại vào thiết bị số Điều 77 Nghị định 15 2020",
            "hình thức xử phạt bổ sung tịch thu tang vật phương tiện vi phạm hành chính Điều 77 Nghị định 15 2020",
            "biện pháp khắc phục hậu quả buộc khôi phục lại tình trạng ban đầu đã bị thay đổi Điều 77 Nghị định 15 2020",
        ]
    ),
    # Rule 8: Cản trở dịch vụ chữ ký số / chứng thực chữ ký số
    # → Điều 79 NĐ 15/2020: Vi phạm quy định về an toàn, an ninh trong giao dịch điện tử sử dụng chữ ký số, chứng thư số
    (
        {"cản trở chữ ký số", "cản trở dịch vụ chữ ký", "ngưng hoạt động chữ ký số", "không hoạt động chữ ký số",
         "tấn công chữ ký số", "phá hoại chữ ký số", "cản trở chứng thực", "phá hoại chứng thực"},
        [
            "cản trở trái pháp luật hoạt động cung cấp và sử dụng dịch vụ chứng thực chữ ký số Điều 79 Nghị định 15 2020",
            "vi phạm quy định về an toàn an ninh trong giao dịch điện tử sử dụng chữ ký số chứng thư số Điều 79 Nghị định 15 2020",
        ]
    ),
]


# Word-pair triggers: if ALL words in a pair appear in the query, trigger the rule
# Format: (list of word-pair sets, index into _DOMAIN_STATIC_RULES to trigger)
# NOTE: Rule indices:
#   0=SHTT, 1=ANM, 2=GDDT, 3=DLCN, 4=TMDT, 5=Cài phần mềm/xâm nhập, 6=Mã độc, 7=DDoS, 8=Tạo phần mềm gây hại, 9=Cản trở chữ ký số
_WORD_PAIR_TRIGGERS = [
    # "phần mềm" + "cài/hệ thống/máy tính" → rule 5 (xâm nhập/cài phần mềm trái phép)
    ({"phần mềm", "cài đặt"}, 5),
    ({"phần mềm", "cài"},     5),
    ({"phần mềm", "hệ thống"}, 5),
    ({"phần mềm", "máy tính"}, 5),
    # "viết phần mềm" + "cài" → rule 5
    ({"viết", "phần mềm"},   5),
    # "truy cập" + "tài khoản/hệ thống" (khác chiếm đoạt tài sản) → rule 5
    ({"truy cập", "hệ thống"}, 5),
    ({"truy cập", "máy tính"}, 5),
    # "phát tán" + "phần mềm" → rule 6 (mã độc)
    ({"phát tán", "phần mềm"}, 6),
    ({"phần mềm", "độc hại"}, 6),
    
    # Word-pair triggers for Rule 7 (index 8):
    ({"viết", "chương trình"}, 8),
    ({"tạo", "chương trình"}, 8),
    ({"viết", "phần mềm"}, 8),
    
    # Word-pair triggers for Rule 8 (index 9):
    ({"chữ ký số", "cản trở"}, 9),
    ({"chữ ký số", "xâm nhập"}, 9),
    ({"chữ ký số", "hoạt động"}, 9),
    ({"chữ ký số", "tấn công"}, 9),
    ({"chứng thực", "cản trở"}, 9),
    ({"chứng thực", "xâm nhập"}, 9),
    ({"chứng thực", "hoạt động"}, 9),
    ({"chứng thực", "tấn công"}, 9),
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
    triggered_indices = set()
    
    # Check standard keyword triggers
    for idx, (trigger_keywords, queries) in enumerate(_DOMAIN_STATIC_RULES):
        if any(kw in query_lower for kw in trigger_keywords):
            static_queries.extend(queries)
            triggered_indices.add(idx)
            logger.info(f"[DomainStatic] Injected {len(queries)} static queries for topic: {list(trigger_keywords)[:3]}")
    
    # Check word-pair triggers (for vague queries where keywords are separated)
    for word_pair, rule_idx in _WORD_PAIR_TRIGGERS:
        if rule_idx not in triggered_indices:
            if all(w in query_lower for w in word_pair):
                queries = _DOMAIN_STATIC_RULES[rule_idx][1]
                static_queries.extend(queries)
                triggered_indices.add(rule_idx)
                logger.info(f"[DomainStatic] Word-pair trigger {word_pair} -> injected {len(queries)} static queries")
    
    return static_queries

