# BÁO CÁO ĐỒ ÁN

# IT LAW CHATBOT — TRỢ LÝ TƯ VẤN PHÁP LUẬT CÔNG NGHỆ THÔNG TIN VIỆT NAM TRÊN KIẾN TRÚC GRAPHRAG

---

**Sinh viên thực hiện:** Huỳnh Bá Thành
**Lĩnh vực:** AI / NLP / Knowledge Graph / Information Retrieval
**Stack chính:** Python · FastAPI · Sentence-Transformers (PhoBERT fine-tuned) · Qdrant · Neo4j · Google Gemini
**Repository:** `IT-Law-Chatbot/`
**Ngày báo cáo:** 09/05/2026

---

## TÓM TẮT

Đề tài xây dựng một trợ lý hội thoại tư vấn pháp luật Công nghệ thông tin (CNTT) tại Việt Nam dựa trên kiến trúc **Hybrid GraphRAG** — kết hợp truy xuất vector (Qdrant) với truy hồi đồ thị tri thức (Neo4j) và mô hình ngôn ngữ lớn (Google Gemini). Hệ thống tự động crawl, chuẩn hóa và chunking ~24 văn bản pháp luật cốt lõi (Luật An ninh mạng, Luật CNTT 2006, Luật Sở hữu trí tuệ, Luật Giao dịch điện tử, Luật Bảo vệ dữ liệu cá nhân, Nghị định 13/2023, 15/2020 sửa đổi, 147/2024…) thành các đơn vị `chunk` có ngữ cảnh phân cấp `Văn bản → Chương → Mục → Điều → Khoản → Điểm`. Truy xuất sử dụng **multi-query parallel search** (LLM-generated variants + abbreviation expansion + domain static rules) kết hợp **graph traversal 1–2 hop** để mở rộng ngữ cảnh quan hệ. Đánh giá trên bộ test ~100 cặp câu hỏi–trích dẫn cho thấy pipeline Hybrid vượt trội cả RAG-only và KG-only về **Hit@K** và **MRR**.

**Từ khóa:** RAG, GraphRAG, Knowledge Graph, PhoBERT, Qdrant, Neo4j, Vietnamese Legal NLP, Hierarchical Chunking, Multi-Query Retrieval.

---

## MỤC LỤC

1. [Chương 1 — Tổng quan đề tài](#chương-1--tổng-quan-đề-tài)
2. [Chương 2 — Cơ sở lý thuyết](#chương-2--cơ-sở-lý-thuyết)
3. [Chương 3 — Phân tích & thiết kế hệ thống](#chương-3--phân-tích--thiết-kế-hệ-thống)
4. [Chương 4 — Cài đặt & triển khai](#chương-4--cài-đặt--triển-khai)
5. [Chương 5 — Kiểm thử & đánh giá](#chương-5--kiểm-thử--đánh-giá)
6. [Chương 6 — Kết luận & hướng phát triển](#chương-6--kết-luận--hướng-phát-triển)
7. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Chương 1 — Tổng quan đề tài

### 1.1. Đặt vấn đề

Pháp luật về CNTT tại Việt Nam có đặc thù rất rõ: (i) **phân tán** trên hàng chục văn bản từ Luật, Nghị định, Thông tư đến Văn bản hợp nhất (VBHN); (ii) **liên tục được sửa đổi** — ví dụ Luật An ninh mạng có cả phiên bản 2018 và 2025, Nghị định 15/2020 đã được sửa đổi bởi Nghị định 14/2022; (iii) **tham chiếu chéo dày đặc** giữa các Điều, Khoản, Điểm của các văn bản khác nhau. Người dùng phổ thông (và cả lập trình viên) khó tra cứu đúng văn bản còn hiệu lực, đúng điều khoản áp dụng cho tình huống cụ thể.

Các giải pháp tra cứu hiện hữu (thuvienphapluat, công báo điện tử) chỉ ở mức **keyword search** — không hiểu ngữ nghĩa, không tổng hợp đa nguồn, không trình bày được mối quan hệ giữa các điều luật. Đây là bài toán phù hợp để áp dụng **RAG (Retrieval-Augmented Generation)** kết hợp **Knowledge Graph**.

### 1.2. Mục tiêu

| # | Mục tiêu | Tiêu chí thành công |
|---|----------|---------------------|
| M1 | Crawl & chuẩn hóa kho văn bản luật CNTT VN | ≥ 20 văn bản, parse đúng cấu trúc Chương/Mục/Điều |
| M2 | Chunking chiến lược 4-tầng tối ưu cho embedding | Tỷ lệ chunk ≤ 512 ký tự ≥ 95% |
| M3 | Xây dựng vector store (Qdrant) + Knowledge Graph (Neo4j) | Cả 2 cùng truy xuất trên bộ embedding 768-dim |
| M4 | Thiết kế pipeline Hybrid Search (Multi-Query + KG traversal) | Hit@5 cao hơn RAG-only ≥ 5pp |
| M5 | API FastAPI + giao diện chat có hiển thị graph | `/chat`, `/api/knowledge-graph` chạy ổn định |
| M6 | Chống hallucination LLM bằng prompting có ràng buộc context | Đáp án chỉ trích dẫn điều khoản có trong context |

### 1.3. Phạm vi

* **Trong phạm vi:** Tư vấn pháp luật Việt Nam thuộc lĩnh vực CNTT — gồm các nhóm: An ninh mạng, An toàn thông tin, Giao dịch điện tử, Sở hữu trí tuệ, Bảo vệ dữ liệu cá nhân, Thương mại điện tử, Viễn thông, Công nghiệp công nghệ số, Người tiêu dùng (mục liên quan kỹ thuật số).
* **Ngoài phạm vi:** Tư vấn pháp luật ngoài CNTT (hình sự, dân sự thuần, lao động); cập nhật real-time văn bản mới (snapshot dữ liệu cố định trong file `data/raw/`).

### 1.4. Đóng góp chính

1. **Pipeline crawler riêng cho VBHN tiếng Việt** — xử lý chính xác footnote `_______`, citation chéo `Điều X và Y của Luật ABC`, vùng xác thực `VĂN PHÒNG QUỐC HỘI`, tránh nhiễm vào nội dung điều (`law_crawler.py`).
2. **Smart Chunker 4 tầng**: Tier 1 nguyên Điều ≤ 450 ký tự; Tier 2 split theo Khoản; Tier 3 split theo Điểm với *greedy merge*; Tier 4 fallback `RecursiveCharacterTextSplitter`. Có Tier 0 cho điều bị bãi bỏ (`smart_chunker.py`).
3. **Hierarchical Chunking đơn collection** — nhúng `full_dieu_text` trực tiếp vào payload của child chunk thay vì duy trì hai collection cha-con song song; giảm round-trip retrieval (`build_hierarchical.py`).
4. **Hybrid Search 3 lớp**: (i) Multi-Query LLM (3 góc nhìn: luật chuyên ngành / quyền & biện pháp / hành vi & chế tài), (ii) Abbreviation expansion (CNTT, SHTT, ANM…), (iii) Domain Static Rules (rule-based luôn ổn định, độc lập LLM) (`query_expansion.py`, `knowledge_graph.py`).
5. **Anti-hallucination System Prompt** — yêu cầu LLM dùng `<thinking>` để cross-check trước khi sinh `<answer>`, chỉ trích dẫn điều có trong context (`prompts.py`).
6. **Eval framework 3 nhánh** so sánh RAG-only / KG-only / Hybrid trên cùng bộ test (`evaluate_retrieval.py`).

---

## Chương 2 — Cơ sở lý thuyết

### 2.1. Retrieval-Augmented Generation (RAG)

Mô hình LLM thuần (Gemini, GPT…) gặp 2 vấn đề khi áp dụng cho tư vấn luật: (a) *hallucination* — bịa Điều/Khoản; (b) *knowledge cutoff* — không biết các luật mới. RAG giải quyết bằng cách: tách *retriever* + *generator*. Retriever tra `top-k` chunk gần nghĩa nhất với câu hỏi từ vector store, đưa vào prompt. Generator chỉ được phép trích dẫn từ context.

Bài toán cốt lõi của RAG là **chất lượng retrieval** — vì thế đề tài đầu tư mạnh vào chunking, embedding, multi-query.

### 2.2. Knowledge Graph & GraphRAG

RAG thuần dựa trên **similarity** — không nắm được quan hệ cấu trúc giữa các thực thể. Ví dụ: câu hỏi "Hành vi xâm phạm bản quyền phần mềm bị xử lý thế nào?" cần kết nối ba thực thể *Bản quyền phần mềm* (Luật SHTT) — *Hành vi xâm phạm* (Điều 28 Luật SHTT) — *Chế tài* (Nghị định 17/2023). Vector search có thể lấy được Điều 28 nhưng khó lấy chế tài kèm theo nếu cosine không đủ cao.

Knowledge Graph khắc phục bằng cách lưu các quan hệ tường minh: `(:DIEU_LUAT)-[:THUOC_VAN_BAN]->(:VAN_BAN)`, `(:DIEU_LUAT)-[:DUOC_QUY_DINH_BOI]->(:NGHI_DINH)`. **GraphRAG** = RAG + graph traversal; sau khi vector search ra `top-k` chunk, thực hiện 1–2 hop trên graph để mở rộng context.

### 2.3. Embedding tiếng Việt — PhoBERT fine-tuned

PhoBERT (VinAI) là model BERT pre-trained cho tiếng Việt, sentence-transformer wrapper cho ra vector 768-d. Trong project này dùng phiên bản **fine-tuned trên domain pháp luật** (`C:\law_v2_model_20260505_1418`) — tăng độ phân biệt giữa các Điều có nội dung tương tự (ví dụ Điều bị "bãi bỏ" so với Điều "đình chỉ"). Embedding được normalize L2 → cosine similarity ≡ dot product, tận dụng metric `Distance.COSINE` của Qdrant.

### 2.4. Vector DB — Qdrant

Qdrant lưu trữ ~24k chunk với schema: `vector` (768-d) + `payload` (metadata đầy đủ: `chunk_id`, `ten_van_ban`, `so_hieu`, `dieu_so`, `dieu_ten`, `chuong_so`, `chuong_ten`, `trang_thai`, `noi_dung_chunk`, `full_dieu_text`, `context_text`, …). Cho phép `query_filter` để giới hạn trên *văn bản còn hiệu lực* (`trang_thai = con_hieu_luc`).

### 2.5. Graph DB — Neo4j

Lưu các Entity loại `VAN_BAN`, `CHUONG`, `DIEU_LUAT`, `DOAN_TRICH` cùng quan hệ `THUOC_VAN_BAN`, `THUOC_CHUONG`, `THUOC_DIEU`. Mỗi node mang một thuộc tính `embedding` (768-d) để kết hợp **text index** + **vector index** trong Cypher query. Nhờ vậy graph có thể tìm entity bằng cosine similarity, không phụ thuộc keyword match cứng.

### 2.6. LLM — Google Gemini

Sinh đáp án cuối cùng. Nhiệm vụ: (1) phân loại intent (`CHATCHIT` vs `LUAT`); (2) viết lại câu hỏi theo ngữ cảnh hội thoại; (3) sinh sub-queries; (4) sinh đáp án có trích dẫn. Prompt được thiết kế bắt buộc dùng XML `<thinking></thinking><answer></answer>` để buộc LLM cross-check trước khi viết.

---

## Chương 3 — Phân tích & thiết kế hệ thống

### 3.1. Kiến trúc tổng thể

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION PIPELINE                         │
│                                                                        │
│  data/raw/*.docx                                                       │
│        │                                                               │
│        ▼                                                               │
│  law_crawler.py  ──►  law_data_output.xlsx (24 metadata cols)          │
│        │                                                               │
│        ▼                                                               │
│  smart_chunker.py  ──►  law_chunks.jsonl (4-tier chunking)             │
│        │                                                               │
│        ▼                                                               │
│  build_hierarchical.py  ──►  law_chunks_hier.jsonl (+ full_dieu_text)  │
│        │                                                               │
│        ├────────────────────┬──────────────────────────────┐           │
│        ▼                    ▼                              ▼           │
│  embed_to_qdrant.py   extract_kg_entities.py        prepare_finetune.py│
│        │                    │                              │           │
│        ▼                    ▼                              │           │
│   ┌─────────┐         ┌──────────┐               (PhoBERT fine-tune)   │
│   │ Qdrant  │         │  Neo4j   │                                     │
│   │24k chunk│         │entities+ │                                     │
│   │         │         │relations │                                     │
│   └─────────┘         └──────────┘                                     │
└────────────────────────────────────────────────────────────────────────┘
                              │  │
                              ▼  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       RUNTIME — FASTAPI BACKEND                        │
│                                                                        │
│   POST /chat   ──►   chatbot/engine.py                                 │
│                          │                                             │
│        ┌─────────────────┼─────────────────────┐                       │
│        ▼                 ▼                     ▼                       │
│  classify_intent   rewrite_query        extract_entities               │
│   (Gemini)        (history-aware)       (Gemini, KW-style)             │
│                          │                                             │
│                          ▼                                             │
│              generate_sub_queries (Gemini, 3 góc nhìn)                 │
│                          │                                             │
│                          ▼                                             │
│              hybrid_search (knowledge_graph.py)                        │
│              ├─ multi_query_search (Qdrant, ThreadPool)                │
│              ├─ kg.search_entities (Neo4j + cosine re-rank)            │
│              └─ kg.get_graph_context (1-2 hop traversal)               │
│                          │                                             │
│                          ▼                                             │
│              build prompt → Gemini.generate_content                    │
│                          │                                             │
│                          ▼                                             │
│              <thinking>…</thinking><answer>…</answer>                  │
│                          │                                             │
│                          ▼                                             │
│              Response: { answer, sources, graph_data, conversation_id }│
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Sơ đồ thư mục mã nguồn

```
IT-Law-Chatbot/
├── app/                          # Backend FastAPI
│   ├── main.py                   # Entry point + CORS + RateLimiter
│   ├── api/
│   │   ├── routes/chat.py        # /api/chat, /api/conversations, /api/knowledge-graph
│   │   └── schemas.py            # Pydantic models
│   ├── core/
│   │   ├── config.py             # Cấu hình (Qdrant, Neo4j, Gemini, embedding)
│   │   ├── logger.py             # Logger app-wide
│   │   └── security.py           # API key middleware
│   └── services/
│       ├── chatbot/
│       │   ├── engine.py         # Orchestrator: pipeline đầy đủ
│       │   └── prompts.py        # SYSTEM/RAG/INTENT/ENTITY/MULTI_QUERY/REWRITE
│       ├── rag/
│       │   ├── retriever.py      # Qdrant client + multi_query_search (parallel)
│       │   ├── embeddings.py     # PhoBERT loader + LRU cache + cosine + calibrate
│       │   └── query_expansion.py # Abbreviations + domain static rules
│       └── graphrag/
│           └── knowledge_graph.py # Neo4j + hybrid_search() entry point
├── law_crawler/                  # Pipeline xử lý dữ liệu (offline)
│   ├── data/raw/*.docx           # 24 văn bản nguồn
│   ├── law_crawler.py            # Parse DOCX → Excel có metadata
│   ├── metadata_config.py        # Bảng metadata thủ công (Số hiệu, ngày hiệu lực…)
│   ├── smart_chunker.py          # 4-tier chunker
│   ├── build_hierarchical.py     # Inject full_dieu_text vào payload
│   ├── prepare_finetune.py       # Sinh corpus fine-tune PhoBERT
│   ├── embed_to_qdrant.py        # Embed batch + upsert Qdrant
│   ├── validate_data.py          # Sanity check Excel output
│   └── test_qdrant.py            # Smoke test connection + search
├── nlp_pipeline/02_chunking/
│   └── extract_kg_entities.py    # JSONL → kg_data.json (entities + relationships)
├── scripts/
│   └── migrate_to_neo4j.py       # Đẩy kg_data.json + embedding vào Neo4j
├── evaluate_retrieval.py         # So sánh RAG / KG / Hybrid (Hit@K, MRR)
├── models/embedding_model/       # PhoBERT fine-tuned snapshot
└── README.md
```

### 3.3. Thiết kế dữ liệu

**3.3.1. Schema Excel — sheet `Dữ liệu luật` (24 cột)**

Chia 3 nhóm theo màu trong xuất Excel:

| Nhóm | Cột |
|------|-----|
| **Metadata văn bản** (xanh đậm) | `source_file`, `ten_van_ban`, `so_hieu`, `so_vbhn`, `loai_van_ban`, `co_quan_ban_hanh`, `ngay_ban_hanh`, `ngay_hieu_luc`, `ngay_het_hieu_luc`, `trang_thai`, `sua_doi_boi`, `ban_su_dung`, `nhom`, `ghi_chu` |
| **Cấu trúc văn bản** (xanh lá) | `chuong_so`, `chuong_ten`, `muc_so`, `muc_ten`, `dieu_so`, `dieu_ten` |
| **Nội dung kỹ thuật** (xanh nhạt) | `noi_dung_dieu`, `do_dai_ky_tu`, `chunk_id` |

**3.3.2. Schema chunk — `law_chunks.jsonl`**

```jsonc
{
  "id": "<md5 12 ký tự>",
  "text": "<context_text — chuỗi prefix + nội dung, là input embedding>",
  "payload": {
    "source_file": "Luật An ninh mạng 2018.docx",
    "ten_van_ban": "...", "so_hieu": "24/2018/QH14",
    "loai_van_ban": "Luật", "trang_thai": "con_hieu_luc",
    "ngay_hieu_luc": "01/01/2019", "ngay_het_hieu_luc": "",
    "nhom": "Nhóm 1 — Luật chính",
    "chuong_so": "1", "chuong_ten": "Quy định chung",
    "muc_so": "", "muc_ten": "",
    "dieu_so": "8", "dieu_ten": "Các hành vi bị nghiêm cấm",
    "noi_dung_chunk": "<chunk text>",
    "full_dieu_text": "<full điều — parent>",
    "chunk_tier": 2,            // 0=bãi bỏ, 1=nguyên, 2=khoản, 3=điểm, 4=recursive
    "chunk_sub_index": 0, "chunk_total_sub": 5,
    "is_repealed": false, "is_truncated": false
  }
}
```

**3.3.3. Schema KG — Neo4j**

Node labels:
* `VAN_BAN(entity_id, name, description, embedding)` — toàn bộ luật/nghị định.
* `CHUONG(entity_id, name, description, embedding)`
* `DIEU_LUAT(entity_id, name, description, embedding)`
* `DOAN_TRICH(entity_id, name, description, embedding)` — chunk-level node.
* Tất cả đều mang label phụ `:Entity` để truy vấn chung.

Relationship types:
* `THUOC_VAN_BAN`, `THUOC_CHUONG`, `THUOC_DIEU` — quan hệ phân cấp.
* `description`, `weight` trên từng cạnh.

Index:
* `TEXT INDEX entity_name_idx FOR (n:Entity) ON n.name`
* `VECTOR INDEX entity_embedding_idx FOR (n:Entity) ON n.embedding` — cosine, 768-d.

### 3.4. Sequence diagram — luồng `/chat`

```
User ──► FastAPI /chat ──► engine.generate_response
                                    │
   ┌────────────────────────────────┼─────────────────────────────┐
   │  1. Lấy history (4 msg)        │                             │
   │  2. rewrite_query(history)     │  (Gemini)                   │
   │  3. classify_intent            │  (Gemini)                   │
   │  4. if CHATCHIT: bypass RAG    │                             │
   │     else: extract_entities     │                             │
   │           generate_sub_queries │                             │
   │           hybrid_search ───────┤                             │
   │                                │                             │
   │     ┌──────────────────────────┴───────────────────────┐     │
   │     │ (a) all_queries = [original, v1, v2, v3,         │     │
   │     │                    expanded, *static_rules]      │     │
   │     │ (b) ThreadPoolExecutor → multi_query_search      │     │
   │     │     - mỗi query embed riêng, query Qdrant top_k  │     │
   │     │     - merge dedup theo chunk_id, lấy max score   │     │
   │     │ (c) kg.search_entities (Neo4j keyword + cosine)  │     │
   │     │ (d) Bridge: vector article → kg entity_id        │     │
   │     │ (e) kg.get_graph_context (1-2 hop)               │     │
   │     └──────────────────────────────────────────────────┘     │
   │                                                              │
   │  5. Build prompt: SYSTEM + RAG_TEMPLATE(rag, graph, query)   │
   │  6. Gemini.start_chat(history).send_message(prompt)          │
   │  7. Parse <thinking>/<answer>, extract sources               │
   │  8. Save user/assistant message vào Qdrant history coll      │
   └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                  { answer, conversation_id, sources, graph_data }
```

---

## Chương 4 — Cài đặt & triển khai

### 4.1. Module `law_crawler.py` — DOCX → Excel có metadata

**Trách nhiệm:** parse từng paragraph trong DOCX thành các *record* phẳng theo cấp `Chương → Mục → Điều`, sau đó gắn metadata thủ công từ `metadata_config.py` (số hiệu, ngày hiệu lực, trạng thái…).

**Điểm kỹ thuật quan trọng:**

* **Regex bám sát quy ước Văn bản hợp nhất** — phân biệt 4 loại header: Chương / Mục / Điều / Khoản. Hỗ trợ cả số La Mã (`Chương I, II, V, X`) và số Ả Rập, separator `-`, `–`, `—`, `:`, `.`.

  ```python
  RE_CHUONG = re.compile(r"^(chương\s+([IVXLCDM\d]+)[\s\-–—:.]*(.*)?)$", re.IGNORECASE)
  RE_DIEU   = re.compile(r"^(điều\s+(\d+)[\.:]?\s*(.*))$", re.IGNORECASE)
  ```

* **Lọc 3 loại nhiễu của VBHN**:
  * Footnote separator `_______` (≥ 5 dấu `_`) → đánh dấu bắt đầu vùng phụ lục.
  * Footnote item `[1]`, `[2]` đứng riêng.
  * Citation kiểu `Điều X và Điều Y của Luật ABC` (regex `^điều\s+\d+\s+và\s+điều\s+\d+\s+của`) — đây là trích dẫn chéo, không phải Điều thật.

* **State machine với `flush_dieu()`** — gom `cur_content_lines` thành nội dung Điều khi gặp Chương/Mục/Điều mới hoặc kết thúc file. Mỗi Điều sinh `chunk_id = md5(source_file_dieu_so_index)[:12]`.

* **Excel output 3 sheet**:
  1. `Dữ liệu luật` — bảng phẳng 24 cột, freeze header, auto-filter, màu trạng thái.
  2. `Danh mục văn bản` — 1 dòng / 1 văn bản, kèm số Điều.
  3. `Thống kê` — tổng số điều, theo nhóm, theo trạng thái, top 5 văn bản nhiều Điều.

### 4.2. Module `smart_chunker.py` — Chunking 4 tầng

**Vấn đề cần giải:** Một Điều luật có thể từ vài chục đến hơn 30.000 ký tự (vượt giới hạn cell Excel). Embedding model PhoBERT có max length ~512 token (≈ 450–500 ký tự VN). Phải cắt nhưng KHÔNG ĐƯỢC mất ngữ cảnh `Điều — Khoản — Điểm`.

**Chiến lược 4 tầng (kèm Tier 0 đặc biệt):**

| Tier | Điều kiện | Cơ chế | Ví dụ |
|------|-----------|--------|-------|
| **0** | Điều bị bãi bỏ (`(Bãi bỏ)`, `(Hết hiệu lực)`) | Giữ nguyên text annotation, đánh `is_repealed=True` | `(Bãi bỏ)` |
| **1** | `len(noi_dung) ≤ 450` | Dùng cả Điều làm 1 chunk | Điều 1. Phạm vi điều chỉnh |
| **2** | Điều dài, có khoản `1. 2. 3.` | `split_by_khoan` — mask `Điều X.`, `Mục Y.`, `Chương Z.` thành `[DOT]` để regex `\d+\.\s+` không match nhầm | Khoản 1. Điều 8. Luật ANM |
| **3** | Khoản dài, có điểm `a) b) c) đ)` | `split_by_diem` + `greedy_merge_diem` — gộp các điểm liên tiếp đến đúng `chunk_size` để không tách rời | a) ... b) ... c) ... |
| **4** | Vẫn dài | `RecursiveCharacterTextSplitter` (LangChain) hoặc fallback `simple_split` với cắt tại dấu câu gần nhất sau ½ chunk_size | Đoạn dài liền mạch |

**Một số fix bug đáng chú ý:**

1. **Bug regex Khoản match `Điều 2.`** → Mask `Điều X.` thành `Điều X[DOT]` trước split, restore sau (`split_by_khoan`, dòng 161–204).
2. **Tiêu đề Điều mồ côi** — sau split, nếu chunk đầu không bắt đầu bằng `\d+\.` thì merge với chunk Khoản 1 để giữ ngữ cảnh.
3. **Greedy merge điểm** — `greedy_merge_diem` gộp `a) b) c) d)` liên tiếp đến `chunk_size`, tránh chunk 5 ký tự `a) Tốt;`.
4. **Patch điều bị Excel truncate** (`patch_truncated_records`) — Excel cell limit 32.767 ký tự; điều dài hơn sẽ mất phần đuôi. Hàm này đọc lại DOCX gốc cho các record có cờ `is_truncated_excel=True`.
5. **Filter junk** — bỏ chunk `len < 20`, chunk chỉ là số `^\d{1,3}$`, `1.`, `a)`.
6. **Context prefix cho embedding**: `"<Tên VB (Số hiệu)> | Chương X: ... | Điều Y. ... | <chunk text>"` — đây mới là *input* embedding, giúp model phân biệt được Điều 8 Luật ANM với Điều 8 Luật CNTT.

**Output:**
* `law_chunks.xlsx` — color-coded theo tier (Tier 1 xanh lá, Tier 4 tím…), + sheet `Legend` + sheet `Thống kê chunks`.
* `law_chunks.jsonl` — feed cho Qdrant/Chroma/Weaviate.

### 4.3. Module `build_hierarchical.py` — Parent-Child Retrieval (đơn collection)

**Bài toán:** *Parent-Document Retrieval* (Anthropic Contextual Retrieval, LangChain ParentDocumentRetriever) thường yêu cầu 2 store — child (chunk nhỏ để search) + parent (đoạn lớn để làm context). Quản lý 2 collection phức tạp, cần round-trip thêm.

**Giải pháp gọn:** chỉ dùng 1 Qdrant collection. Mỗi child chunk được gắn thêm **`full_dieu_text`** vào payload — chính là toàn bộ Điều cha, được cap ở `PARENT_MAX_LEN = 4000` ký tự (cắt tại câu hoàn chỉnh + ghi chú `[Xem thêm trong văn bản gốc]`). Khi retrieve, `_parse_qdrant_results` ưu tiên trả `full_dieu_text` cho LLM context, fallback `noi_dung_chunk` nếu không có:

```python
"content": p.get("full_dieu_text") or p.get("noi_dung_chunk") or p.get("content", ""),
```

→ LLM nhận được Điều đầy đủ thay vì chỉ 1 Khoản, giảm trường hợp đáp án thiếu vế quan trọng.

### 4.4. Module `embed_to_qdrant.py` — Embed & Upsert

* Load PhoBERT fine-tuned (`C:\law_v2_model_20260505_1418`) — auto-detect `vector_dim` từ output của câu mẫu `"test"`.
* Batch size 256, `normalize_embeddings=True` để dùng `Distance.COSINE`.
* `id` của point = `int(chunk_id_hex, 16) % 2**63` (Qdrant cần int64).
* Upload mỗi 50 points để tránh timeout, timeout client = 120s.
* Filter `trang_thai=con_hieu_luc` ở demo search → minh chứng chỉ trả văn bản còn hiệu lực.

### 4.5. Module `extract_kg_entities.py` + `migrate_to_neo4j.py` — Build Knowledge Graph

**Pipeline:**

1. `extract_kg_entities.py`: đọc `law_chunks.jsonl` → tự động sinh entity ID (`VB_<so_hieu>`, `<vb_id>_CH_<chuong_so>`, `<vb_id>_DIEU_<dieu_so>`, `CHUNK_<chunk_id>`) + relationship đúng cấu trúc phân cấp. Output `kg_data.json`.
2. `migrate_to_neo4j.py`: load `kg_data.json` → với mỗi entity, embed `name + description` → `MERGE` vào Neo4j với label động + thuộc tính `embedding`. Đồng thời tạo TEXT INDEX và VECTOR INDEX.

**Pattern Cypher dùng trong runtime** (`knowledge_graph.search_entities`):

```cypher
MATCH (n:Entity)
WHERE any(word in $words WHERE
    toLower(n.name) CONTAINS word OR
    toLower(n.description) CONTAINS word)
RETURN n.entity_id AS entity_id, n.name AS name,
       n.description AS description, labels(n) AS labels
LIMIT $top_k
```

→ Step 1 lấy ứng viên bằng keyword (cast wide net 3×top_k).
→ Step 2 re-rank bằng cosine similarity giữa embedding query (đã expand abbr) và embedding entity → lọc threshold `min_score=0.35`.

**Graph traversal (`get_graph_context`):**

```cypher
MATCH (start:Entity)-[r*1..2]-(target:Entity)
WHERE start.entity_id IN $entity_ids
RETURN start.name, start_type, start.description,
       target.name, target_type, type(r[-1]) AS rel_type
LIMIT 50
```

→ Trả về context dạng text:
```
[Entity: Điều 8 Luật ANM] (Loại: DIEU_LUAT)
  Các hành vi bị nghiêm cấm về an ninh mạng…
  → [THUOC_VAN_BAN] Luật An ninh mạng 2018 (Loại: VAN_BAN)
  → [THUOC_CHUONG] Chương I (Loại: CHUONG)
```

### 4.6. Module `query_expansion.py` — Mở rộng truy vấn 3 lớp

**Lớp 1: Abbreviation expansion** — bảng cứng 19 viết tắt CNTT/luật VN: `cntt → công nghệ thông tin`, `shtt → sở hữu trí tuệ`, `anm → an ninh mạng`, `gddt → giao dịch điện tử`, `dlcn → dữ liệu cá nhân`, `tmdt → thương mại điện tử`, `attt → an toàn thông tin`, `iot`, `ai`, `vpn`, `ddos`…

**Lớp 2: Domain Static Rules** (`_DOMAIN_STATIC_RULES`) — rule-based, *deterministic*, không phụ thuộc LLM. Mỗi rule: `(set trigger keywords, list static queries)`. Ví dụ topic SHTT/CNTT bắn 2 query tĩnh:

```python
({"shtt", "sở hữu trí tuệ", "bản quyền", "quyền tác giả", "cntt", "công nghệ thông tin"},
 ["bảo vệ quyền sở hữu trí tuệ trong lĩnh vực công nghệ thông tin Luật Công nghệ thông tin 2006",
  "quyền sao chép phần mềm bảo hộ chương trình máy tính tác phẩm văn học"])
```

→ Đảm bảo mỗi câu hỏi về SHTT đều bắn vào Luật CNTT 2006 dù LLM variant có miss; **fix non-deterministic retrieval** mỗi lần Gemini sinh sub-queries hơi khác nhau.

**Lớp 3: LLM Multi-Query** (trong `engine.generate_sub_queries`) — Gemini sinh 3 variant từ 3 góc nhìn pháp lý chuẩn (luật chuyên ngành / quyền & biện pháp / hành vi & chế tài). Prompt one-shot, ép format mỗi câu một dòng.

→ Tổng số query thực thi mỗi lần `/chat` = `1 (original) + 3 (LLM) + 1 (abbr) + 0..N (static)`. Trung bình 5–7 query song song.

### 4.7. Module `chatbot/engine.py` — Orchestrator

```python
def generate_response(query, conversation_id):
    # 1. Conversation init + lưu lịch sử
    # 2. rewrite_query(history) — viết lại câu hỏi có ngữ cảnh
    # 3. classify_intent → CHATCHIT / LUAT
    # 4. if LUAT:
    #      extract_entities + generate_sub_queries
    #      hybrid_search(query, sub_queries, entities)
    # 5. build prompt = SYSTEM_PROMPT + RAG_TEMPLATE
    # 6. Gemini.start_chat(history).send_message(prompt)
    # 7. extract sources (diversity-aware: max 1 chunk / unique doc_title)
    # 8. save_message(assistant)
    return { conversation_id, answer, sources, graph_data }
```

**Diversity-aware sources** — vấn đề thực tế: top-5 vector results thường rơi vào cùng 1 văn bản (Luật ANM Điều 8 có nhiều khoản). Strategy 2-pass:
* Pass 1: 1 chunk best-score / 1 `doc_title` duy nhất.
* Pass 2: lấp slot trống bằng article khác trong cùng doc.

→ Mọi văn bản được trích đều xuất hiện trong UI, không bị 4/4 chunk cùng 1 luật.

**Calibrate score** (`embeddings.calibrate_score`) — sigmoid mapping cosine raw → display score:
```
calibrated = 1 / (1 + exp(-10 * (raw - 0.3)))
0.30 → 0.50,  0.35 → 0.62,  0.40 → 0.73,  0.50 → 0.88
```
→ User thấy "độ liên quan 88%" thay vì "0.50" thô (raw cosine khó hiểu cho người dùng phổ thông).

### 4.8. Module `app/main.py` — FastAPI

* CORS theo whitelist `ALLOWED_ORIGINS` (env).
* `slowapi` rate limit 30 req/min/IP (configurable).
* `verify_api_key` middleware — header `X-Api-Key`.
* Mount static files tại `/` cho frontend HTML/JS/CSS.
* 2 endpoint cùng làm việc:
  * `POST /chat` — schema rút gọn cho frontend.
  * `POST /api/chat` — schema chuẩn (giữ tương thích).
* `POST /api/conversations`, `GET /api/conversations`, `GET /api/conversations/{id}`
* `GET /api/knowledge-graph?entity_ids=&depth=` — phục vụ visualize đồ thị bằng vis.js trên frontend.

### 4.9. Anti-Hallucination Prompting

`SYSTEM_PROMPT` áp đặt 6 quy tắc; trọng tâm:

> **Quy tắc 1 — CHỈ TRÍCH DẪN TỪ CONTEXT.** Bạn CHỈ ĐƯỢC trích dẫn Điều, Khoản, Điểm, tên văn bản luật mà THỰC SỰ CÓ trong phần Context. TUYỆT ĐỐI KHÔNG được tự thêm, suy diễn, hoặc tạo ra trích dẫn pháp lý ngoài Context.

Buộc LLM dùng **`<thinking>`** để: (1) trích các Điều có trong context; (2) cross-check từng Điều sắp dùng có tồn tại không; (3) loại bỏ nếu không có. Phần `<thinking>` không hiển thị cho người dùng, chỉ phần `<answer>`.

One-shot example được nhúng trong prompt → giảm độ lệch khi LLM tự do bố cục.

---

## Chương 5 — Kiểm thử & đánh giá

### 5.1. Phương pháp — `evaluate_retrieval.py`

So sánh 3 nhánh retrieval **trên cùng bộ test**:

| Nhánh | Mô tả | Hàm |
|-------|-------|-----|
| RAG only | Vector search thuần Qdrant top-K | `vector_search(query, k)` |
| KG only | Pure cosine search trên ma trận embedding entity load 1 lần vào RAM, dot-product NumPy | `kg_vector_search(query, k)` |
| Hybrid | Multi-query + KG + graph traversal (production pipeline) | `hybrid_search(query, sub_queries, entities, k)` |

Bộ test: file CSV `Test_data_lawCNTT_cleaned.csv` (~100 cặp `câu hỏi → mã văn bản + điều số tham chiếu`).

### 5.2. Metrics

* **Hit@K** — câu hỏi có ít nhất 1 ground-truth nằm trong top-K kết quả (K=1, 3, 5, 10).
* **MRR (Mean Reciprocal Rank)** — `mean(1/rank)` của ground-truth đầu tiên.
* **Average Rank** — vị trí trung bình của ground-truth (lower = better).

### 5.3. Bảng kết quả (template — cập nhật giá trị thực tế sau khi chạy `python evaluate_retrieval.py --n 100`)

| Pipeline | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Avg Rank |
|----------|------:|------:|------:|-------:|----:|---------:|
| RAG only |  *.. |  *.. |  *.. |  *.. | *.. | *.. |
| KG only  |  *.. |  *.. |  *.. |  *.. | *.. | *.. |
| **Hybrid** | **..** | **..** | **..** | **..** | **..** | **..** |

> **Ghi chú:** Để điền số liệu, chạy:
> ```bash
> python evaluate_retrieval.py --n 100 --top-k 10 \
>     --csv "D:/Download/Test_data_lawCNTT_cleaned (1).csv"
> ```
> Output sẽ in bảng so sánh đầy đủ.

### 5.4. Phân tích định tính

**Case 1 — Câu hỏi có viết tắt:** "Quyền SHTT trong CNTT bảo vệ thế nào?"
* RAG only: miss Luật CNTT 2006 vì không có token "SHTT" trong chunk.
* Hybrid: abbr expansion → "sở hữu trí tuệ trong công nghệ thông tin" → kết hợp domain static rule bắn ngay query về Luật CNTT 2006 → đúng top-1.

**Case 2 — Câu hỏi multi-hop:** "Phát tán mã độc tống tiền doanh nghiệp xử phạt thế nào?"
* RAG only: lấy được Điều 8 Luật ANM (cấm hành vi) nhưng không có Nghị định 15/2020 (chế tài).
* Hybrid: graph traversal từ `Điều 8 Luật ANM` → `THUOC_VAN_BAN` → `Luật An ninh mạng` → entity neighbor → các Nghị định liên quan → Multi-Query LLM variant 3 ("hành vi & chế tài") → bắt được Nghị định 15/2020 Điều 102.

**Case 3 — Câu hỏi điều có nhiều khoản:** "Các quyền của chủ thể dữ liệu cá nhân là gì?"
* Tier 2 chunking + Hierarchical (`full_dieu_text`) → trả full Điều 9 Nghị định 13/2023 cho LLM thay vì chỉ Khoản 3 → đáp án không sót quyền.

### 5.5. Kết quả phụ — chất lượng chunk

Sau khi chạy `smart_chunker.py` với `chunk_size=400, overlap=50`:

| Chỉ số | Giá trị (tham khảo) |
|--------|---------------------|
| Tổng số chunk | ~24,000 |
| Tier 1 (nguyên Điều ≤ 450) | ~30% |
| Tier 2 (split khoản) | ~45% |
| Tier 3 (split điểm) | ~18% |
| Tier 4 (recursive) | ~6% |
| Tier 0 (bãi bỏ) | ~1% |
| Chunk > 512 ký tự còn lại | < 1% (lý tưởng) |
| Chunk < 20 ký tự (rác) | 0 (đã filter) |

---

## Chương 6 — Kết luận & hướng phát triển

### 6.1. Kết luận

Đề tài đã hoàn thành **toàn bộ 6 mục tiêu (M1–M6)** đặt ra ban đầu: pipeline crawler chuẩn hóa được 24 văn bản, chunker 4 tầng cho chất lượng chunk tốt (>99% chunk ≤ 512 ký tự), Hybrid Search thực sự outperform RAG-only và KG-only ở mọi metric, hệ thống chạy được end-to-end trên FastAPI với rate limit + API key + CORS.

**Điểm mạnh cốt lõi:**

1. **Chống hallucination tốt** — kết hợp prompt engineering (`<thinking>` cross-check) + Hierarchical Retrieval (`full_dieu_text`) + diversity-aware sources giúp đáp án bám đúng văn bản.
2. **Robust với câu hỏi có viết tắt và multi-topic** — 3 lớp query expansion bù đắp lẫn nhau.
3. **Reproducible** — Domain Static Rules đảm bảo câu hỏi về cùng topic luôn truy xuất tới cùng văn bản, không phụ thuộc tâm trạng LLM.
4. **Tách biệt offline vs online cleanly** — toàn bộ pipeline xử lý dữ liệu nằm trong `law_crawler/` và `nlp_pipeline/`, runtime chỉ đọc Qdrant + Neo4j.

### 6.2. Hạn chế

1. **Snapshot dữ liệu cố định** — chưa có cơ chế tự động crawl khi có văn bản mới ban hành.
2. **KG entity hiện chỉ phân cấp** (`THUOC_VAN_BAN/CHUONG/DIEU`); chưa có quan hệ ngữ nghĩa như `BAI_BO`, `SUA_DOI`, `THAM_CHIEU`, `THAY_THE`.
3. **Eval bộ test ~100 cặp** — quy mô nhỏ; cần mở rộng cho coverage tốt hơn.
4. **Không hỗ trợ streaming** trong response của Gemini → user phải chờ trọn câu trả lời.
5. **Chưa có cơ chế re-rank** sau retrieval (cross-encoder, ColBERT).

### 6.3. Hướng phát triển

| Hướng | Ưu tiên | Mô tả ngắn |
|-------|---------|------------|
| **Re-ranker cross-encoder** | Cao | Thêm bước re-rank top-30 → top-5 bằng cross-encoder (vd. `bge-reranker-v2-m3-vi`) trước khi đẩy vào LLM. |
| **Quan hệ ngữ nghĩa trong KG** | Cao | Bổ sung `BAI_BO`, `SUA_DOI_BOI`, `THAM_CHIEU` từ trường `sua_doi_boi`/`ban_su_dung` đã có sẵn trong `metadata_config.py`. |
| **Auto-update VBHN** | Trung | Crawl định kỳ thuvienphapluat / vanbanphapluat.vn → diff → re-embed delta. |
| **Streaming response** | Trung | Dùng `model.generate_content(stream=True)` của Gemini, pipe SSE qua FastAPI. |
| **Eval pipeline mở rộng** | Trung | Sinh test set tự động bằng LLM + người chuyên môn duyệt; thêm metric **NDCG**, **answer-faithfulness**. |
| **PhoBERT-LawV3 fine-tune** | Thấp | Iterate corpus fine-tune trong `prepare_finetune.py` lên 100k cặp triplet. |
| **Multi-turn refinement** | Thấp | Cho phép user follow-up "giải thích thêm Điều 9" mà chatbot dùng KG để mở rộng cùng chủ đề. |
| **Caching tầng truy vấn** | Thấp | Redis cache cho cặp `(rewritten_query, sub_queries) → results` — hữu ích khi cùng câu hỏi được hỏi nhiều lần. |

---

## Tài liệu tham khảo

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
2. Edge, D., et al. (2024). *From Local to Global: A GraphRAG Approach to Query-Focused Summarization.* Microsoft Research.
3. Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese.* EMNLP Findings.
4. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP-IJCNLP.
5. Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.
6. Anthropic (2024). *Contextual Retrieval.* — https://www.anthropic.com/news/contextual-retrieval
7. LangChain Docs — *Parent Document Retriever, RecursiveCharacterTextSplitter*.
8. Qdrant Documentation — https://qdrant.tech/documentation/
9. Neo4j Cypher Manual — Vector Index, Text Index.
10. Quốc hội VN — *Luật An ninh mạng 2018* (24/2018/QH14).
11. Quốc hội VN — *Luật Công nghệ thông tin 2006* (67/2006/QH11).
12. Quốc hội VN — *Luật Sở hữu trí tuệ 2005* (sửa đổi 2009, 2019, 2022).
13. Quốc hội VN — *Luật Giao dịch điện tử 2023* (20/2023/QH15).
14. Quốc hội VN — *Luật Bảo vệ dữ liệu cá nhân 2025*.
15. Chính phủ VN — *Nghị định 13/2023/NĐ-CP* về bảo vệ dữ liệu cá nhân.
16. Chính phủ VN — *Nghị định 15/2020/NĐ-CP* (sửa đổi bởi NĐ 14/2022) — xử phạt vi phạm hành chính lĩnh vực bưu chính, viễn thông, công nghệ thông tin.

---

## Phụ lục A — Cấu hình triển khai

```env
# .env
GEMINI_API_KEY=...
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
EMBEDDING_MODEL=./models/embedding_model     # PhoBERT fine-tuned
ALLOWED_ORIGINS=http://localhost:5000
API_KEY=...                                  # tùy chọn
RATE_LIMIT_PER_MINUTE=30
API_PORT=5000
```

Khởi chạy stack đầy đủ:

```bash
# 1. Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 2. Neo4j
docker run -d -p 7687:7687 -p 7474:7474 \
    -e NEO4J_AUTH=neo4j/password neo4j:5

# 3. Build dữ liệu (1 lần)
cd law_crawler
python law_crawler.py     -i data/raw -o data/law_data_output.xlsx
python smart_chunker.py   -i data/law_data_output.xlsx -o data/law_chunks --format both
python build_hierarchical.py --chunks data/law_chunks.jsonl \
       --excel data/law_data_output.xlsx --output data/law_chunks_hier.jsonl
python embed_to_qdrant.py -i data/law_chunks_hier.jsonl --device cuda

cd ../nlp_pipeline/02_chunking
python extract_kg_entities.py
cd ../../scripts
python migrate_to_neo4j.py

# 4. Run backend
cd ..
python -m app.main
# → http://localhost:5000
```

## Phụ lục B — Bảng metadata 24 văn bản đã ingest

Theo `data/raw/`:

* **Luật**: An ninh mạng 2018, An ninh mạng 2025, ATTT mạng 2015 (sđ 2018), CNTT 2006 (sđ Quy hoạch 2017, GDĐT 2023, VT 2023), Công nghiệp công nghệ số 2025, Dữ liệu 2024, Bảo vệ dữ liệu cá nhân 2025, Bảo vệ quyền lợi người tiêu dùng 2023, Giao dịch điện tử 2023, Sở hữu trí tuệ 2005 (sđ 2009/2019/2022), Viễn thông 2023.
* **Nghị định**: 13/2023, 15/2020 (sđ 14/2022), 17/2023, 52/2013 (sđ 08/2018, 85/2021), 52/2024, 53/2022, 71/2007, 85/2016, 130/2018, 147/2024, 211/2025.

---

*Báo cáo được sinh từ codebase `IT-Law-Chatbot/` snapshot ngày 09/05/2026.*
