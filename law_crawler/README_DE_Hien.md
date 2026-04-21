# 📋 Hướng dẫn quy trình DE – Lục Sỹ Minh Hiền
**Project:** Chatbot AI tra cứu Luật CNTT Việt Nam  
**Vai trò:** Data Engineer – Thu thập, xử lý, cấu trúc dữ liệu

---

## Cấu trúc thư mục thực tế

```
law_crawler/
│
├── data/
│   ├── raw/                        ← Đặt 22 file DOCX vào đây
│   │   ├── luat_an_ninh_mang_2018.docx
│   │   ├── luat_du_lieu_2024.docx
│   │   └── ... (22 file)
│   │
│   ├── law_data_output.xlsx        ← Output bước 1 (tên file thực tế)
│   ├── law_chunks.xlsx             ← Output bước 3 (xem review)
│   └── law_chunks.jsonl            ← Output bước 3 (→ gửi Phú)
│
├── doi.py                          ← (file phụ nếu có)
├── export_for_rag.py
├── law_crawler.py                  ← Script bước 1
├── metadata_config.py              ← Config metadata 22 văn bản
├── README_DE_Hien.md               ← File này
├── smart_chunker.py                ← Script bước 3
└── validate_data.py                ← Script bước 2
```

---

## Cài đặt môi trường (chỉ làm 1 lần)

```powershell
# Mở terminal tại thư mục law_crawler
pip install python-docx openpyxl

# Tùy chọn - nếu muốn dùng LangChain cho Tầng 4 splitting
# (không bắt buộc, script vẫn chạy tốt không có LangChain)
pip install langchain-text-splitters
```

> ⚠️ **Lưu ý về LangChain:**  
> Chỉ cài `langchain-text-splitters` (có chữ **s** ở cuối).  
> Không dùng `langchain.text_splitter` (không có s) — đó là module **cũ đã deprecated** từ LangChain v0.2+, sẽ báo lỗi.

---

## Quy trình 4 bước

---

### Bước 1 — Crawl DOCX → Excel

**Script:** `law_crawler.py`  
**Input:** Folder `data/raw/` chứa 22 file DOCX  
**Output:** `data/law_data_output.xlsx`

```powershell
python law_crawler.py --input ./data/raw --output ./data/law_data_output.xlsx
```

**Kết quả:** File Excel 3 sheet, 24 cột:
- Sheet `Dữ liệu luật` — toàn bộ điều khoản
- Sheet `Danh mục văn bản` — tổng hợp 22 văn bản
- Sheet `Thống kê` — số liệu tổng quan

**Lưu ý đặt tên file DOCX:**  
Script tự nhận metadata qua keyword trong tên file. Xem bảng keyword ở cuối README.

---

### Bước 2 — Kiểm tra chất lượng

**Script:** `validate_data.py`  
**Input:** `data/law_data_output.xlsx`  
**Output:** Báo cáo in ra terminal (không tạo file)

```powershell
python validate_data.py --input ./data/law_data_output.xlsx
```

**Đọc kết quả:**

| Cảnh báo | Nguyên nhân | Xử lý |
|----------|-------------|--------|
| `Chunk quá ngắn (< 30 ký tự)` | Điều bị bãi bỏ trong VBHN | **Bình thường** – smart_chunker sẽ đánh flag `is_repealed=True` |
| `Chunk dài (> 512 ký tự, nên split)` | Điều có nhiều khoản/điểm | **Bình thường** – smart_chunker sẽ tự split |
| `Thiếu metadata tên văn bản` | Tên file không khớp keyword | Sửa tên file theo bảng keyword |
| `Nội dung trống` | Parse sai cấu trúc DOCX | Kiểm tra file DOCX gốc |

---

### Bước 3 — Smart Chunking

**Script:** `smart_chunker.py`  
**Input:** `data/law_data_output.xlsx`  
**Output:** `data/law_chunks.xlsx` + `data/law_chunks.jsonl`

```powershell
python smart_chunker.py --input ./data/law_data_output.xlsx --output ./data/law_chunks --format both
```

**Chiến lược chunking 4 tầng:**

```
Điều gốc
  │
  ├─ ≤ 450 ký tự  →  [Tier 1] Dùng thẳng (màu xanh lá)
  │
  └─ > 450 ký tự  →  Split theo Khoản "1. 2. 3."
                        │
                        ├─ ≤ 450 ký tự  →  [Tier 2] Dùng (màu xanh dương)
                        │
                        └─ > 450 ký tự  →  Split theo Điểm "a) b) c)"
                                            │
                                            ├─ ≤ 450 ký tự  →  [Tier 3] (màu vàng)
                                            │
                                            └─ > 450 ký tự  →  [Tier 4] Simple/LangChain (màu tím)

Điều bị bãi bỏ (< 30 ký tự)  →  [Tier 0] Flag is_repealed=True (màu đỏ nhạt)
```

**Kết quả kỳ vọng (với ~1567 điều từ 22 văn bản):**

```
Tổng chunks:              ~12,000 – 15,000
Chunk > 512 ký tự còn lại:       0  ✅
Avg độ dài chunk:           ~150 ký tự
```

**Tùy chỉnh chunk size** (nếu cần thay đổi):
```powershell
# Mặc định: chunk_size=450, overlap=50
python smart_chunker.py --input ./data/law_data_output.xlsx --output ./data/law_chunks --chunk_size 400 --overlap 40
```

---

### Bước 4 — Giao file cho nhóm

**Giao cho Phú (AI Engineer):**
```
data/law_chunks.jsonl   ← import vào Qdrant/Chroma
```

**Giao cho Thành (AI Application):**
```
data/law_chunks.xlsx    ← review dữ liệu, kiểm tra context_text
```

---

## Giải thích màu sắc trong law_chunks.xlsx

| Màu | Tier | Ý nghĩa |
|-----|------|---------|
| 🔴 Đỏ nhạt | 0 | Điều bị bãi bỏ/hết hiệu lực |
| 🟢 Xanh lá | 1 | Điều ngắn, giữ nguyên |
| 🔵 Xanh dương | 2 | Split theo Khoản (1. 2. 3.) |
| 🟡 Vàng | 3 | Split theo Điểm (a) b) c)) |
| 🟣 Tím | 4 | Split bằng Simple/LangChain |

---

## Schema file JSONL (giao cho Phú)

Mỗi dòng trong `.jsonl` là 1 JSON object:

```json
{
  "id": "abc123def456",        ← chunk_id duy nhất (dùng làm vector ID trong Qdrant)
  "text": "Luật An ninh mạng 2018 (24/2018/QH14) | Chương I: Những quy định chung | Điều 2. Giải thích từ ngữ | 1. An ninh mạng là...",
                               ← context_text → đây là INPUT cho PhoBERT embedding
  "payload": {
    "source_file":       "luat_an_ninh_mang_2018.docx",
    "ten_van_ban":       "Luật An ninh mạng 2018",
    "so_hieu":           "24/2018/QH14",
    "loai_van_ban":      "Luật",
    "trang_thai":        "con_hieu_luc",   ← dùng để filter khi query
    "ngay_hieu_luc":     "01/01/2019",
    "ngay_het_hieu_luc": "30/06/2026",
    "nhom":              "Nhóm 1 - Đạo luật nền tảng",
    "chuong_so":         "1",
    "chuong_ten":        "NHỮNG QUY ĐỊNH CHUNG",
    "muc_so":            "",
    "muc_ten":           "",
    "dieu_so":           "2",
    "dieu_ten":          "Giải thích từ ngữ",
    "noi_dung_chunk":    "1. An ninh mạng là sự bảo đảm...",   ← nội dung chunk thực
    "chunk_tier":        2,
    "chunk_sub_index":   0,    ← chunk thứ mấy trong điều này
    "chunk_total_sub":   5,    ← điều này có tổng 5 chunks
    "is_repealed":       false
  }
}
```

> **Lưu ý cho Phú:** Khi query retrieval, nên filter `trang_thai = "con_hieu_luc"` để loại bỏ văn bản hết hiệu lực. Điều `is_repealed = true` nên được giữ lại trong DB để chatbot có thể trả lời "điều này đã bị bãi bỏ".

---

## Bảng keyword đặt tên file DOCX

| Keyword phải có trong tên file | Văn bản tương ứng |
|-------------------------------|-------------------|
| `an_ninh_mang_2018` | Luật An ninh mạng 2018 (24/2018/QH14) |
| `an_ninh_mang_2025` | Luật An ninh mạng 2025 (116/2025/QH15) |
| `attt_2015` hoặc `an_toan_thong_tin_2015` | Luật ATTT mạng 2015 |
| `cntt_2006` hoặc `cong_nghe_thong_tin_2006` | Luật CNTT 2006 (VBHN 2023) |
| `giao_dich_dien_tu_2023` | Luật Giao dịch điện tử 2023 |
| `vien_thong_2023` | Luật Viễn thông 2023 |
| `shtt_2005` hoặc `so_huu_tri_tue` | Luật SHTT 2005 |
| `du_lieu_2024` | Luật Dữ liệu 2024 |
| `bvdlcn_2025` hoặc `bao_ve_du_lieu_ca_nhan` | Luật BVDLCN 2025 |
| `cong_nghiep_cong_nghe_so` hoặc `cnts_2025` | Luật CNTS 2025 |
| `bvqlntd_2023` hoặc `nguoi_tieu_dung_2023` | Luật BVQLNTD 2023 |
| `71_2007` | NĐ 71/2007/NĐ-CP |
| `85_2016` | NĐ 85/2016/NĐ-CP |
| `130_2018` | NĐ 130/2018/NĐ-CP |
| `53_2022` | NĐ 53/2022/NĐ-CP |
| `13_2023` | NĐ 13/2023/NĐ-CP |
| `15_2020` | NĐ 15/2020/NĐ-CP |
| `147_2024` | NĐ 147/2024/NĐ-CP |
| `52_2013` hoặc `thuong_mai_dien_tu` | NĐ TMĐT (VBHN 14/VBHN-BCT) |
| `17_2023` | NĐ 17/2023/NĐ-CP |
| `52_2024` hoặc `thanh_toan` | NĐ 52/2024/NĐ-CP |
| `211_2025` hoặc `mat_ma` | NĐ 211/2025/NĐ-CP |

**Ví dụ tên file hợp lệ:**
```
✅ luat_an_ninh_mang_2018.docx
✅ Luat_CNTT_2006_VBHN_2023.docx          ← viết hoa cũng được (script lowercase)
✅ nd_13_2023_bvdlcn.docx
✅ thuong_mai_dien_tu_52_2013_vbhn.docx
```

---

## Lỗi thường gặp & cách xử lý

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `ImportError: langchain.text_splitter` | Dùng module LangChain **cũ** | Xoá dòng import cũ, chỉ dùng `langchain_text_splitters` (có **s**) |
| `Không tìm thấy sheet 'Dữ liệu luật'` | Sai tên sheet Excel | Đảm bảo chạy `law_crawler.py` trước |
| `Không tìm được metadata cho: xxx.docx` | Tên file không có keyword | Đổi tên file theo bảng keyword |
| `0 điều` từ 1 file | Cấu trúc DOCX không có "Điều X." | Kiểm tra file: phải có dòng bắt đầu bằng "Điều" |
| Excel cell 32767 ký tự | Điều quá dài, Excel truncate | Bình thường – smart_chunker đọc từ Excel và sẽ flag `is_truncated=True` |

---

## Lệnh hay dùng (Quick Reference)

```powershell
# === Chạy đầy đủ pipeline ===
python law_crawler.py   --input ./data/raw               --output ./data/law_data_output.xlsx
python validate_data.py --input ./data/law_data_output.xlsx
python smart_chunker.py --input ./data/law_data_output.xlsx --output ./data/law_chunks --format both

# === Chỉ muốn xuất JSONL (cho Phú) ===
python smart_chunker.py --input ./data/law_data_output.xlsx --output ./data/law_chunks --format jsonl

# === Chỉ muốn xem Excel (review) ===
python smart_chunker.py --input ./data/law_data_output.xlsx --output ./data/law_chunks --format excel

# === Thử với 1 file duy nhất ===
python law_crawler.py --input ./data/raw/luat_an_ninh_mang_2018.docx --output ./data/test_1file.xlsx
```
