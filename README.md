# IT Law Chatbot - Tư vấn Luật Công nghệ thông tin (GraphRAG Edition)

IT Law Chatbot là một trợ lý ảo hỗ trợ tư vấn pháp luật chuyên sâu về lĩnh vực Công nghệ thông tin (Việt Nam). Dự án sử dụng kiến trúc **GraphRAG** (Retrieval-Augmented Generation kết hợp Knowledge Graph) để tăng cường độ chính xác cho câu trả lời.

Hệ thống kết hợp sức mạnh của **Vector Search** (truy xuất đoạn văn bản liên quan) và **Knowledge Graph** (truy xuất mối quan hệ giữa các điều luật) để cung cấp câu trả lời có tính hệ thống và đầy đủ dẫn chứng.

---

## Kiến trúc GraphRAG

Hệ thống GraphRAG trong dự án này bao gồm:

### 1. Dữ liệu Tri thức (Knowledge Graph)
Được lưu trữ trong **Neo4j**, bao gồm các thực thể pháp lý và mối liên kết giữa chúng:
- **Thực thể (Nodes):**
    - `VAN_BAN`: Các luật, nghị định, thông tư (v dụ: Luật CNTT 2006).
    - `CHUONG`: Các chương mục lớn trong văn bản pháp luật.
    - `DIEU_LUAT`: Các điều luật cụ thể - đơn vị tra cứu chính.
    - `KHAI_NIEM`: Các định nghĩa, khái niệm pháp lý quan trọng.
    - `HANH_VI`: Các hành vi bị nghiêm cấm được quy định cụ thể.
    - `CHU_THE`: Đối tượng áp dụng (Cá nhân, Tổ chức, Doanh nghiệp...).
- **Quan hệ (Edges):**
    - `THUOC`: Phân cấp Điều -> Chương -> Văn bản.
    - `THAM_CHIEU`: Khi một điều luật dẫn chiếu đến một điều luật hoặc văn bản khác.
    - `NGHIEM_CAM`: Liên kết giữa điều luật và các hành vi không được phép.
    - `DINH_NGHIA`: Liên kết giữa điều luật và khái niệm nó giải thích.
    - `AP_DUNG`: Đối tượng chịu sự điều chỉnh của điều luật.

### 2. Truy xuất Vector (Vector RAG)
Sử dụng **MySQL** lưu trữ nội dung chi tiết và Embeddings (`all-MiniLM-L6-v2`). Giúp tìm kiếm các đoạn văn bản có nội dung tương đồng về mặt ngữ nghĩa với câu hỏi người dùng.

### 3. Mô hình ngôn ngữ (LLM)
Sử dụng **Google Gemini (gemini-2.5-flash)** để tổng hợp thông tin từ cả Vector Search và Graph Search nhằm tạo ra câu trả lời cuối cùng chính xác, có cấu trúc.

---

## Yêu cầu hệ thống

- **Python 3.10+**
- **Neo4j Desktop** (Bản cài đặt cho Windows)
- **XAMPP** (MySQL 3306)

---

## Hướng dẫn cài đặt và chạy chi tiết

### Bước 1: Khởi động cơ sở dữ liệu

1. **MySQL:** Mở XAMPP và Start module **MySQL**.
2. **Neo4j Desktop:** 
   - Tải và cài đặt [Neo4j Desktop](https://neo4j.com/download/).
   - Tạo một Project mới, chọn **Add** -> **Local DBMS**.
   - Đặt tên DBMS (vD: `it-law-db`) và đặt mật khẩu (khuyến nghị dùng: `password` để khớp với `.env`).
   - Nhấn **Start** để khởi chạy database. 
   - *Đảm bảo trạng thái là "Active" trước khi chạy các script Python.*

### Bước 2: Thiết lập môi trường Python

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

### Bước 3: Cấu hình `.env`

Tạo file `.env` từ nội dung dưới đây:
```env
GEMINI_API_KEY=điền_api_key_cua_ban

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=it_law_chatbot

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### Bước 4: Nạp dữ liệu (Data Pipeline)

Quy trình nạp dữ liệu bao gồm 2 công đoạn:

1. **Nạp dữ liệu vào MySQL & Sinh Vector:**
   ```bash
   python database/seed_data.py
   ```
   *(Quá trình này tạo bảng, lưu trữ văn bản luật và tạo embeddings)*

2. **Đồng bộ sang Neo4j (Xây dựng Graph):**
   ```bash
   python scripts/migrate_to_neo4j.py
   ```
   *(Script này sẽ đọc các quan hệ từ MySQL và chuyển đổi thành Nodes/Edges trong Neo4j)*

### Bước 5: Chạy máy chủ

```bash
python app.py
```
Truy cập: **[http://localhost:5000](http://localhost:5000)**

---

## Cách sử dụng

1. **Đặt câu hỏi:** Nhập các vấn đề pháp lý liên quan đến CNTT (Ví dụ: "Hành vi phá hoại hệ thống thông tin bị xử lý thế nào?").
2. **Kết quả Hybrid:** 
   - Hệ thống sẽ dùng Vector Search để tìm các điều luật liên quan trực tiếp.
   - Hệ thống dùng Knowledge Graph để tìm các khái niệm liên quan, các điều luật tham chiếu hoặc văn bản gốc.
3. **Phân tích Đồ thị:** Bạn có thể xem biểu đồ trực quan các mối liên hệ giữa các điều luật ngay trên giao diện chat.
4. **Trích dẫn:** Mọi câu trả lời đều có trích dẫn nguồn luật cụ thể để đảm bảo tính pháp lý.
