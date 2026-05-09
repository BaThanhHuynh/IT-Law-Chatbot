# IT Law Chatbot - Trợ lý pháp lý thông minh

## 1. Mô tả dự án
**IT Law Chatbot** là một trợ lý AI chuyên tư vấn pháp lý, luật về lĩnh vực Công nghệ thông tin tại Việt Nam. Dự án ứng dụng kiến trúc tiên tiến **GraphRAG** (kết hợp Đồ thị tri thức - Knowledge Graph và Tìm kiếm Vector), kết hợp với mô hình LLM mạnh mẽ để cung cấp các câu trả lời chính xác, bám sát văn bản pháp luật và luôn đi kèm trích dẫn minh bạch.

Hệ thống được đóng gói hoàn toàn bằng **Docker** để đảm bảo tính ổn định, dễ dàng triển khai và tối ưu hóa hiệu suất với độ trễ (latency) thấp (~7 giây/phản hồi).

## 2. Tính năng nổi bật
- **Kiến trúc GraphRAG Hybrid**: Kết hợp sức mạnh tìm kiếm ngữ nghĩa của **Qdrant Vector DB** và suy luận mối quan hệ phức tạp từ **Neo4j Graph DB**.
- **Contextual Query Rewriting**: Tự động viết lại câu hỏi dựa trên lịch sử hội thoại, giúp chatbot hiểu được đại từ xưng hô (VD: "nó", "điều đó") trong các cuộc hội thoại nhiều lượt.
- **Tối ưu hóa độ trễ (Low Latency)**: Xử lý song song đa luồng (Multi-threading) các tác vụ LLM (Phân loại ý định, Trích xuất thực thể, Sinh đa câu hỏi), giúp giảm hơn 50% thời gian chờ.
- **Lưu trữ lịch sử dài hạn**: Mọi phiên chat được vector hóa và lưu trữ trực tiếp vào Qdrant, giúp quản lý context một cách bền vững.
- **Mô hình nhúng Tiếng Việt**: Sử dụng mô hình **huyydangg/DEk21_hcmute_embedding** đã được fine-tune Môh trên bộ dữ liệu nội bộ bao gồm khoảng 100.000 ví dụ về các câu hỏi pháp lý và bối cảnh liên quan của chúng.

## 3. Ảnh Demo
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)
## 4. Hướng dẫn cài đặt (Dockerized)

**Yêu cầu hệ thống:**
- Docker & Docker Compose
- Neo4j Database (Cài đặt trên máy host hoặc server riêng)
- Python 3.10 (Nếu muốn chạy script nạp dữ liệu từ máy host)

**Các bước triển khai:**

1. **Clone mã nguồn:**
   ```bash
   git clone https://github.com/BaThanhHuynh/IT-Law-Chatbot.git
   cd IT-Law-Chatbot
   ```

2. **Cấu hình môi trường (`.env`):**
   Tạo file `.env` ở thư mục gốc và cung cấp các thông tin:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   # Dùng host.docker.internal để Docker giao tiếp với Neo4j trên máy thật
   NEO4J_URI=bolt://host.docker.internal:7687 
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   API_PORT=5000
   ```

3. **Khởi chạy hệ thống bằng Docker:**
   ```bash
   docker-compose up --build -d
   ```
   *Lệnh này sẽ khởi động 2 container: ứng dụng Chatbot và CSDL Vector Qdrant.*

4. **Nạp dữ liệu (Ingestion):**
   Nếu Qdrant hoặc Neo4j chưa có dữ liệu, hãy chạy các script sau từ máy host:
   ```bash
   # Nạp dữ liệu Vector vào Qdrant
   python law_crawler/embed_to_qdrant.py --input data/law_chunks_hier.jsonl

   # Nạp dữ liệu Đồ thị vào Neo4j
   python scripts/migrate_to_neo4j.py
   ```

## 5. Cách sử dụng

- **Truy cập ứng dụng:** Mở trình duyệt web và đi tới địa chỉ: `http://localhost:5000`
- **Tra cứu:** Gõ các câu hỏi pháp lý vào khung chat (ví dụ: *"Công ty tôi muốn mở một website thương mại điện tử thì cần xin phép cơ quan nào?"*).
- **Khám phá:** Xem phần **Knowledge Graph** hiển thị bên cạnh để hiểu rõ các bộ luật, chương, điều khoản liên kết với nhau như thế nào.

## 6. Công nghệ sử dụng

| Thành phần | Công nghệ / Thư viện |
|------------|-----------------------|
| **Backend API** | FastAPI, Uvicorn, Docker |
| **Mô hình Ngôn ngữ (LLM)** | Google Gemini-3.1-flash-lite-preview |
| **Mô hình Nhúng (Embedding)** | huyydangg/DEk21_hcmute_embedding (Fine-tuned cho Luật VN) |
| **Cơ sở dữ liệu Đồ thị (Graph DB)** | Neo4j, LangChain |
| **Cơ sở dữ liệu Vector (Vector DB)** | Qdrant |
| **Giao diện (Frontend)** | HTML/CSS/JS (Vanilla) |

## 7. Liên hệ
- **Tác giả**: BaThanhHuynh (Huỳnh Bá Thành)
- **GitHub**: [@BaThanhHuynh](https://github.com/BaThanhHuynh)
- **Link dự án**: [IT-Law-Chatbot](https://github.com/BaThanhHuynh/IT-Law-Chatbot)
