"""
Prompts for Mem0-inspired Memory System in IT Law Chatbot.
"""

MEMORY_EXTRACTION_PROMPT = """Bạn là trợ lý AI chuyên trích xuất thông tin người dùng và ngữ cảnh pháp lý để lưu vào bộ nhớ (Memory Layer).

Dưới đây là một lượt hội thoại giữa Người dùng (User) và Trợ lý (Assistant):
User: {query}
Assistant: {answer}

Ngữ cảnh trí nhớ hiện tại đã biết về người dùng:
{existing_memories}

NHIỆM VỤ:
Trích xuất các thông tin/facts MỚI, có giá trị dài hạn về:
1. Hồ sơ người dùng (User Profile): Ngành nghề, vai trò (cá nhân, doanh nghiệp CNTT, sàn TMĐT, lập trình viên, nhà cung cấp dịch vụ viễn thông/Internet...), quy mô hoặc đặc điểm hoạt động.
2. Vấn đề/Tình huống pháp lý cụ thể: Các văn bản luật, điều khoản, hành vi vi phạm, hoặc sự việc cụ thể mà người dùng đang tìm hiểu hoặc gặp phải.
3. Ràng buộc áp dụng: Ví dụ: hỏi về mức phạt cho cá nhân (chia 2), doanh nghiệp cung cấp dịch vụ xuyên biên giới, điều kiện cấp phép...

QUY TẮC BẮT BUỘC:
- BỎ QUA các câu chào hỏi xã giao, cảm ơn, câu hỏi chung chung lý thuyết suông không chứa thông tin ngữ cảnh riêng.
- Nếu không có thông tin mới đáng lưu, trả về danh sách facts rỗng [].
- Mỗi fact viết thành một câu khẳng định ngắn gọn, súc tích (1 câu, tối đa 30 từ).
- Trả về JSON đúng định dạng sau, không kèm bất kỳ giải thích nào khác:

```json
{{
  "facts": [
    "Người dùng là doanh nghiệp phát triển ứng dụng di động có thu thập dữ liệu người dùng",
    "Người dùng đang tìm hiểu mức phạt hành vi truy cập trái phép hệ thống theo Nghị định 15/2020/NĐ-CP"
  ],
  "session_updates": {{
    "user_role": "doanh nghiệp",
    "focused_laws": ["Nghị định 15/2020/NĐ-CP"],
    "focused_articles": ["Điều 80"],
    "topic": "xử phạt vi phạm an toàn thông tin"
  }}
}}
```
"""
