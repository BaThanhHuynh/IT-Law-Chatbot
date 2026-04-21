import json
import re
from pyvi import ViTokenizer

# Cấu hình đường dẫn
INPUT_FILE = "data/finetune_corpus.jsonl"
OUTPUT_FILE = "phobert_mlm_law_40k.txt"
MAX_LINES = 40000  # Lấy khoảng 40k mẫu theo yêu cầu

# Biểu thức chính quy (Regex) để xóa các đoạn log nhiễu trong dữ liệu
# Ví dụ xóa: "...[Nội dung còn lại của Điều 17 (Luật An ninh mạng 2018.docx) quá dài (4,325 ký tự), đã rút gọn. Xem đầy đủ trong văn bản gốc.]"
truncation_pattern = re.compile(r'\.\.\.\[Nội dung còn lại.*?\]')

def preprocess_data():
    processed_count = 0
    
    print("⏳ Đang tiến hành dọn dẹp dữ liệu và tách từ (Word Segmentation)...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if processed_count >= MAX_LINES:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                # 1. Đọc định dạng JSONL
                data = json.loads(line)
                text = data.get("text", "")
                
                if text:
                    # 2. Xóa bỏ các đoạn log rác bị rút gọn trong quá trình cào dữ liệu (nếu có)
                    clean_text = re.sub(truncation_pattern, '', text).strip()
                    
                    # 3. Tách từ tiếng Việt bằng PyVi (Bắt buộc đối với PhoBERT)
                    # Ví dụ: "An ninh mạng" sẽ thành "An_ninh mạng"
                    segmented_text = ViTokenizer.tokenize(clean_text)
                    
                    # 4. Ghi ra file text (mỗi câu/đoạn là một dòng)
                    if segmented_text:
                        f_out.write(segmented_text + "\n")
                        processed_count += 1
                        
                        # In tiến độ cho dễ theo dõi
                        if processed_count % 5000 == 0:
                            print(f"  -> Đã xử lý {processed_count} mẫu...")
                            
            except json.JSONDecodeError:
                continue

    print(f"✅ Hoàn tất! Đã tiền xử lý thành công {processed_count} mẫu và lưu vào '{OUTPUT_FILE}'.")

if __name__ == '__main__':
    preprocess_data()