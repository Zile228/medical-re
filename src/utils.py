import os
import joblib
import json
from pathlib import Path
import glob

# --- CẤU HÌNH ĐƯỜNG DẪN CHUNG ---
# Lấy đường dẫn thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TOKENIZER_DIR = os.path.join(MODEL_DIR, 'tokenizer_config') # Nơi lưu Tokenizer đồng bộ

# Đảm bảo thư mục tồn tại
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# --- CÁC HÀM XỬ LÝ VĂN BẢN DÙNG CHUNG ---
def load_unlabeled_data(unlabeled_dir):
    """
    Đọc tất cả file .json trong thư mục data/raw/unlabeled/
    Format: [{"text": "Câu 1..."}, ...]
    """
    all_texts = []
    # Quét tất cả file json
    file_paths = glob.glob(os.path.join(unlabeled_dir, "*.json"))
    
    print(f"Tìm thấy {len(file_paths)} file dữ liệu chưa gán nhãn.")
    
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # data là list các dict [{"text": "..."}]
                for item in data:
                    if "text" in item:
                        txt = item["text"].strip()
                        if txt: all_texts.append(txt)
        except Exception as e:
            print(f"Lỗi đọc file {path}: {e}")
            
    # Lọc trùng lặp
    unique_texts = list(set(all_texts))
    print(f"Đã tải {len(unique_texts)} câu văn bản duy nhất.")
    return unique_texts

def simple_tokenizer(text):
    """
    Tokenizer đơn giản tách theo khoảng trắng.
    Dùng cho BoW và TF-IDF.
    """
    return text.split()

def add_markers(text, subj_span, obj_span, subj_type, obj_type):
    """
    Hàm chèn marker [S]...[/S] và [O]...[/O] vào câu.
    Dùng chung cho cả Preprocessing và Evaluation.
    """
    s_start, s_end = subj_span['start'], subj_span['end']
    o_start, o_end = obj_span['start'], obj_span['end']
    
    insertions = [
        (s_end, f" [/{subj_type}]"),
        (s_start, f"[{subj_type}] "),
        (o_end, f" [/{obj_type}]"),
        (o_start, f"[{obj_type}] ")
    ]
    
    # Sắp xếp giảm dần để không bị lệch index khi chèn
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    modified_text = text
    for pos, string in insertions:
        modified_text = modified_text[:pos] + string + modified_text[pos:]
        
    # Chuyển về chữ thường toàn bộ để đồng bộ
    return modified_text.lower()

def load_re_model_resources(vec_name, model_name):
    """
    Hàm tiện ích để load nhanh Model, Vectorizer và LabelEncoder.
    Dùng trong file evaluate.py.
    """
    try:
        # 1. Load Vectorizer
        if vec_name == 'bow':
            vec = joblib.load(os.path.join(MODEL_DIR, 'bow_vectorizer.joblib'))
        elif vec_name == 'tfidf':
            vec = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
        elif vec_name == 'w2v':
            from gensim.models import Word2Vec
            vec = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
        else:
            vec = None # BERT load riêng qua Tokenizer saved
            
        # 2. Load Scaler (nếu có)
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{vec_name}.pkl')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        # 3. Load Model Phân loại
        safe_name = model_name.replace(" ", "").replace("(", "").replace(")", "")
        clf_path = os.path.join(MODEL_DIR, f"{safe_name}_{vec_name}.pkl")
        clf = joblib.load(clf_path)
        
        # 4. Load Label Encoder
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        
        return vec, scaler, clf, le
    except Exception as e:
        print(f"Lỗi khi load tài nguyên ({vec_name}, {model_name}): {e}")
        return None, None, None, None