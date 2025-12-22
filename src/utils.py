import os
import joblib
import json
import re  
from pathlib import Path
import glob
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN CHUNG ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TOKENIZER_DIR = os.path.join(MODEL_DIR, 'tokenizer_config')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# --- CÁC HÀM XỬ LÝ VĂN BẢN CƠ BẢN ---
def load_unlabeled_data(unlabeled_dir):
    """Đọc file json chưa gán nhãn"""
    all_texts = []
    file_paths = glob.glob(os.path.join(unlabeled_dir, "*.json"))
    print(f"Tìm thấy {len(file_paths)} file dữ liệu chưa gán nhãn.")
    
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if "text" in item:
                        txt = item["text"].strip()
                        if txt: all_texts.append(txt)
        except Exception as e:
            print(f"Lỗi đọc file {path}: {e}")
            
    unique_texts = list(set(all_texts))
    print(f"Đã tải {len(unique_texts)} câu văn bản duy nhất.")
    return unique_texts

def simple_tokenizer(text):
    return text.split()

def add_markers(text, subj_span, obj_span, subj_type, obj_type):
    """Chèn marker [S]...[/S] và [O]...[/O]"""
    s_start, s_end = subj_span['start'], subj_span['end']
    o_start, o_end = obj_span['start'], obj_span['end']
    
    insertions = [
        (s_end, f" [/{subj_type}]"), (s_start, f"[{subj_type}] "),
        (o_end, f" [/{obj_type}]"), (o_start, f"[{obj_type}] ")
    ]
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    modified_text = text
    for pos, string in insertions:
        modified_text = modified_text[:pos] + string + modified_text[pos:]
    return modified_text.lower()

def load_re_model_resources(vec_name, model_name):
    """Load Model, Vectorizer, Scaler, LabelEncoder"""
    try:
        if vec_name == 'bow':
            vec = joblib.load(os.path.join(MODEL_DIR, 'bow_vectorizer.joblib'))
        elif vec_name == 'tfidf':
            vec = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
        elif vec_name == 'w2v':
            from gensim.models import Word2Vec
            vec = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
        else:
            vec = None 
            
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{vec_name}.pkl')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        safe_name = model_name.replace(" ", "").replace("(", "").replace(")", "")
        clf_path = os.path.join(MODEL_DIR, f"{safe_name}_{vec_name}.pkl")
        clf = joblib.load(clf_path)
        
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        return vec, scaler, clf, le
    except Exception as e:
        print(f"Lỗi khi load tài nguyên ({vec_name}, {model_name}): {e}")
        return None, None, None, None

# BỘ LUẬT RULE-BASED 

# 1. Từ khóa nhận diện quan hệ
KEYWORDS = {
    'Gây_ra': [
        r'gây ra', r'dẫn đến', r'biến chứng', r'hậu quả', r'do', r'bởi', 
        r'làm tăng nguy cơ', r'phát sinh', r'nguy cơ mắc', r'tiến triển thành'
    ],
    'Điều_trị_bằng': [
        r'điều trị', r'chữa trị', r'uống', r'sử dụng', r'dùng thuốc', 
        r'chỉ định', r'phẫu thuật', r'xạ trị', r'kiểm soát bằng', r'tiêm'
    ],
    'Chẩn_đoán_bằng': [
        r'chẩn đoán', r'xét nghiệm', r'chụp', r'siêu âm', r'nội soi', 
        r'phát hiện qua', r'kiểm tra', r'sinh thiết', r'tầm soát'
    ],
    'Có_triệu_chứng': [
        r'triệu chứng', r'biểu hiện', r'dấu hiệu', r'cảm thấy', r'xuất hiện', 
        r'kèm theo', r'đau', r'sốt', r'mệt', r'khó thở', r'biểu lộ'
    ]
}

# 2. Các cặp thực thể hợp lệ (Logic Y khoa cơ bản)
VALID_PAIRS = {
    'Gây_ra': [('Nguyên nhân', 'Bệnh'), ('Bệnh', 'Bệnh'), ('Nguyên nhân', 'Triệu chứng')],
    'Có_triệu_chứng': [('Bệnh', 'Triệu chứng')],
    'Điều_trị_bằng': [('Bệnh', 'Điều trị')],
    'Chẩn_đoán_bằng': [('Bệnh', 'Chẩn đoán')]
}

def _check_no_relation_rules(text, s_type, o_type):
    """Hàm nội bộ: Kiểm tra các trường hợp phủ định (chắc chắn không có quan hệ)"""
    # Rule 1: Các cặp vô lý
    if s_type == 'Điều trị' and o_type == 'Nguyên nhân': return True
    if s_type == 'Chẩn đoán' and o_type == 'Điều trị': return True
    if s_type == 'Triệu chứng' and o_type == 'Điều trị': return True 
    if s_type == 'Bệnh' and o_type == 'Nguyên nhân': return True 
    if s_type == 'Điều trị' and o_type == 'Bệnh': return True 
    if s_type == 'Điều trị' and o_type == 'Triệu chứng': return True
    if s_type == 'Triệu chứng' and o_type == 'Bệnh': return True

    # Rule 2: Liệt kê (cùng loại, ngăn cách bởi dấu phẩy/và/hoặc)
    if s_type == o_type:
        is_listing = re.fullmatch(r'\s*(,|;|và|hoặc|với|\/|-)\s*', text)
        if is_listing: return True
    return False

def apply_rules(sentence, subj_text, subj_type, obj_text, obj_type):
    """
    Hàm chính để áp dụng luật.
    Input: Câu gốc, text và type của 2 thực thể.
    Output: Nhãn quan hệ (str) hoặc None (nếu luật không bắt được).
    """
    sentence = sentence.lower()
    subj = subj_text.lower()
    obj = obj_text.lower()
    
    # Lấy văn bản ở giữa 2 thực thể
    try:
        s_idx = sentence.find(subj)
        o_idx = sentence.find(obj)
        if s_idx == -1 or o_idx == -1: return None
        
        start, end = sorted([s_idx, o_idx])
        if start == s_idx: start += len(subj)
        else: start += len(obj)
        middle_text = sentence[start:end]
    except:
        return None

    # 1. Check phủ định trước
    if _check_no_relation_rules(middle_text, subj_type, obj_type):
        return "No_relation"

    # 2. Check từ khóa (Keywords)
    detected_label = None
    for label, keywords in KEYWORDS.items():
        # Chỉ check nếu cặp Type hợp lệ
        if (subj_type, obj_type) not in VALID_PAIRS.get(label, []): continue
        
        for kw in keywords:
            if kw in middle_text:
                detected_label = label
                break
        if detected_label: break
    
    return detected_label