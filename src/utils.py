import os
import joblib
import json
import re
import glob
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# CẤU HÌNH ĐƯỜNG DẪN CHUNG
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
TOKENIZER_DIR = os.path.join(MODEL_DIR, 'tokenizer_config')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Tên Model BERT dùng chung
BERT_MODEL_NAME = "vinai/phobert-base"

# CÁC HÀM XỬ LÝ VĂN BẢN CƠ BẢN
def load_unlabeled_data(unlabeled_dir):
    """
    Đọc các file json chưa gán nhãn từ thư mục quy định.
    Trả về danh sách các câu văn bản duy nhất.
    """
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
                        if txt:
                            all_texts.append(txt)
        except Exception as e:
            print(f"Lỗi đọc file {path}: {e}")
            
    unique_texts = list(set(all_texts))
    print(f"Đã tải {len(unique_texts)} câu văn bản duy nhất.")
    return unique_texts

def simple_tokenizer(text):
    """Tokenizer đơn giản dựa trên khoảng trắng."""
    return text.split()

def add_markers(text, subj_span, obj_span, subj_type, obj_type):
    """
    Chèn marker đánh dấu thực thể vào câu.
    Ví dụ: [S]Tiểu đường[/S] ... [O]metformin[/O]
    """
    s_start, s_end = subj_span['start'], subj_span['end']
    o_start, o_end = obj_span['start'], obj_span['end']

    insertions = [
        (s_end, f" [/{subj_type}]"), (s_start, f"[{subj_type}] "),
        (o_end, f" [/{obj_type}]"), (o_start, f"[{obj_type}] ")
    ]
    # Sắp xếp giảm dần theo vị trí để không làm lệch chỉ số khi chèn
    insertions.sort(key=lambda x: x[0], reverse=True)

    modified_text = text
    for pos, string in insertions:
        modified_text = modified_text[:pos] + string + modified_text[pos:]
    return modified_text.lower()

def get_bert_embedding_single(sentence, model, tokenizer, device):
    """Lấy vector embedding [CLS] cho một câu đơn bằng BERT."""
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", max_length=256, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def load_re_model_resources(vec_name, model_name, use_silver=False):
    """
    Load toàn bộ tài nguyên cần thiết cho Relation Extraction.
    Hỗ trợ xử lý hậu tố _hybrid nếu use_silver=True.
    """
    suffix = "_hybrid" if use_silver else ""
    vec = None
    bert_tokenizer = None
    bert_model = None

    print(f"Đang tải tài nguyên RE: {model_name} ({vec_name}) [Hybrid={use_silver}]")

    try:
        # 1. Load Vectorizer
        if vec_name == 'bow':
            vec = joblib.load(os.path.join(MODEL_DIR, 'bow_vectorizer.joblib'))
        elif vec_name == 'tfidf':
            vec = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
        elif vec_name == 'w2v':
            from gensim.models import Word2Vec
            vec = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
        elif vec_name == 'bert':
            # Load BERT Tokenizer & Model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
            bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
            
            # Thêm token đặc biệt giống như lúc training
            special_tokens = {
                'additional_special_tokens': [
                    '[s]', '[/s]', '[o]', '[/o]', 
                    '[s:bệnh]', '[/s:bệnh]', '[o:bệnh]', '[/o:bệnh]',
                    '[s:triệu chứng]', '[/s:triệu chứng]', '[o:triệu chứng]', '[/o:triệu chứng]',
                    '[s:nguyên nhân]', '[/s:nguyên nhân]', '[o:nguyên nhân]', '[/o:nguyên nhân]',
                    '[s:chẩn đoán]', '[/s:chẩn đoán]', '[o:chẩn đoán]', '[/o:chẩn đoán]',
                    '[s:điều trị]', '[/s:điều trị]', '[o:điều trị]', '[/o:điều trị]'
                ]
            }
            bert_tokenizer.add_special_tokens(special_tokens)
            bert_model.resize_token_embeddings(len(bert_tokenizer))
            bert_model.to(device)

        # 2. Load Scaler
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{vec_name}{suffix}.pkl')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        # 3. Load Classifier Model
        safe_name = model_name.replace(" ", "").replace("(", "").replace(")", "")
        clf_path = os.path.join(MODEL_DIR, f"{safe_name}_{vec_name}{suffix}.pkl")
        if not os.path.exists(clf_path):
            print(f"Không tìm thấy file model: {clf_path}")
            return None, None, None, None, None, None
        
        clf = joblib.load(clf_path)
        
        # 4. Load Label Encoder
        le_path = os.path.join(MODEL_DIR, f'label_encoder{suffix}.pkl')
        le = joblib.load(le_path)
        
        return vec, scaler, clf, le, bert_tokenizer, bert_model

    except Exception as e:
        print(f"Lỗi khi load tài nguyên ({vec_name}, {model_name}): {e}")
        return None, None, None, None, None, None

# BỘ LUẬT RULE-BASED
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

VALID_PAIRS = {
    'Gây_ra': [('Nguyên nhân', 'Bệnh'), ('Bệnh', 'Bệnh'), ('Nguyên nhân', 'Triệu chứng')],
    'Có_triệu_chứng': [('Bệnh', 'Triệu chứng')],
    'Điều_trị_bằng': [('Bệnh', 'Điều trị')],
    'Chẩn_đoán_bằng': [('Bệnh', 'Chẩn đoán')]
}

def _check_no_relation_rules(text, s_type, o_type):
    """Kiểm tra các trường hợp chắc chắn không có quan hệ."""
    if s_type == 'Điều trị' and o_type == 'Nguyên nhân': return True
    if s_type == 'Chẩn đoán' and o_type == 'Điều trị': return True
    if s_type == 'Triệu chứng' and o_type == 'Điều trị': return True
    if s_type == 'Bệnh' and o_type == 'Nguyên nhân': return True
    if s_type == 'Điều trị' and o_type == 'Bệnh': return True
    if s_type == 'Điều trị' and o_type == 'Triệu chứng': return True
    if s_type == 'Triệu chứng' and o_type == 'Bệnh': return True

    # Rule: Liệt kê
    if s_type == o_type:
        is_listing = re.fullmatch(r'\s*(,|;|và|hoặc|với|\/|-)\s*', text)
        if is_listing: return True
    return False

def apply_rules(sentence, subj_text, subj_type, obj_text, obj_type):
    """Áp dụng bộ luật Rule-based để gán nhãn."""
    sentence = sentence.lower()
    subj = subj_text.lower()
    obj = obj_text.lower()

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

    if _check_no_relation_rules(middle_text, subj_type, obj_type):
        return "No_relation"

    detected_label = None
    for label, keywords in KEYWORDS.items():
        if (subj_type, obj_type) not in VALID_PAIRS.get(label, []): continue
        
        for kw in keywords:
            if kw in middle_text:
                detected_label = label
                break
        if detected_label: break
    return detected_label