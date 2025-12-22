import pandas as pd
import numpy as np
import os
import joblib
import json
import re
import sys
import torch
import itertools
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

sys.path.append(str(Path(__file__).resolve().parent))
from utils import add_markers, load_re_model_resources, load_unlabeled_data, MODEL_DIR, DATA_DIR
from evaluate import predict_ner_manual

# CẤU HÌNH
NER_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_bert_model')
UNLABELED_DIR = os.path.join(DATA_DIR, 'raw', 'unlabeled')
OUTPUT_SILVER_PATH = os.path.join(DATA_DIR, 'processed', 'silver_data.csv')

# --- BỘ LUẬT REGEX (TỪ KHÓA) ---
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

# --- BỘ LUẬT LOGIC (QUAN TRỌNG) ---
# Định nghĩa các cặp (Subject_Type, Object_Type) hợp lệ cho từng quan hệ
VALID_PAIRS = {
    'Gây_ra': [
        ('Nguyên nhân', 'Bệnh'), 
        ('Bệnh', 'Bệnh'),          # Biến chứng
        ('Nguyên nhân', 'Triệu chứng') # Ít gặp nhưng có thể
    ],
    'Có_triệu_chứng': [('Bệnh', 'Triệu chứng')],
    'Điều_trị_bằng': [('Bệnh', 'Điều trị')],
    'Chẩn_đoán_bằng': [('Bệnh', 'Chẩn đoán')]
}

def check_no_relation_rules(text, s_type, o_type):
    """
    Kiểm tra các trường hợp chắc chắn là No_relation (Luật phủ định)
    """
    # 1. Các cặp bất khả thi :D (Impossible Pairs)
    # Ví dụ: Thuốc (Điều trị) không bao giờ gây ra Nguyên nhân
    if s_type == 'Điều trị' and o_type == 'Nguyên nhân': return True
    if s_type == 'Chẩn đoán' and o_type == 'Điều trị': return True
    if s_type == 'Triệu chứng' and o_type == 'Điều trị': return True # Thường là Bệnh dùng thuốc, ko phải triệu chứng
    
    # 2. Cùng loại thực thể (Bệnh - Bệnh, Thuốc - Thuốc) nằm trong danh sách liệt kê
    # Ví dụ: "Bệnh nhân mắc tiểu đường, cao huyết áp và mỡ máu."
    # Giữa chúng chỉ là dấu phẩy hoặc chữ "và", "hoặc" -> No_relation
    if s_type == o_type:
        # Regex kiểm tra nếu ở giữa chỉ có dấu câu hoặc từ nối
        is_listing = re.fullmatch(r'\s*(,|;|và|hoặc|với|\/|-)\s*', text)
        if is_listing:
            return True
            
    return False

def apply_rules(sentence, subj_text, subj_type, obj_text, obj_type):
    """
    Áp dụng tập luật để gán nhãn
    """
    sentence = sentence.lower()
    subj = subj_text.lower()
    obj = obj_text.lower()
    
    # Lấy đoạn văn bản nằm giữa 2 thực thể
    try:
        s_idx = sentence.find(subj)
        o_idx = sentence.find(obj)
        if s_idx == -1 or o_idx == -1: return None
        
        start, end = sorted([s_idx, o_idx])
        # Cộng thêm độ dài của từ đứng trước để lấy khoảng giữa chính xác
        if start == s_idx: start += len(subj)
        else: start += len(obj)
            
        middle_text = sentence[start:end]
    except:
        return None

    # --- BƯỚC 1: CHECK NO_RELATION TRƯỚC ---
    if check_no_relation_rules(middle_text, subj_type, obj_type):
        return "No_relation"

    # --- BƯỚC 2: CHECK KEYWORDS ---
    detected_label = None
    
    for label, keywords in KEYWORDS.items():
        # Kiểm tra xem cặp Type này có hợp lệ với Label không
        if (subj_type, obj_type) not in VALID_PAIRS.get(label, []):
            continue
            
        for kw in keywords:
            if kw in middle_text:
                detected_label = label
                break
        if detected_label: break
    
    return detected_label

def main():
    print("BẮT ĐẦU QUY TRÌNH HYBRID LABELING...")
    
    # 1. Load Model NER & RE
    print("--> Loading Models...")
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH, use_fast=False)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model.to(device)
    
    # Load RE Model tốt nhất (Ví dụ BoW + RandomForest)
    # Cậu có thể sửa thành 'bert' + 'SVM' tùy kết quả thực nghiệm
    TARGET_VEC = 'bow' 
    TARGET_MODEL = 'RandomForest'
    vec_re, scaler_re, clf_re, le_re = load_re_model_resources(TARGET_VEC, TARGET_MODEL)
    
    # 2. Load Data
    raw_texts = load_unlabeled_data(UNLABELED_DIR)
    
    generated_samples = []
    stats = {'Rule': 0, 'Model': 0, 'No_relation': 0, 'Ignored': 0}
    
    # 3. Labeling Loop
    print(f"--> Đang xử lý {len(raw_texts)} câu...")
    for text in raw_texts:
        # A. NER Extract
        pred_ents = predict_ner_manual(text, ner_model, ner_tokenizer, ner_model.config.id2label, device)
        if len(pred_ents) < 2: continue
        
        # B. Pair Generation
        for subj, obj in itertools.permutations(pred_ents, 2):
            final_label = None
            method = None
            
            # --- PHASE 1: RULE-BASED ---
            rule_label = apply_rules(text, subj['text'], subj['label'], obj['text'], obj['label'])
            
            if rule_label:
                final_label = rule_label
                method = "Rule"
            else:
                # --- PHASE 2: MODEL-BASED (SELF-TRAINING) ---
                # Chỉ chạy model nếu Rule không bắt được (và không phải No_relation của Rule)
                marked = add_markers(
                    text,
                    {'start': subj['start'], 'end': subj['end']},
                    {'start': obj['start'], 'end': obj['end']},
                    "S", "O"
                )
                
                # Vectorize
                vector = vec_re.transform([marked])
                if scaler_re: vector = scaler_re.transform(vector)
                
                # Predict
                probs = clf_re.predict_proba(vector)[0]
                max_prob = np.max(probs)
                pred_idx = np.argmax(probs)
                pred_label_str = le_re.inverse_transform([pred_idx])[0]
                
                # Ngưỡng tin cậy (0.85 cho Positive, 0.95 cho No_relation)
                # Giúp giảm nhiễu từ model
                if pred_label_str != "No_relation" and max_prob > 0.6:
                    final_label = pred_label_str
                    method = "Model"
                elif pred_label_str == "No_relation" and max_prob > 0.7:
                    final_label = "No_relation"
                    method = "Model"
            
            # Lưu kết quả (Chỉ lưu nếu có nhãn xác định, hoặc No_relation từ Rule/HighConf Model)
            if final_label:
                stats[method if final_label != "No_relation" else 'No_relation'] += 1
                
                marked_final = add_markers(
                    text,
                    {'start': subj['start'], 'end': subj['end']},
                    {'start': obj['start'], 'end': obj['end']},
                    "S", "O"
                )
                
                generated_samples.append({
                    'original_sentence': text,
                    'subject_text': subj['text'],
                    'subject_type': subj['label'],
                    'object_text': obj['text'],
                    'object_type': obj['label'],
                    'marked_sentence': marked_final,
                    'relation_label': final_label,
                    'source': f'Silver-{method}'
                })
            else:
                stats['Ignored'] += 1

    # Xuất file
    df_silver = pd.DataFrame(generated_samples)
    if not df_silver.empty:
        df_silver.to_csv(OUTPUT_SILVER_PATH, index=False, encoding='utf-8')
        print(f"\nHoàn tất. File saved: {OUTPUT_SILVER_PATH}")
        print(f"Thống kê nhãn sinh ra: {stats}")
        print(f"Tổng số mẫu Silver: {len(df_silver)}")
    else:
        print("Không sinh ra được mẫu nào.")

if __name__ == "__main__":
    main()