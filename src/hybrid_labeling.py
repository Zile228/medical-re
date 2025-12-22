import pandas as pd
import numpy as np
import os
import sys
import torch
import itertools
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import từ utils (bao gồm apply_rules vừa chuyển sang)
sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    add_markers, load_re_model_resources, load_unlabeled_data, 
    apply_rules, # <--- Import hàm này
    MODEL_DIR, DATA_DIR
)
from evaluate import predict_ner_manual

# CẤU HÌNH
NER_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_bert_model')
UNLABELED_DIR = os.path.join(DATA_DIR, 'raw', 'unlabeled')
OUTPUT_SILVER_PATH = os.path.join(DATA_DIR, 'processed', 'silver_data.csv')

def main():
    print("BẮT ĐẦU QUY TRÌNH HYBRID LABELING...")
    
    # 1. Load Resources
    print("--> Loading Models...")
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH, use_fast=False)
    ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model.to(device)
    
    # Load Model RE tốt nhất để gán nhãn phụ trợ
    TARGET_VEC = 'bow' 
    TARGET_MODEL = 'RandomForest'
    vec_re, scaler_re, clf_re, le_re = load_re_model_resources(TARGET_VEC, TARGET_MODEL)
    
    # 2. Load Data
    raw_texts = load_unlabeled_data(UNLABELED_DIR)
    
    generated_samples = []
    stats = {'Rule': 0, 'Model': 0, 'No_relation': 0, 'Ignored': 0}
    
    print(f"--> Đang xử lý {len(raw_texts)} câu...")
    for text in raw_texts:
        # A. NER Extract
        pred_ents = predict_ner_manual(text, ner_model, ner_tokenizer, ner_model.config.id2label, device)
        if len(pred_ents) < 2: continue
        
        # B. Duyệt từng cặp thực thể
        for subj, obj in itertools.permutations(pred_ents, 2):
            final_label = None
            method = None
            
            # --- PHASE 1: RULE-BASED (Gọi từ utils) ---
            rule_label = apply_rules(text, subj['text'], subj['label'], obj['text'], obj['label'])
            
            if rule_label:
                final_label = rule_label
                method = "Rule"
            else:
                # --- PHASE 2: MODEL-BASED (SELF-TRAINING) ---
                # Chỉ chạy model nếu Rule không bắt được
                marked = add_markers(
                    text,
                    {'start': subj['start'], 'end': subj['end']},
                    {'start': obj['start'], 'end': obj['end']},
                    "S", "O"
                )
                
                # Vectorize & Predict
                vector = vec_re.transform([marked])
                if scaler_re: vector = scaler_re.transform(vector)
                
                if hasattr(clf_re, "predict_proba"):
                    probs = clf_re.predict_proba(vector)[0]
                    max_prob = np.max(probs)
                    pred_idx = np.argmax(probs)
                    pred_label_str = le_re.inverse_transform([pred_idx])[0]
                    
                    # Ngưỡng tin cậy cao cho dữ liệu Silver
                    if pred_label_str != "No_relation" and max_prob > 0.85:
                        final_label = pred_label_str
                        method = "Model"
                    elif pred_label_str == "No_relation" and max_prob > 0.95:
                        final_label = "No_relation"
                        method = "Model"
            
            # C. Lưu kết quả
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
        print(f"Thống kê: {stats}")
    else:
        print("Không sinh ra được mẫu nào.")

if __name__ == "__main__":
    main()