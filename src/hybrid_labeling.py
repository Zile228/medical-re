import pandas as pd
import os
import sys
import itertools
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    load_unlabeled_data, add_markers, 
    DATA_DIR, MODEL_DIR
)
# Import Pipeline và Logic dự đoán lẻ (để loop qua từng cặp)
from evaluate import MedicalKnowledgePipeline, find_best_model_from_results, predict_re_pair_logic, predict_ner_general

# CẤU HÌNH
NER_MODEL_PATH = os.path.join(MODEL_DIR, 'ner_spacy_model') 
UNLABELED_DIR = os.path.join(DATA_DIR, 'raw', 'unlabeled')
OUTPUT_SILVER_PATH = os.path.join(DATA_DIR, 'processed', 'silver_data.csv')

def main():
    """File này cần import predict_re_pair_logic từ evaluate để chạy logic dự đoán thủ công 
    (do chúng ta muốn gán nhãn cho cả trường hợp "No_relation" để tạo dữ liệu training phong phú 
    hơn, trong khi hàm process_text của Pipeline chỉ trả về positive relations)."""

    print("BẮT ĐẦU QUY TRÌNH HYBRID LABELING...")

    # 1. Tự động tìm model tốt nhất từ Supervised Phase
    # (Lưu ý: Luôn tìm từ Supervised trước để tạo silver data)
    best_model_name, best_vec_name = find_best_model_from_results(use_silver=False)
    
    if not best_model_name:
        print("Lỗi: Không tìm thấy thông tin model tốt nhất. Hãy chạy train_re.py (Supervised) trước.")
        return

    # 2. Khởi tạo Pipeline (Load tài nguyên 1 lần dùng mãi mãi)
    try:
        pipeline = MedicalKnowledgePipeline(
            ner_model_path=NER_MODEL_PATH,
            re_model_name=best_model_name,
            vec_name=best_vec_name,
            use_silver=False
        )
    except Exception as e:
        print(f"Lỗi khởi tạo pipeline: {e}")
        return

    # 3. Load Data & Predict
    raw_texts = load_unlabeled_data(UNLABELED_DIR)
    generated_samples = []
    stats = {'Relations_Found': 0, 'No_relation': 0, 'Ignored': 0}

    print(f"--> Đang xử lý {len(raw_texts)} câu...")
    for text in raw_texts:
        # A. NER Phase (Sử dụng hàm từ evaluate nhưng truyền resources của pipeline vào)
        ents = predict_ner_general(text, pipeline.ner_model, pipeline.ner_device, pipeline.ner_tokenizer)
        
        # Deduplication
        unique_ents = []
        seen_texts = set()
        for ent in ents:
            norm_text = ent['text'].lower().strip()
            if norm_text not in seen_texts:
                seen_texts.add(norm_text)
                unique_ents.append(ent)

        if len(unique_ents) < 2: 
            stats['Ignored'] += 1
            continue

        # B. RE Phase (Duyệt qua tất cả các cặp để gán nhãn)
        for subj, obj in itertools.permutations(unique_ents, 2):
            # Gọi hàm logic cốt lõi, truyền tài nguyên từ pipeline vào
            label = predict_re_pair_logic(
                text, subj, obj, 
                vec_name=pipeline.vec_name,
                vec_model=pipeline.vec,
                scaler=pipeline.scaler,
                clf=pipeline.clf,
                le=pipeline.le,
                device=pipeline.device,
                bert_tokenizer=pipeline.bert_tokenizer,
                bert_model=pipeline.bert_model,
                threshold = 0.8
            )
            
            if label != "No_relation":
                stats['Relations_Found'] += 1
            else:
                stats['No_relation'] += 1
                
            # C. Tạo mẫu dữ liệu Silver
            marked_final = add_markers(text, {'start': subj['start'], 'end': subj['end']}, 
                                       {'start': obj['start'], 'end': obj['end']}, "S", "O")
            
            generated_samples.append({
                'original_sentence': text,
                'subject_text': subj['text'], 'subject_type': subj['label'],
                'object_text': obj['text'], 'object_type': obj['label'],
                'marked_sentence': marked_final,
                'relation_label': label,
                'source': f'Silver-{best_model_name}'
            })

    # 4. Lưu kết quả
    df_silver = pd.DataFrame(generated_samples)
    if not df_silver.empty:
        df_silver.to_csv(OUTPUT_SILVER_PATH, index=False, encoding='utf-8')
        print(f"\nHoàn tất. File saved: {OUTPUT_SILVER_PATH}")
        print(f"Thống kê: {stats}")
    else:
        print("Không sinh ra được mẫu nào.")

if __name__ == "__main__":
    main()