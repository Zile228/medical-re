import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
import random
import json
import os
import sys
from pathlib import Path

# Import utils
sys.path.append(str(Path(__file__).resolve().parent))
from utils import DATA_DIR, MODEL_DIR

# Cấu hình đường dẫn
PROCESSED_TRAIN_JSON = os.path.join(DATA_DIR, 'processed', 'train_tasks.json')
PROCESSED_TEST_JSON = os.path.join(DATA_DIR, 'processed', 'test_tasks.json')
MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, 'ner_spacy_model')

def process_data_with_alignment(nlp, json_path):
    """
    Đọc JSON và dùng NLP tokenizer để căn chỉnh (align) lại entity.
    Giúp sửa lỗi W030: Misaligned entities.
    """
    training_data = []
    skipped_count = 0
    fixed_count = 0
    
    if not os.path.exists(json_path):
        print(f"Không tìm thấy file: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data:
        text = item['data']['text']
        raw_entities = []
        
        if 'annotations' in item:
            for res in item['annotations'][0]['result']:
                if res['type'] == 'labels':
                    start = res['value']['start']
                    end = res['value']['end']
                    label = res['value']['labels'][0]
                    raw_entities.append((start, end, label))
        
        # --- XỬ LÝ ALIGNMENT ---
        doc = nlp.make_doc(text)
        valid_entities = []
        
        for start, end, label in raw_entities:
            # char_span giúp map ký tự sang token
            # alignment_mode="expand": Nếu chọn thiếu/thừa 1 chút, tự mở rộng ra hết token
            span = doc.char_span(start, end, label=label, alignment_mode="expand")
            
            if span is None:
                # Nếu vẫn lỗi thì thử contract (thu hẹp lại)
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
            
            if span is not None:
                valid_entities.append((span.start_char, span.end_char, label))
            else:
                # Vẫn không khớp được thì đành bỏ qua
                pass
                
        if len(valid_entities) != len(raw_entities):
            fixed_count += 1
            
        if valid_entities:
            training_data.append((text, {"entities": valid_entities}))
        else:
            skipped_count += 1
            
    print(f"[{os.path.basename(json_path)}] Tổng: {len(data)} | Hợp lệ: {len(training_data)} | Bỏ qua (rỗng): {skipped_count}")
    return training_data

def print_detailed_report(scorer):
    """In báo cáo đẹp theo yêu cầu"""
    print("\n" + "="*55)
    print(f"{'LOẠI THỰC THỂ':<20} {'PRECISION':<10} {'RECALL':<10} {'F1-SCORE':<10}")
    print("-" * 55)

    # In chi tiết từng class
    # ents_per_type trả về dict: {'Bệnh': {'p': ..., 'r': ..., 'f': ...}, ...}
    sorted_labels = sorted(scorer.get('ents_per_type', {}).keys())
    
    for label in sorted_labels:
        metrics = scorer['ents_per_type'][label]
        p = metrics['p'] * 100
        r = metrics['r'] * 100
        f = metrics['f'] * 100
        print(f"{label:<20} {p:<10.2f} {r:<10.2f} {f:<10.2f}")
    
    print("-" * 55)
    
    # In Overall
    overall_p = scorer.get('ents_p', 0) * 100
    overall_r = scorer.get('ents_r', 0) * 100
    overall_f = scorer.get('ents_f', 0) * 100
    print(f"{'OVERALL':<20} {overall_p:<10.2f} {overall_r:<10.2f} {overall_f:<10.2f}")
    print("="*55 + "\n")

def evaluate_spacy(nlp, test_data):
    """Đánh giá model SpaCy"""
    print("\nĐang đánh giá trên tập test...")
    examples = []
    for text, annots in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annots)
        examples.append(example)
        
    scores = nlp.evaluate(examples)
    return scores

def main():
    print("--- TRAIN NER VỚI SPACY BLANK('vi') ---")
    
    # 1. Khởi tạo Model trước (để dùng tokenizer fix data)
    nlp = spacy.blank("vi") 
    
    if "ner" not in nlp.pipe_names:
        print("--> Thêm pipe 'ner'")
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # 2. Load & Fix Data Alignment
    print("--> Đang tải và xử lý dữ liệu...")
    train_data = process_data_with_alignment(nlp, PROCESSED_TRAIN_JSON)
    test_data = process_data_with_alignment(nlp, PROCESSED_TEST_JSON)
    
    # 3. Thêm nhãn vào pipe NER
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    print(f"--> Các nhãn đã học: {ner.labels}")
    
    # 4. Training Loop
    pipe_exceptions = ["ner", "trf_data"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    n_iter = 100
    dropout_start = 0.5
    dropout_end = 0.2
    
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        print(f"\nBắt đầu huấn luyện {n_iter} epochs...")
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            dropout = dropout_start - (dropout_start - dropout_end) * (itn / n_iter)
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                texts, annotations = zip(*batch)
                examples = []
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    try:
                        example = Example.from_dict(doc, annotations[i])
                        examples.append(example)
                    except ValueError:
                        continue # Bỏ qua nếu vẫn còn lỗi alignment hiếm gặp
                
                if examples:
                    nlp.update(examples, drop=dropout, losses=losses, sgd=optimizer)
            
            if (itn + 1) % 5 == 0 or itn == 0:
                print(f"Epoch {itn+1:02d}/{n_iter} | Loss: {losses.get('ner', 0):.2f}")

    # 5. Lưu Model
    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)
    
    print(f"\n--> Lưu model tại: {MODEL_OUTPUT_DIR}")
    nlp.to_disk(MODEL_OUTPUT_DIR)
    
    # 6. Đánh giá chi tiết
    scores = evaluate_spacy(nlp, test_data)
    print_detailed_report(scores)

if __name__ == "__main__":
    main()