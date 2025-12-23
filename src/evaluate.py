import torch
import numpy as np
import pandas as pd
import itertools
import scipy.sparse
import os
import sys
import joblib
import json
import spacy
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel

# Import từ utils 
sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    add_markers, load_re_model_resources, apply_rules,
    get_bert_embedding_single, 
    MODEL_DIR, DATA_DIR, RESULT_DIR
)
from vectorizer import sentence_to_vector_w2v

# CẤU HÌNH ĐƯỜNG DẪN
PROCESSED_TEST_PATH = os.path.join(DATA_DIR, 'processed', 'test_data.csv')
PROCESSED_TEST_JSON = os.path.join(DATA_DIR, 'processed', 'test_tasks.json')


# 1. CÁC HÀM HỖ TRỢ NER (NAMED ENTITY RECOGNITION)
def _predict_ner_bert(text, model, tokenizer, id2label, device):
    """Hàm nội bộ: Dự đoán thực thể bằng BERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad(): 
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    entities = []
    current_ent = None
    idx = 0
    text_len = len(text)

    for token, pred in zip(tokens, predictions):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]: 
            continue
        
        label = id2label[pred]
        clean_token = token.replace("##", "").replace(" ", " ").strip()
        if not clean_token: continue
        
        # Mapping token back to original text position
        search_window = text[idx:min(idx+20, text_len)].lower()
        rel_start = search_window.find(clean_token.lower())
        
        if rel_start != -1:
            found_start = idx + rel_start
            found_end = found_start + len(clean_token)
            idx = found_end
            
            if label.startswith("B-"):
                if current_ent: entities.append(current_ent)
                current_ent = {"text": text[found_start:found_end], "start": found_start, "end": found_end, "label": label[2:]}
            elif label.startswith("I-"):
                if current_ent and current_ent["label"] == label[2:]:
                    current_ent["end"] = found_end
                    current_ent["text"] = text[current_ent["start"]:found_end]
                else:
                    if current_ent: entities.append(current_ent)
                    current_ent = {"text": text[found_start:found_end], "start": found_start, "end": found_end, "label": label[2:]}
            else:
                if current_ent: entities.append(current_ent); current_ent = None
                
    if current_ent: entities.append(current_ent)
    return entities

def _predict_ner_spacy(text, nlp_model):
    """Hàm nội bộ: Dự đoán thực thể bằng SpaCy."""
    doc = nlp_model(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_
        })
    return entities

def predict_ner_general(text, ner_model_obj, device=None, tokenizer=None):
    """Wrapper chung xử lý cả BERT và SpaCy."""
    if isinstance(ner_model_obj, spacy.language.Language):
        return _predict_ner_spacy(text, ner_model_obj)
    elif hasattr(ner_model_obj, 'config') and tokenizer is not None:
        return _predict_ner_bert(text, ner_model_obj, tokenizer, ner_model_obj.config.id2label, device)
    else:
        return []

def load_ner_model_unified(model_path):
    """Tự động phát hiện và load model NER (BERT hoặc SpaCy)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Thử load như SpaCy model
    if os.path.isdir(model_path) and "meta.json" in os.listdir(model_path):
        try:
            print(f"Loading SpaCy model from: {model_path}")
            nlp = spacy.load(model_path)
            return nlp, None, None
        except Exception: pass
    
    # 2. Thử load như BERT model
    try:
        print(f"Loading BERT model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        print(f"Lỗi load NER model: {e}")
        return None, None, None


# 2. CORE LOGIC DỰ ĐOÁN QUAN HỆ (RE)
def predict_re_pair_logic(text, subj, obj, vec_name, vec_model, scaler, clf, le, device, bert_tokenizer, bert_model, threshold = 0.5):
    """
    Logic cốt lõi để dự đoán quan hệ giữa 2 thực thể.
    Dùng chung cho cả Pipeline (Inference) và Evaluation.
    """
    # 1. Rule-based check (Ưu tiên cao nhất)
    rule_label = apply_rules(text, subj['text'], subj['label'], obj['text'], obj['label'])
    if rule_label: return rule_label

    if clf is None: return "No_relation"

    # 2. Tạo câu đã đánh dấu (Marked Sentence)
    marked = add_markers(text, {'start': subj['start'], 'end': subj['end']}, 
                         {'start': obj['start'], 'end': obj['end']}, "S", "O")
    
    # 3. Vector hóa
    vector = None
    if vec_name in ['bow', 'tfidf']:
        vector = vec_model.transform([marked])
    elif vec_name == 'w2v':
        # vec_model ở đây là w2v_model
        vector = sentence_to_vector_w2v(marked, vec_model, 100).reshape(1, -1)
    elif vec_name == 'bert':
        if bert_model is None or bert_tokenizer is None:
            raise AttributeError("BERT components thiếu trong quá trình dự đoán.")
        vector = get_bert_embedding_single(marked, bert_model, bert_tokenizer, device)

    # 4. Scaling 
    if scaler:
        if scipy.sparse.issparse(vector): 
            vector = scaler.transform(vector.toarray())
        else: 
            vector = scaler.transform(vector)
    
    # 5. Predict
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(vector)[0]
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]
        
        # Thresholding đơn giản
        if pred_label != "No_relation" and max_prob > threshold: 
            return pred_label
        else: 
            return "No_relation"
    else:
        pred_idx = clf.predict(vector)[0]
        return le.inverse_transform([pred_idx])[0]


# 3. CLASS PIPELINE (DÙNG CHO INFERENCE/APP/HYBRID)
class MedicalKnowledgePipeline:
    def __init__(self, ner_model_path, re_model_name, vec_name, use_silver=False):
        print(f"--- Initializing Pipeline: RE={re_model_name} ({vec_name}) ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vec_name = vec_name
        self.use_silver = use_silver
        
        # 1. Load NER
        self.ner_model, self.ner_tokenizer, self.ner_device = load_ner_model_unified(ner_model_path)
        
        # 2. Load RE Resources 
        (self.vec, self.scaler, self.clf, self.le, 
         self.bert_tokenizer, self.bert_model) = load_re_model_resources(vec_name, re_model_name, use_silver)

        if not self.clf:
            raise ValueError(f"CRITICAL: Không thể load RE model: {re_model_name}")

    def process_text(self, text):
        """Input: Raw text -> Output: (Entities, Relations)"""
        # A. NER Phase
        raw_ents = predict_ner_general(text, self.ner_model, self.ner_device, self.ner_tokenizer)
        
        # Deduplication (Lọc trùng lặp thực thể)
        unique_ents = []
        seen_texts = set()
        for ent in raw_ents:
            norm_text = ent['text'].lower().strip()
            if norm_text not in seen_texts:
                seen_texts.add(norm_text)
                unique_ents.append(ent)
        
        results = []
        # B. RE Phase
        if len(unique_ents) >= 2:
            for subj, obj in itertools.permutations(unique_ents, 2):
                label = predict_re_pair_logic(
                    text, subj, obj, 
                    self.vec_name, self.vec, self.scaler, self.clf, self.le, 
                    self.device, self.bert_tokenizer, self.bert_model
                )
                
                if label != "No_relation":
                    results.append({
                        "subject": subj['text'], "subject_type": subj['label'],
                        "relation": label, 
                        "object": obj['text'], "object_type": obj['label']
                    })
        return unique_ents, results


# 4. HÀM ĐÁNH GIÁ TỔNG HỢP & TẠO CSV 
def evaluate_all_models(use_silver=False):
    """
    Quét toàn bộ các model đã train trong thư mục models, 
    đánh giá trên tập Test (dựa vào file X_test đã lưu sẵn) và xuất ra CSV.
    Dùng để visualize kết quả.
    """
    print(f"\nBẮT ĐẦU ĐÁNH GIÁ TỔNG HỢP (Mode: {'HYBRID' if use_silver else 'STANDARD'})")

    # 1. Load Label & Test Config
    suffix = "_hybrid" if use_silver else ""
    le_path = os.path.join(MODEL_DIR, f'label_encoder{suffix}.pkl')
    
    if not os.path.exists(le_path) or not os.path.exists(PROCESSED_TEST_PATH):
        print("Lỗi: Không tìm thấy Label Encoder hoặc Test Data CSV.")
        return None, None

    df_test = pd.read_csv(PROCESSED_TEST_PATH)
    df_test['relation_label'] = df_test['relation_label'].fillna("No_relation")
    
    le = joblib.load(le_path)
    y_true = le.transform(df_test['relation_label'])

    # 2. Định nghĩa danh sách cần quét
    vector_types = ['bow', 'tfidf', 'w2v', 'bert']
    model_types = ['LogisticRegression', 'SVM', 'RandomForest', 'MLPDeepLearning']

    results = []
    best_f1 = 0.0
    best_model_name = ""
    best_vec_name = ""

    # 3. Vòng lặp đánh giá
    for vec_name in vector_types:
        print(f"--- Đang đánh giá nhóm Vector: {vec_name} ---")
        
        # Load X_test tương ứng 
        try:
            if vec_name in ['bow', 'tfidf']:
                X_test = scipy.sparse.load_npz(os.path.join(MODEL_DIR, f'X_test_{vec_name}.npz'))
            else:
                X_test = np.load(os.path.join(MODEL_DIR, f'X_test_{vec_name}.npy'))
        except Exception as e:
            print(f"   -> Bỏ qua (Chưa có file X_test): {e}")
            continue

        # Load Scaler nếu có
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{vec_name}{suffix}.pkl')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        if scaler:
            # Scaler thường yêu cầu dense matrix
            if scipy.sparse.issparse(X_test): 
                X_test_scaled = scaler.transform(X_test.toarray())
            else: 
                X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        # Duyệt qua từng model classifier
        for model_name in model_types:
            # Tên file phải khớp logic save ở train_re.py
            safe_name = model_name.replace(" ", "")
            model_path = os.path.join(MODEL_DIR, f"{safe_name}_{vec_name}{suffix}.pkl")
            
            if not os.path.exists(model_path):
                continue
                
            try:
                clf = joblib.load(model_path)
                y_pred = clf.predict(X_test_scaled)
                
                # Metrics Calculation
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                
                print(f"   -> Model: {model_name:<20} | F1: {f1:.4f}")
                
                results.append({
                    'Vectorizer': vec_name,
                    'Model': model_name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1_Macro': f1
                })
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = model_name
                    best_vec_name = vec_name
                    
            except Exception as e:
                print(f"   -> Lỗi khi evaluate {model_name}: {e}")

    # 4. Lưu kết quả
    if results:
        df_results = pd.DataFrame(results)
        save_path = os.path.join(RESULT_DIR, f'model_comparison{suffix}.csv')
        df_results.to_csv(save_path, index=False)
        print(f"\n>>> Đã lưu bảng so sánh tại: {save_path}")
        print(f">>> BEST MODEL: {best_model_name} ({best_vec_name}) - F1: {best_f1:.4f}")
        return df_results, (best_model_name, best_vec_name)
    else:
        print("Không có kết quả nào được ghi nhận.")
        return pd.DataFrame(), (None, None)

def find_best_model_from_results(use_silver=False):
    """Đọc file CSV kết quả để tìm model có F1 cao nhất."""
    suffix = "_hybrid" if use_silver else ""
    csv_path = os.path.join(RESULT_DIR, f'model_comparison{suffix}.csv')
    
    if not os.path.exists(csv_path):
        # Nếu chưa có file CSV, thử chạy hàm evaluate_all_models để tạo
        print(f"Chưa tìm thấy file {csv_path}. Đang chạy đánh giá tổng hợp...")
        _, best_info = evaluate_all_models(use_silver)
        return best_info
        
    df = pd.read_csv(csv_path)
    if df.empty: return None, None
    
    # Tìm dòng có F1_Macro cao nhất
    best_row = df.loc[df['F1_Macro'].idxmax()]
    print(f"Loaded Best Config ({'Hybrid' if use_silver else 'Supervised'}): {best_row['Model']} (Vec: {best_row['Vectorizer']}) - F1: {best_row['F1_Macro']:.4f}")
    
    return best_row['Model'], best_row['Vectorizer']

def evaluate_pipeline_gold(ner_model_path, re_model_name, vec_name, use_silver=False):
    """
    Đánh giá End-to-End Pipeline so với Gold Standard (File test_tasks.json).
    So khớp (Entity Match + Relation Match).
    """
    print(f"\n--- PIPELINE EVALUATION (Gold Standard Comparison) ---")
    
    # Khởi tạo Pipeline
    pipeline = MedicalKnowledgePipeline(ner_model_path, re_model_name, vec_name, use_silver)
    
    # Load Gold Data
    with open(PROCESSED_TEST_JSON, 'r', encoding='utf-8') as f: 
        test_tasks = json.load(f)
        
    total_gold = 0
    total_correct = 0
    total_pred = 0

    for task in test_tasks:
        text = task['data']['text']
        
        # 1. Parse Gold Relations
        gold_rels = []
        if task.get('annotations'):
            result = task['annotations'][0]['result']
            id_to_text = {r['id']: r['value']['text'] for r in result if r['type'] == 'labels'}
            for res in result:
                if res['type'] == 'relation':
                    try:
                        s = id_to_text[res['from_id']]
                        o = id_to_text[res['to_id']]
                        l = res['labels'][0]
                        gold_rels.append((s, o, l))
                    except: continue
        
        total_gold += len(gold_rels)
        
        # 2. Pipeline Prediction
        # Pipeline trả về results = [{'subject':..., 'relation':..., 'object':...}]
        _, preds = pipeline.process_text(text)
        total_pred += len(preds)
        
        # 3. Matching
        for p in preds:
            match = False
            for s_gold, o_gold, l_gold in gold_rels:
                # So sánh text (relaxed lower case check)
                s_match = (p['subject'].lower() in s_gold.lower() or s_gold.lower() in p['subject'].lower())
                o_match = (p['object'].lower() in o_gold.lower() or o_gold.lower() in p['object'].lower())
                
                if s_match and o_match and p['relation'] == l_gold:
                    match = True
                    break
            if match:
                total_correct += 1

    p = total_correct / total_pred if total_pred > 0 else 0
    r = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    print(f"Result: Gold={total_gold}, Pred={total_pred}, Correct={total_correct}")
    print(f"Pipeline F1: {f1:.4f} (Precision={p:.4f}, Recall={r:.4f})")
    return f1

if __name__ == "__main__":
    # Khi chạy trực tiếp file này, nó sẽ thực hiện đánh giá tổng hợp để tạo CSV
    print(">>> MODE: Manual Evaluation Run")
    evaluate_all_models(use_silver=False)
    evaluate_all_models(use_silver=True) 