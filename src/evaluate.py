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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from gensim.models import Word2Vec

# Import từ utils
sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    add_markers, load_re_model_resources, apply_rules, 
    MODEL_DIR, DATA_DIR
)

# --- CẤU HÌNH ---
PROCESSED_TEST_PATH = os.path.join(DATA_DIR, 'processed', 'test_data.csv')
RESULT_DIR = os.path.join(Path(__file__).resolve().parent.parent, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# --- CÁC HÀM HỖ TRỢ DỰ ĐOÁN NER (BERT & SPACY) ---
def _predict_ner_bert(text, model, tokenizer, id2label, device):
    """Dự đoán dùng BERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(): outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    entities = []
    current_ent = None
    idx = 0
    text_len = len(text)
    
    for token, pred in zip(tokens, predictions):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]: continue
        label = id2label[pred]
        clean_token = token.replace("##", "").replace(" ", " ").strip()
        if not clean_token: continue
        
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
    """Dự đoán dùng SpaCy"""
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
    """Wrapper chung cho NER"""
    if isinstance(ner_model_obj, spacy.language.Language):
        return _predict_ner_spacy(text, ner_model_obj)
    elif hasattr(ner_model_obj, 'config') and tokenizer is not None:
        return _predict_ner_bert(text, ner_model_obj, tokenizer, ner_model_obj.config.id2label, device)
    else:
        print("Lỗi: Không nhận diện được loại NER Model.")
        return []

def load_ner_model_unified(model_path):
    """Load model thông minh (SpaCy hoặc BERT)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isdir(model_path) and "meta.json" in os.listdir(model_path):
        try:
            print(f"Loading SpaCy model from: {model_path}")
            nlp = spacy.load(model_path)
            return nlp, None, None
        except Exception: pass

    try:
        print(f"Loading BERT model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        print(f"Lỗi load NER model: {e}")
        return None, None, None

# --- CÁC HÀM VECTORIZER & PREDICT ---
def sentence_to_vector_w2v(sentence, model, vector_size):
    tokens = sentence.lower().split()
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens: return np.zeros(vector_size)
    return np.mean(model.wv[valid_tokens], axis=0)

def get_bert_embedding_single(sentence, model, tokenizer, device):
    inputs = tokenizer(sentence, return_tensors="pt", max_length=256, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad(): outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def predict_relation_hybrid(text, subj, obj, vec_model, clf_model, scaler, le, vec_name, device, bert_tokenizer=None, bert_embed_model=None):
    rule_label = apply_rules(text, subj['text'], subj['label'], obj['text'], obj['label'])
    if rule_label: return rule_label
    
    if clf_model is None: return "No_relation"
    
    marked = add_markers(text, {'start': subj['start'], 'end': subj['end']}, 
                         {'start': obj['start'], 'end': obj['end']}, "S", "O")
    
    vector = None
    if vec_name in ['bow', 'tfidf']:
        vector = vec_model.transform([marked])
    elif vec_name == 'w2v':
        vector = sentence_to_vector_w2v(marked, vec_model, 100).reshape(1, -1)
    elif vec_name == 'bert':
        vector = get_bert_embedding_single(marked, bert_embed_model, bert_tokenizer, device)
    
    if scaler:
        if scipy.sparse.issparse(vector): vector = scaler.transform(vector.toarray())
        else: vector = scaler.transform(vector)
            
    if hasattr(clf_model, "predict_proba"):
        probs = clf_model.predict_proba(vector)[0]
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]
        if pred_label != "No_relation" and max_prob > 0.5: return pred_label
        else: return "No_relation"
    else:
        pred_idx = clf_model.predict(vector)[0]
        return le.inverse_transform([pred_idx])[0]

# --- HÀM ĐÁNH GIÁ TỔNG HỢP (MỚI) ---
def evaluate_all_models(use_silver=False):
    """
    Đánh giá toàn bộ các tổ hợp Vectorizer + Model có trong folder models.
    Hỗ trợ cả chế độ thường và hybrid (dựa vào suffix file).
    """
    print(f"\n--- BẮT ĐẦU ĐÁNH GIÁ TỔNG HỢP (Mode: {'HYBRID' if use_silver else 'STANDARD'}) ---")
    
    # 1. Load Test Data & Labels
    if not os.path.exists(PROCESSED_TEST_PATH):
        print("Không tìm thấy file test data csv.")
        return None, None

    df_test = pd.read_csv(PROCESSED_TEST_PATH)
    df_test['relation_label'] = df_test['relation_label'].fillna("No_relation")
    
    suffix = "_hybrid" if use_silver else ""
    le_path = os.path.join(MODEL_DIR, f'label_encoder{suffix}.pkl')
    
    if not os.path.exists(le_path):
        print(f"Không tìm thấy Label Encoder tại {le_path}")
        return None, None

    le = joblib.load(le_path)
    y_true = le.transform(df_test['relation_label'])
    
    # 2. Định nghĩa cấu hình
    vector_types = ['bow', 'tfidf', 'w2v', 'bert']
    model_types = ['LogisticRegression', 'SVM', 'RandomForest', 'MLPDeepLearning'] # Tên khớp với logic save ở train_re
    
    results = []
    best_f1 = 0.0
    best_model_name = ""
    best_vec_name = ""
    
    # 3. Vòng lặp đánh giá
    for vec_name in vector_types:
        # Load X_test tương ứng
        try:
            if vec_name in ['bow', 'tfidf']:
                X_test = scipy.sparse.load_npz(os.path.join(MODEL_DIR, f'X_test_{vec_name}.npz'))
            else:
                X_test = np.load(os.path.join(MODEL_DIR, f'X_test_{vec_name}.npy'))
        except Exception as e:
            print(f"Bỏ qua vector {vec_name} (chưa có file test): {e}")
            continue

        # Load Scaler nếu có
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{vec_name}{suffix}.pkl')
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        if scaler:
            print(f"Applying scaler for {vec_name}...")
            if scipy.sparse.issparse(X_test): X_test = X_test.toarray() # Scaler thường cần dense
            X_test = scaler.transform(X_test)
        
        # Duyệt qua từng model
        for model_name in model_types:
            # Tên file giống logic train_re: RandomForest_bow.pkl hoặc RandomForest_bow_hybrid.pkl
            # model_name trong list ở trên cần map về tên file (bỏ khoảng trắng nếu có)
            safe_name = model_name.replace(" ", "")
            model_path = os.path.join(MODEL_DIR, f"{safe_name}_{vec_name}{suffix}.pkl")
            
            if not os.path.exists(model_path):
                continue
                
            try:
                clf = joblib.load(model_path)
                y_pred = clf.predict(X_test)
                
                # Metrics
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                
                print(f"Checked: {model_name} + {vec_name} -> F1: {f1:.4f}")
                
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
                print(f"Lỗi khi evaluate {model_name}_{vec_name}: {e}")

    # 4. Lưu kết quả
    if results:
        df_results = pd.DataFrame(results)
        save_path = os.path.join(RESULT_DIR, f'model_comparison{suffix}.csv')
        df_results.to_csv(save_path, index=False)
        print(f"\nĐã lưu bảng so sánh tại: {save_path}")
        print(f"BEST MODEL: {best_model_name} ({best_vec_name}) - F1: {best_f1:.4f}")
        return df_results, (best_vec_name, best_model_name)
    else:
        print("Không có kết quả nào được ghi nhận.")
        return pd.DataFrame(), (None, None)

# --- PIPELINE ĐÁNH GIÁ (GIỮ NGUYÊN NHƯNG UPDATE LOAD RES) ---
def evaluate_pipeline(test_json_path, ner_model_path, vec_name, re_model_name, use_silver=False):
    print(f"\n--- PIPELINE EVALUATION (Silver={use_silver}): NER + {re_model_name} ({vec_name}) ---")
    
    # 1. Load NER
    ner_model_obj, ner_tokenizer, ner_device = load_ner_model_unified(ner_model_path)
    if ner_model_obj is None: return 0
    
    # 2. Load RE
    re_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logic load RE resources tùy chỉnh để khớp với suffix
    # Lưu ý: load_re_model_resources trong utils có thể chưa hỗ trợ suffix, ta load thủ công ở đây cho chắc
    suffix = "_hybrid" if use_silver else ""
    
    # Load Vectorizer (thường vectorizer gốc không đổi, chỉ model thay đổi, trừ khi train lại vectorizer)
    # Tuy nhiên, trong train_re logic, vectorizer được load lại.
    if vec_name == 'bow': vec_re = joblib.load(os.path.join(MODEL_DIR, 'bow_vectorizer.joblib'))
    elif vec_name == 'tfidf': vec_re = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
    elif vec_name == 'w2v': vec_re = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
    else: vec_re = None # BERT xử lý riêng
        
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{vec_name}{suffix}.pkl')
    scaler_re = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    safe_name = re_model_name.replace(" ", "")
    clf_path = os.path.join(MODEL_DIR, f"{safe_name}_{vec_name}{suffix}.pkl")
    if not os.path.exists(clf_path):
        print(f"Không tìm thấy model RE: {clf_path}")
        return 0
    clf_re = joblib.load(clf_path)
    
    le_path = os.path.join(MODEL_DIR, f'label_encoder{suffix}.pkl')
    le_re = joblib.load(le_path)
    
    bert_re_tokenizer = None; bert_re_model = None
    if vec_name == 'bert':
        bert_re_name = "vinai/phobert-base"
        bert_re_tokenizer = AutoTokenizer.from_pretrained(bert_re_name)
        bert_re_model = AutoModel.from_pretrained(bert_re_name)
        bert_re_tokenizer.add_special_tokens({'additional_special_tokens': ['[s]', '[/s]', '[o]', '[/o]']})
        bert_re_model.resize_token_embeddings(len(bert_re_tokenizer))
        bert_re_model.to(re_device)

    # 3. Predict Loop
    with open(test_json_path, 'r', encoding='utf-8') as f: test_tasks = json.load(f)
    total_gold = 0; total_correct = 0; total_pred = 0
    
    for task in test_tasks:
        text = task['data']['text']
        gold_rels = []
        if task.get('annotations'):
            result = task['annotations'][0]['result']
            id_to_text = {r['id']: r['value']['text'] for r in result if r['type'] == 'labels'}
            for res in result:
                if res['type'] == 'relation':
                    try:
                        s, o, l = id_to_text[res['from_id']], id_to_text[res['to_id']], res['labels'][0]
                        gold_rels.append((s, o, l))
                    except: continue
        total_gold += len(gold_rels)
        
        pred_ents = predict_ner_general(text, ner_model_obj, ner_device, ner_tokenizer)
        
        # Deduplication for EVALUATION (Optional but good for consistency)
        # Ở đây ta không cần deduplicate quá gắt vì ta đang so sánh với Gold, 
        # nhưng việc so sánh bên dưới đã dùng lower() nên cũng an toàn.
        
        if len(pred_ents) >= 2:
            for subj, obj in itertools.permutations(pred_ents, 2):
                pred_label = predict_relation_hybrid(
                    text, subj, obj, 
                    vec_re, clf_re, scaler_re, le_re, vec_name, re_device, 
                    bert_re_tokenizer, bert_re_model
                )
                
                if pred_label != "No_relation":
                    total_pred += 1
                    match = False
                    for s_gold, o_gold, l_gold in gold_rels:
                        # --- CẬP NHẬT: So sánh không phân biệt hoa thường ---
                        s_gold_norm = s_gold.lower().strip()
                        o_gold_norm = o_gold.lower().strip()
                        s_pred_norm = subj['text'].lower().strip()
                        o_pred_norm = obj['text'].lower().strip()
                        
                        text_match = (
                            (s_gold_norm in s_pred_norm or s_pred_norm in s_gold_norm) and
                            (o_gold_norm in o_pred_norm or o_pred_norm in o_gold_norm)
                        )
                        
                        if l_gold == pred_label and text_match:
                            match = True; break
                    if match: total_correct += 1

    p = total_correct / total_pred if total_pred > 0 else 0
    r = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    print(f"Result: Gold={total_gold}, Pred={total_pred}, Correct={total_correct}")
    print(f"Pipeline F1: {f1:.4f} (Precision={p:.4f}, Recall={r:.4f})")
    return f1

def inference_pipeline(text, ner_model_path, vec_name, re_model_name, use_silver=False):
    """Hàm Demo"""
    # Load resources (Tương tự evaluate pipeline)
    # LƯU Ý: KHÔNG lower text ở đây để NER hoạt động tốt nhất!
    
    ner_model_obj, ner_tokenizer, ner_device = load_ner_model_unified(ner_model_path)
    re_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    suffix = "_hybrid" if use_silver else ""
    
    if vec_name == 'bow': vec_re = joblib.load(os.path.join(MODEL_DIR, 'bow_vectorizer.joblib'))
    elif vec_name == 'tfidf': vec_re = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
    elif vec_name == 'w2v': vec_re = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
    else: vec_re = None
    
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{vec_name}{suffix}.pkl')
    scaler_re = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    safe_name = re_model_name.replace(" ", "")
    clf_re = joblib.load(os.path.join(MODEL_DIR, f"{safe_name}_{vec_name}{suffix}.pkl"))
    le_re = joblib.load(os.path.join(MODEL_DIR, f'label_encoder{suffix}.pkl'))
    
    bert_re_tokenizer = None; bert_re_model = None
    if vec_name == 'bert':
        bert_re_name = "vinai/phobert-base"
        bert_re_tokenizer = AutoTokenizer.from_pretrained(bert_re_name)
        bert_re_model = AutoModel.from_pretrained(bert_re_name)
        bert_re_tokenizer.add_special_tokens({'additional_special_tokens': ['[s]', '[/s]', '[o]', '[/o]']})
        bert_re_model.resize_token_embeddings(len(bert_re_tokenizer))
        bert_re_model.to(re_device)

    # 1. Dự đoán thực thể (dùng text gốc có hoa/thường)
    raw_ents = predict_ner_general(text, ner_model_obj, ner_device, ner_tokenizer)
    
    # 2. Lọc trùng lặp thực thể (Deduplication)
    # Nếu "Tiểu đường" và "tiểu đường" cùng được tìm thấy, chỉ giữ lại 1 để tránh duplicate quan hệ
    unique_ents = []
    seen_texts = set()
    
    for ent in raw_ents:
        norm_text = ent['text'].lower().strip()
        if norm_text not in seen_texts:
            seen_texts.add(norm_text)
            unique_ents.append(ent)
    
    results = []
    
    if len(unique_ents) >= 2:
        for subj, obj in itertools.permutations(unique_ents, 2):
            label = predict_relation_hybrid(
                text, subj, obj, 
                vec_re, clf_re, scaler_re, le_re, vec_name, re_device, 
                bert_re_tokenizer, bert_re_model
            )
            if label != "No_relation":
                results.append({"subject": subj['text'], "relation": label, "object": obj['text']})
    return unique_ents, results

if __name__ == "__main__":
    # Test nhanh khi chạy script trực tiếp
    evaluate_all_models(use_silver=True)