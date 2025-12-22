import torch
import numpy as np
import pandas as pd
import itertools
import scipy.sparse
import os
import sys
import joblib
import json
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from gensim.models import Word2Vec

# Import từ utils
sys.path.append(str(Path(__file__).resolve().parent))
from utils import add_markers, load_re_model_resources, MODEL_DIR, DATA_DIR

# --- HÀM HỖ TRỢ NER (ĐÃ SỬA LỖI ALIGNMENT) ---
def predict_ner_manual(text, model, tokenizer, id2label, device):
    """
    Dự đoán NER và map chính xác về vị trí ký tự trong câu gốc.
    Sử dụng logic dò tìm thông minh hơn để xử lý Tokenizer âm tiết (syllable).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    # Convert ids to tokens (giữ nguyên format của tokenizer)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    entities = []
    current_ent = None
    
    # Cursor theo dõi vị trí trong văn bản gốc
    idx = 0
    text_len = len(text)
    
    for token, pred in zip(tokens, predictions):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
            
        label = id2label[pred]
        
        # Xử lý token từ BERT (thường có ## hoặc thay thế khoảng trắng bằng _)
        # Với ViHealthBERT syllable, token thường là chữ thường hoặc có dấu _
        clean_token = token.replace("##", "").replace(" ", " ").strip()
        
        if not clean_token: continue # Bỏ qua token rỗng
        
        # Tìm vị trí xuất hiện tiếp theo của token trong text
        # Cho phép bỏ qua khoảng trắng hoặc ký tự lạ giữa các token
        found_start = -1
        
        # Thử tìm trong khoảng 20 ký tự tiếp theo (cửa sổ trượt)
        search_window = text[idx:min(idx+20, text_len)].lower()
        rel_start = search_window.find(clean_token.lower())
        
        if rel_start != -1:
            found_start = idx + rel_start
            found_end = found_start + len(clean_token)
            
            # Cập nhật cursor
            idx = found_end
            
            # Logic BIO
            if label.startswith("B-"):
                if current_ent: entities.append(current_ent)
                current_ent = {
                    "text": text[found_start:found_end],
                    "start": found_start,
                    "end": found_end,
                    "label": label[2:]
                }
            elif label.startswith("I-"):
                if current_ent and current_ent["label"] == label[2:]:
                    # Nối dài entity cũ
                    current_ent["end"] = found_end
                    current_ent["text"] = text[current_ent["start"]:found_end]
                else:
                    # I- đứng một mình -> coi như B- mới
                    if current_ent: entities.append(current_ent)
                    current_ent = {
                        "text": text[found_start:found_end],
                        "start": found_start,
                        "end": found_end,
                        "label": label[2:]
                    }
            else: # Label O
                if current_ent:
                    entities.append(current_ent)
                    current_ent = None
        else:
            # Trường hợp không tìm thấy token trong window (lệch quá xa) -> Bỏ qua token này
            pass
            
    if current_ent: entities.append(current_ent)
    return entities

def sentence_to_vector_w2v(sentence, model, vector_size):
    tokens = sentence.lower().split()
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    return np.mean(model.wv[valid_tokens], axis=0)

def get_bert_embedding_single(sentence, model, tokenizer, device):
    inputs = tokenizer(sentence, return_tensors="pt", max_length=256, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# --- ĐÁNH GIÁ RE ĐỘC LẬP ---
def evaluate_re_module(test_csv_path, vec_name, model_name):
    """Đánh giá RE trả về đầy đủ Acc, F1, Matrix"""
    df_test = pd.read_csv(test_csv_path)
    df_test['relation_label'] = df_test['relation_label'].fillna("No_relation")
    
    vec, scaler, clf, le = load_re_model_resources(vec_name, model_name)
    if not clf: return None
    
    sentences = df_test['marked_sentence'].fillna("").tolist()
    
    # Vectorize (Load pre-computed cho nhanh)
    if vec_name in ['bow', 'tfidf']:
        X_test = vec.transform(sentences)
    elif vec_name == 'w2v':
        X_test = np.load(os.path.join(MODEL_DIR, 'X_test_w2v.npy'))
    elif vec_name == 'bert':
        X_test = np.load(os.path.join(MODEL_DIR, 'X_test_bert.npy'))
    
    if scaler:
        if scipy.sparse.issparse(X_test):
            X_test = scaler.transform(X_test.toarray())
        else:
            X_test = scaler.transform(X_test)
            
    y_pred = clf.predict(X_test)
    y_true = le.transform(df_test['relation_label'])
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'f1_macro': report['macro avg']['f1-score'],
        'report': report,
        'confusion_matrix': cm,
        'classes': le.classes_
    }

# --- ĐÁNH GIÁ PIPELINE ---
def evaluate_pipeline(test_json_path, ner_model_path, vec_name, re_model_name):
    """Pipeline với hàm predict_ner_manual đã sửa lỗi"""
    print(f"\n--- ĐÁNH GIÁ PIPELINE: NER + RE ({vec_name} - {re_model_name}) ---")
    
    # 1. Load Resources
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path, use_fast=False)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model.to(device)
    
    vec_re, scaler_re, clf_re, le_re = load_re_model_resources(vec_name, re_model_name)
    
    # Load raw vectorizer để infer mẫu mới
    re_tokenizer = None
    re_embed_model = None
    
    if vec_name == 'bert':
        re_name = "vinai/phobert-base"
        re_tokenizer = AutoTokenizer.from_pretrained(re_name)
        re_embed_model = AutoModel.from_pretrained(re_name)
        special_tokens = {'additional_special_tokens': ['[s]', '[/s]', '[o]', '[/o]', '[s:bệnh]', '[/s:bệnh]', '[o:bệnh]', '[/o:bệnh]', '[s:triệu chứng]', '[/s:triệu chứng]', '[o:triệu chứng]', '[/o:triệu chứng]', '[s:nguyên nhân]', '[/s:nguyên nhân]', '[o:nguyên nhân]', '[/o:nguyên nhân]', '[s:chẩn đoán]', '[/s:chẩn đoán]', '[o:chẩn đoán]', '[/o:chẩn đoán]', '[s:điều trị]', '[/s:điều trị]', '[o:điều trị]', '[/o:điều trị]']}
        re_tokenizer.add_special_tokens(special_tokens)
        re_embed_model.resize_token_embeddings(len(re_tokenizer))
        re_embed_model.to(device)

    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_tasks = json.load(f)
        
    total_gold = 0
    total_correct = 0
    total_pred = 0
    
    for task in test_tasks:
        text = task['data']['text']
        
        # Lấy Gold
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
        
        # Predict NER
        pred_ents = predict_ner_manual(text, ner_model, ner_tokenizer, ner_model.config.id2label, device)
        
        # Predict RE
        X_input = []
        meta = []
        
        if len(pred_ents) >= 2:
            for subj, obj in itertools.permutations(pred_ents, 2):
                marked = add_markers(text, {'start': subj['start'], 'end': subj['end']}, {'start': obj['start'], 'end': obj['end']}, "S", "O")
                
                # Vectorize
                vector = None
                if vec_name in ['bow', 'tfidf']:
                    vector = vec_re.transform([marked])
                elif vec_name == 'w2v':
                    vector = sentence_to_vector_w2v(marked, vec_re, 100).reshape(1, -1)
                elif vec_name == 'bert':
                    vector = get_bert_embedding_single(marked, re_embed_model, re_tokenizer, device)
                
                if scaler_re:
                    if scipy.sparse.issparse(vector): vector = scaler_re.transform(vector.toarray())
                    else: vector = scaler_re.transform(vector)
                
                X_input.append(vector)
                meta.append((subj['text'], obj['text']))
        
        if X_input:
            if scipy.sparse.issparse(X_input[0]): X_stack = scipy.sparse.vstack(X_input)
            else: X_stack = np.vstack(X_input)
                
            preds = clf_re.predict(X_stack)
            labels = le_re.inverse_transform(preds)
            
            for i, lbl in enumerate(labels):
                if lbl != "No_relation":
                    total_pred += 1
                    s_pred, o_pred = meta[i]
                    
                    match = False
                    for s_gold, o_gold, l_gold in gold_rels:
                        if l_gold == lbl and (s_gold in s_pred or s_pred in s_gold) and (o_gold in o_pred or o_pred in o_gold):
                            match = True
                            break
                    if match: total_correct += 1

    p = total_correct / total_pred if total_pred > 0 else 0
    r = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    print(f"Tổng Gold: {total_gold}, Tổng Pred: {total_pred}, Đúng: {total_correct}")
    print(f"Pipeline F1: {f1:.4f} (P={p:.4f}, R={r:.4f})")
    return f1

def inference_pipeline(text, ner_model_path, vec_name, re_model_name):
    """
    Hàm demo: Nhận văn bản thô -> Trả về danh sách quan hệ
    """
    # 1. Load Resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NER
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path, use_fast=False)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    ner_model.to(device)
    
    # RE
    _, scaler, clf, le = load_re_model_resources(vec_name, re_model_name)
    
    # Load Vectorizer riêng lẻ để infer
    vec_model = None
    bert_tokenizer = None
    
    if vec_name == 'bow':
        vec_model = joblib.load(os.path.join(MODEL_DIR, 'bow_vectorizer.joblib'))
    elif vec_name == 'tfidf':
        vec_model = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
    elif vec_name == 'w2v':
        vec_model = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
    elif vec_name == 'bert':
        bert_re_name = "vinai/phobert-base"
        bert_tokenizer = AutoTokenizer.from_pretrained(bert_re_name)
        vec_model = AutoModel.from_pretrained(bert_re_name)
        # Add special tokens
        special_tokens = {'additional_special_tokens': ['[s]', '[/s]', '[o]', '[/o]']} # Rút gọn cho demo
        bert_tokenizer.add_special_tokens(special_tokens)
        vec_model.resize_token_embeddings(len(bert_tokenizer))
        vec_model.to(device)

    # 2. Predict Entities
    pred_ents = predict_ner_manual(text, ner_model, ner_tokenizer, ner_model.config.id2label, device)
    
    results = []
    
    # 3. Predict Relations
    if len(pred_ents) >= 2:
        for subj, obj in itertools.permutations(pred_ents, 2):
            marked = add_markers(
                text,
                {'start': subj['start'], 'end': subj['end']},
                {'start': obj['start'], 'end': obj['end']},
                "S", "O"
            )
            
            # Vectorize
            vector = None
            if vec_name in ['bow', 'tfidf']:
                vector = vec_model.transform([marked])
            elif vec_name == 'w2v':
                vector = sentence_to_vector_w2v(marked, vec_model, 100).reshape(1, -1)
            elif vec_name == 'bert':
                vector = get_bert_embedding_single(marked, vec_model, bert_tokenizer, device)
            
            # Scale
            if scaler:
                if scipy.sparse.issparse(vector):
                    vector = scaler.transform(vector.toarray())
                else:
                    vector = scaler.transform(vector)
            
            # Predict
            pred_idx = clf.predict(vector)[0]
            label = le.inverse_transform([pred_idx])[0]
            
            if label != "No_relation":
                results.append({
                    "subject": subj['text'],
                    "relation": label,
                    "object": obj['text']
                })
                
    return pred_ents, results