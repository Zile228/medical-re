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
from utils import (
    add_markers, load_re_model_resources, apply_rules, # <--- Import hàm luật dùng chung
    MODEL_DIR, DATA_DIR
)

# --- CÁC HÀM HỖ TRỢ NER VÀ VECTORIZER ---

def predict_ner_manual(text, model, tokenizer, id2label, device):
    """
    Hàm dự đoán NER và align lại vị trí ký tự trong câu gốc.
    """
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

# --- HÀM DỰ ĐOÁN HYBRID (Rules + Model) ---
def predict_relation_hybrid(text, subj, obj, vec_model, clf_model, scaler, le, vec_name, device, bert_tokenizer=None, bert_embed_model=None):
    """
    Quy trình:
    1. Check Rules (được import từ utils).
    2. Nếu Rules trả về None -> Check Model.
    """
    # 1. Thử Rules trước
    rule_label = apply_rules(text, subj['text'], subj['label'], obj['text'], obj['label'])
    if rule_label:
        return rule_label # Ưu tiên luật
    
    # 2. Nếu Rules bó tay, dùng Model
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
            
    # Logic ngưỡng (Threshold)
    if hasattr(clf_model, "predict_proba"):
        probs = clf_model.predict_proba(vector)[0]
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]
        
        # Ngưỡng thấp hơn khi inference (vì ta muốn phát hiện càng nhiều càng tốt)
        # 0.5 là mức trung bình
        if pred_label != "No_relation" and max_prob > 0.5:
            return pred_label
        else:
            return "No_relation"
    else:
        pred_idx = clf_model.predict(vector)[0]
        return le.inverse_transform([pred_idx])[0]

# --- ĐÁNH GIÁ RE MODEL (CHỈ MODEL) ---
def evaluate_re_module(test_csv_path, vec_name, model_name):
    """Đánh giá khả năng học của Model (không có Rules)"""
    df_test = pd.read_csv(test_csv_path)
    df_test['relation_label'] = df_test['relation_label'].fillna("No_relation")
    vec, scaler, clf, le = load_re_model_resources(vec_name, model_name)
    if not clf: return None
    
    sentences = df_test['marked_sentence'].fillna("").tolist()
    if vec_name in ['bow', 'tfidf']: X_test = vec.transform(sentences)
    elif vec_name == 'w2v': X_test = np.load(os.path.join(MODEL_DIR, 'X_test_w2v.npy'))
    elif vec_name == 'bert': X_test = np.load(os.path.join(MODEL_DIR, 'X_test_bert.npy'))
    
    if scaler:
        if scipy.sparse.issparse(X_test): X_test = scaler.transform(X_test.toarray())
        else: X_test = scaler.transform(X_test)
            
    y_pred = clf.predict(X_test)
    y_true = le.transform(df_test['relation_label'])
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy': acc, 'f1_macro': report['macro avg']['f1-score'], 'report': report, 'confusion_matrix': cm, 'classes': le.classes_}

# --- ĐÁNH GIÁ TOÀN BỘ PIPELINE (HYBRID) ---
def evaluate_pipeline(test_json_path, ner_model_path, vec_name, re_model_name):
    print(f"\n--- ĐÁNH GIÁ PIPELINE: NER + HYBRID RE ({vec_name} - {re_model_name}) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path, use_fast=False)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    ner_model.to(device)
    
    vec_re, scaler_re, clf_re, le_re = load_re_model_resources(vec_name, re_model_name)
    
    bert_re_tokenizer = None; bert_re_model = None
    if vec_name == 'bert':
        bert_re_name = "vinai/phobert-base"
        bert_re_tokenizer = AutoTokenizer.from_pretrained(bert_re_name)
        bert_re_model = AutoModel.from_pretrained(bert_re_name)
        bert_re_tokenizer.add_special_tokens({'additional_special_tokens': ['[s]', '[/s]', '[o]', '[/o]']})
        bert_re_model.resize_token_embeddings(len(bert_re_tokenizer))
        bert_re_model.to(device)

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
        
        pred_ents = predict_ner_manual(text, ner_model, ner_tokenizer, ner_model.config.id2label, device)
        
        if len(pred_ents) >= 2:
            for subj, obj in itertools.permutations(pred_ents, 2):
                pred_label = predict_relation_hybrid(
                    text, subj, obj, 
                    vec_re, clf_re, scaler_re, le_re, vec_name, device, 
                    bert_re_tokenizer, bert_re_model
                )
                
                if pred_label != "No_relation":
                    total_pred += 1
                    match = False
                    for s_gold, o_gold, l_gold in gold_rels:
                        if l_gold == pred_label and (s_gold in subj['text'] or subj['text'] in s_gold) and (o_gold in obj['text'] or obj['text'] in o_gold):
                            match = True; break
                    if match: total_correct += 1

    p = total_correct / total_pred if total_pred > 0 else 0
    r = total_correct / total_gold if total_gold > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    print(f"Tổng Gold: {total_gold}, Tổng Pred: {total_pred}, Đúng: {total_correct}")
    print(f"Hybrid Pipeline F1: {f1:.4f} (P={p:.4f}, R={r:.4f})")
    return f1

# --- INFERENCE PIPELINE (DEMO) ---
def inference_pipeline(text, ner_model_path, vec_name, re_model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_path, use_fast=False)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
    ner_model.to(device)
    
    vec_re, scaler_re, clf_re, le_re = load_re_model_resources(vec_name, re_model_name)
    
    bert_re_tokenizer = None; bert_re_model = None
    if vec_name == 'bow': vec_re = joblib.load(os.path.join(MODEL_DIR, 'bow_vectorizer.joblib'))
    elif vec_name == 'tfidf': vec_re = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
    elif vec_name == 'w2v': vec_re = Word2Vec.load(os.path.join(MODEL_DIR, 'word2vec.model'))
    elif vec_name == 'bert':
        bert_re_name = "vinai/phobert-base"
        bert_re_tokenizer = AutoTokenizer.from_pretrained(bert_re_name)
        bert_re_model = AutoModel.from_pretrained(bert_re_name)
        bert_re_tokenizer.add_special_tokens({'additional_special_tokens': ['[s]', '[/s]', '[o]', '[/o]']})
        bert_re_model.resize_token_embeddings(len(bert_re_tokenizer))
        bert_re_model.to(device)

    pred_ents = predict_ner_manual(text, ner_model, ner_tokenizer, ner_model.config.id2label, device)
    results = []
    
    if len(pred_ents) >= 2:
        for subj, obj in itertools.permutations(pred_ents, 2):
            label = predict_relation_hybrid(
                text, subj, obj, 
                vec_re, clf_re, scaler_re, le_re, vec_name, device, 
                bert_re_tokenizer, bert_re_model
            )
            if label != "No_relation":
                results.append({"subject": subj['text'], "relation": label, "object": obj['text']})
    return pred_ents, results