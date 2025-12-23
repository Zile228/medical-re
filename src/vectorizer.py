import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch
import os
import joblib
import sys
import scipy.sparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from utils import BASE_DIR, DATA_DIR, MODEL_DIR, simple_tokenizer, BERT_MODEL_NAME

PROCESSED_TRAIN_PATH = os.path.join(DATA_DIR, 'processed', 'train_data.csv')
PROCESSED_TEST_PATH = os.path.join(DATA_DIR, 'processed', 'test_data.csv')

def load_bert_model():
    print(f"Đang tải BERT: {BERT_MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME, use_safetensors=True)

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
        tokenizer.add_special_tokens(special_tokens)
        bert_model.resize_token_embeddings(len(tokenizer))
        return tokenizer, bert_model
    except Exception as e:
        print(f"Lỗi tải BERT: {e}")
        return None, None

def sentence_to_vector_w2v(sentence, model, vector_size):
    """Tính trung bình vector từ cho câu (Word2Vec)"""
    tokens = sentence.lower().split()
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    return np.mean(model.wv[valid_tokens], axis=0)

def get_bert_embeddings(sentences, model, tokenizer, device='cpu'):
    """Lấy embedding [CLS] từ BERT cho danh sách câu"""
    if model is None: return None
    print(f"Đang tạo BERT embeddings ({len(sentences)} mẫu)...")
    embeddings = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, sentence in enumerate(sentences):
            if i % 100 == 0: print(f"Đã xử lý {i}/{len(sentences)}")
            inputs = tokenizer(sentence, return_tensors="pt", max_length=256, truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls_embedding)
    return np.array(embeddings)

def process_all_vectorizers():
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        print("Lỗi: Chưa có file CSV train/test.")
        return

    df_train = pd.read_csv(PROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PROCESSED_TEST_PATH)
    train_marked = df_train['marked_sentence'].fillna("").tolist()
    test_marked = df_test['marked_sentence'].fillna("").tolist()

    # 1. BoW
    print("Xử lý Bag of Words (BoW)")
    bow_vec = CountVectorizer(tokenizer=simple_tokenizer, max_features=5000)
    bow_vec.fit(train_marked)
    joblib.dump(bow_vec, os.path.join(MODEL_DIR, 'bow_vectorizer.joblib'))

    scipy.sparse.save_npz(os.path.join(MODEL_DIR, 'X_train_bow.npz'), bow_vec.transform(train_marked))
    scipy.sparse.save_npz(os.path.join(MODEL_DIR, 'X_test_bow.npz'), bow_vec.transform(test_marked))

    # 2. TF-IDF
    print("Xử lý TF-IDF")
    tfidf_vec = TfidfVectorizer(tokenizer=simple_tokenizer, max_features=5000)
    tfidf_vec.fit(train_marked)
    joblib.dump(tfidf_vec, os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))

    scipy.sparse.save_npz(os.path.join(MODEL_DIR, 'X_train_tfidf.npz'), tfidf_vec.transform(train_marked))
    scipy.sparse.save_npz(os.path.join(MODEL_DIR, 'X_test_tfidf.npz'), tfidf_vec.transform(test_marked))

    # 3. Word2Vec
    print("Xử lý Word2Vec")
    tokenized_sentences = [s.lower().split() for s in (train_marked + test_marked)]
    w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=1, sg=1)
    w2v_model.save(os.path.join(MODEL_DIR, 'word2vec.model'))

    X_train_w2v = np.array([sentence_to_vector_w2v(s, w2v_model, 100) for s in train_marked])
    X_test_w2v = np.array([sentence_to_vector_w2v(s, w2v_model, 100) for s in test_marked])
    np.save(os.path.join(MODEL_DIR, 'X_train_w2v.npy'), X_train_w2v)
    np.save(os.path.join(MODEL_DIR, 'X_test_w2v.npy'), X_test_w2v)

    # 4. BERT
    tokenizer, bert_model = load_bert_model()
    if bert_model:
        print("Xử lý BERT")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X_train_bert = get_bert_embeddings(train_marked, bert_model, tokenizer, device)
        X_test_bert = get_bert_embeddings(test_marked, bert_model, tokenizer, device)
        
        np.save(os.path.join(MODEL_DIR, 'X_train_bert.npy'), X_train_bert)
        np.save(os.path.join(MODEL_DIR, 'X_test_bert.npy'), X_test_bert)

if __name__ == "__main__":
    process_all_vectorizers()
    print("--- HOÀN TẤT VECTOR HÓA ---")