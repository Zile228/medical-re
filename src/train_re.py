import pandas as pd
import numpy as np
import os
import joblib
import scipy.sparse
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm 
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Import utils
sys.path.append(str(Path(__file__).resolve().parent))
from utils import DATA_DIR, MODEL_DIR, simple_tokenizer

PROCESSED_TRAIN_PATH = os.path.join(DATA_DIR, 'processed', 'train_data.csv')
PROCESSED_TEST_PATH = os.path.join(DATA_DIR, 'processed', 'test_data.csv')
SILVER_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'silver_data.csv')
RESULT_DIR = os.path.join(Path(__file__).resolve().parent.parent, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# Cấu hình BERT
BERT_MODEL_NAME = "vinai/phobert-base"

def load_labels(use_silver=False):
    print("Đang tải dữ liệu và nhãn...")
    df_train = pd.read_csv(PROCESSED_TRAIN_PATH)
    df_test = pd.read_csv(PROCESSED_TEST_PATH)
    
    # --- LOGIC GỘP DỮ LIỆU SILVER (HYBRID) ---
    if use_silver:
        if os.path.exists(SILVER_DATA_PATH):
            print(f"CHẾ ĐỘ HYBRID: Đang gộp dữ liệu Silver từ {SILVER_DATA_PATH}")
            df_silver = pd.read_csv(SILVER_DATA_PATH)
            
            # Đảm bảo chỉ lấy các cột cần thiết
            cols = ['marked_sentence', 'relation_label']
            if all(col in df_silver.columns for col in cols):
                # Chỉ lấy những dòng mà model cũ chưa thấy (dựa vào source nếu có, hoặc gộp tất)
                df_train = pd.concat([df_train[cols], df_silver[cols]], ignore_index=True)
                print(f"   -> Tổng mẫu Train sau khi gộp: {len(df_train)}")
            else:
                print("   File Silver thiếu cột cần thiết. Bỏ qua.")
        else:
            print(f"   Không tìm thấy file Silver tại {SILVER_DATA_PATH}. Chạy như Supervised.")
    else:
        print("CHẾ ĐỘ SUPERVISED: Chỉ dùng dữ liệu gốc.")
    # -----------------------------------------

    df_train['relation_label'] = df_train['relation_label'].fillna("No_relation")
    df_test['relation_label'] = df_test['relation_label'].fillna("No_relation")

    le = LabelEncoder()
    all_labels = pd.concat([df_train['relation_label'], df_test['relation_label']])
    le.fit(all_labels)
    
    y_train = le.transform(df_train['relation_label'])
    y_test = le.transform(df_test['relation_label'])
    
    # Lấy text để vectorize lại
    train_sentences = df_train['marked_sentence'].fillna("").tolist()
    
    no_rel_idx = -1
    if "No_relation" in le.classes_:
        no_rel_idx = le.transform(["No_relation"])[0]
    
    suffix = "_hybrid" if use_silver else ""
    joblib.dump(le, os.path.join(MODEL_DIR, f'label_encoder{suffix}.pkl'))
    
    return y_train, y_test, no_rel_idx, le, train_sentences

# --- CÁC HÀM RE-VECTORIZE CHO HYBRID ---

def get_w2v_vectors(sentences):
    """Tính vector trung bình W2V cho danh sách câu"""
    print("   -> Đang tính toán lại Word2Vec cho dữ liệu mới...")
    w2v_path = os.path.join(MODEL_DIR, 'word2vec.model')
    if not os.path.exists(w2v_path):
        print("      Lỗi: Không tìm thấy model W2V gốc.")
        return None
        
    w2v_model = Word2Vec.load(w2v_path)
    vector_size = w2v_model.vector_size
    
    vectors = []
    for s in sentences:
        tokens = s.lower().split()
        valid_tokens = [t for t in tokens if t in w2v_model.wv]
        if valid_tokens:
            vectors.append(np.mean(w2v_model.wv[valid_tokens], axis=0))
        else:
            vectors.append(np.zeros(vector_size))
            
    return np.array(vectors)

def get_bert_vectors(sentences):
    """Tính vector BERT [CLS] cho danh sách câu (có Batching)"""
    print("   -> Đang tính toán lại BERT cho dữ liệu mới (Việc này có thể lâu)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        
        # Add special tokens để khớp với lúc train vectorizer gốc
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
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        model.eval()
        
        all_embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="BERT Embedding"):
            batch_texts = sentences[i : i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", max_length=256, truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Lấy CLS token
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
        return np.vstack(all_embeddings)
        
    except Exception as e:
        print(f"      Lỗi BERT Vectorization: {e}")
        return None

def get_vectors_dynamic(vector_type, train_sentences, use_silver=False):
    print(f"Đang chuẩn bị vector: {vector_type}...")
    try:
        X_test = None
        X_train = None
        need_scaling = False

        # 1. Load X_test (Luôn cố định từ file đã process ban đầu để so sánh công bằng)
        if vector_type in ['bow', 'tfidf']:
            X_test = scipy.sparse.load_npz(os.path.join(MODEL_DIR, f'X_test_{vector_type}.npz'))
            need_scaling = False
        else:
            X_test = np.load(os.path.join(MODEL_DIR, f'X_test_{vector_type}.npy'))
            need_scaling = True

        # 2. Load X_train
        if not use_silver:
            # SUPERVISED: Load file có sẵn cho nhanh
            if vector_type in ['bow', 'tfidf']:
                X_train = scipy.sparse.load_npz(os.path.join(MODEL_DIR, f'X_train_{vector_type}.npz'))
            else:
                X_train = np.load(os.path.join(MODEL_DIR, f'X_train_{vector_type}.npy'))
        else:
            # HYBRID: Phải tính toán lại vector cho tập train mới (gồm cả silver)
            if vector_type == 'bow':
                vec = joblib.load(os.path.join(MODEL_DIR, 'bow_vectorizer.joblib'))
                X_train = vec.transform(train_sentences)
            elif vector_type == 'tfidf':
                vec = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
                X_train = vec.transform(train_sentences)
            elif vector_type == 'w2v':
                X_train = get_w2v_vectors(train_sentences)
            elif vector_type == 'bert':
                X_train = get_bert_vectors(train_sentences)

        if X_train is None:
            raise ValueError("X_train is None")

        return X_train, X_test, need_scaling

    except Exception as e:
        print(f"   Lỗi xử lý vector {vector_type}: {e}")
        return None, None, False

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight=None, random_state=42, max_depth=15), 
        'MLP Deep Learning': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, early_stopping=True, random_state=42)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_silver', action='store_true', help='Sử dụng dữ liệu Silver cho Hybrid Learning')
    args = parser.parse_args()

    # Load dữ liệu
    y_train_full, y_test, no_rel_idx, le, train_sentences = load_labels(args.use_silver)
    
    # Cấu hình Sampling
    unique, counts = np.unique(y_train_full, return_counts=True)
    stats = dict(zip(unique, counts))
    print("Phân bố nhãn Train:", stats)
    
    if args.use_silver:
        MAJORITY_TARGET = 300 
        MINORITY_TARGET = 100 
    else:
        MAJORITY_TARGET = 300
        MINORITY_TARGET = 100 
    
    print(f"Mục tiêu Sampling: No_relation={MAJORITY_TARGET}, Others={MINORITY_TARGET}")
    
    under_strategy = {}
    if no_rel_idx != -1 and stats.get(no_rel_idx, 0) > MAJORITY_TARGET:
        under_strategy[no_rel_idx] = MAJORITY_TARGET
    
    over_strategy = {}
    for label in stats.keys():
        if label != no_rel_idx:
            current_count = stats.get(label, 0)
            over_strategy[label] = max(current_count, MINORITY_TARGET)

    vector_types = ['bow', 'tfidf', 'w2v', 'bert'] 
    models_dict = get_models()
    results = [] 
    
    best_f1 = 0.0
    best_info = {}
    
    suffix = "_hybrid" if args.use_silver else ""

    print(f"\nBẮT ĐẦU HUẤN LUYỆN RE (Mode: {'HYBRID' if args.use_silver else 'SUPERVISED'})")
    
    for vec_name in vector_types:
        X_train, X_test, need_scaling = get_vectors_dynamic(vec_name, train_sentences, args.use_silver)
        
        if X_train is None: continue
        
        print(f"\n>> Vector: {vec_name}")

        try:
            steps = []
            if under_strategy:
                steps.append(('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)))
            steps.append(('over', RandomOverSampler(sampling_strategy=over_strategy, random_state=42)))
            pipeline = Pipeline(steps)
            X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train_full)
        except Exception as e:
            print(f"   Lỗi sampling: {e}. Dùng dữ liệu gốc.")
            X_train_res, y_train_res = X_train, y_train_full

        if need_scaling:
            print("   Đang scaling...")
            scaler = StandardScaler()
            if scipy.sparse.issparse(X_train_res):
                X_train_res = X_train_res.toarray()
                X_test_dense = X_test.toarray() if scipy.sparse.issparse(X_test) else X_test
            else:
                X_test_dense = X_test
            
            X_train_final = scaler.fit_transform(X_train_res)
            X_test_final = scaler.transform(X_test_dense)
            
            joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler_{vec_name}{suffix}.pkl'))
        else:
            X_train_final = X_train_res
            X_test_final = X_test

        for model_name, model in models_dict.items():
            print(f"   Training {model_name}...", end=" ")
            model.fit(X_train_final, y_train_res)
            
            safe_name = model_name.replace(" ", "").replace("(", "").replace(")", "")
            joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}_{vec_name}{suffix}.pkl"))
            
            y_pred = model.predict(X_test_final)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            print(f"-> F1={f1:.4f}")
            results.append({'Vectorizer': vec_name, 'Model': model_name, 'F1': f1, 'Acc': acc})
            
            if f1 > best_f1:
                best_f1 = f1
                best_info = {
                    'name': f"{model_name} ({vec_name})",
                    'report': classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
    
    pd.DataFrame(results).to_csv(os.path.join(RESULT_DIR, f'model_comparison{suffix}.csv'), index=False)

    print("\n MODEL TỐT NHẤT TRONG LẦN CHẠY NÀY")
    if best_info:
        print(best_info['name'])
        print(best_info['report'])
        print("Confusion Matrix:")
        print(best_info['confusion_matrix'])
    else:
        print("Không có model nào chạy thành công.")

if __name__ == "__main__":
    main()