# Vietnamese Medical Relation Extraction (VnMedical-RE)

Đồ án trích xuất quan hệ thực thể (Relation Extraction) trong văn bản y tế tiếng Việt, tập trung vào các mối quan hệ giữa **Bệnh**, **Triệu chứng**, **Nguyên nhân**, **Chẩn đoán** và **Điều trị**.

## Tính năng chính
- **NER (Named Entity Recognition):** Nhận diện thực thể y tế.
- **RE (Relation Extraction):** Phân loại quan hệ (Gây_ra, Điều_trị_bằng, Có_triệu_chứng, Chẩn_đoán_bằng).
- **Application:** Web tra cứu và tóm tắt hồ sơ bệnh án tự động.

## Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.8+
- **Model:** Logistic Regression, SVM, Random Forest, MLP (Deep Learning).
- **Embeddings:** BoW, TF-IDF, Word2Vec, ViHealthBERT.
- **App:** Gradio.

## Cấu trúc repo
```text
vn-medical-re/
│
├── data/                   # Chứa dữ liệu
│   ├── raw/                # Dữ liệu thô (file json từ Label Studio, file crawl)
│   └── processed/          # Dữ liệu đã xử lý (file .csv chạy từ code PURE)
│
├── notebooks/              # Chứa Jupyter Notebook để chạy thử nghiệm, EDA
│   ├── 01_eda.ipynb        # Phân tích dữ liệu
│   └── 02_experiment.ipynb # Thử nghiệm các model
│
├──src/
│   ├── utils.py               # File cấu hình gốc: chứa đường dẫn (Path), hàm xử lý text (add_markers), và bộ luật (Rule-based).
│   ├── preprocessing.py       # Tiền xử lý: Đọc JSON Label Studio -> Chia tập Train/Test -> Tạo file CSV chứa cặp thực thể.
│   ├── vectorizer.py          # Vector hóa: Chuyển đổi văn bản sang số (BoW, TF-IDF, Word2Vec, BERT Embedding).
│   ├── train_ner.py           # Huấn luyện NER (Cách 1): Dùng mô hình Deep Learning (PhoBERT + HuggingFace).
│   ├── train_ner_spacy.py     # Huấn luyện NER (Cách 2): Dùng thư viện SpaCy 
│   ├── hybrid_labeling.py     # Gán nhãn tự động (Hybrid): Dùng model đã train + Rule để gán nhãn cho dữ liệu mới (Silver data).
│   ├── train_re.py            # Huấn luyện RE: Train các model phân loại quan hệ (SVM, Random Forest, MLP...) dùng vector đã tạo.
│   └── evaluate.py            # Đánh giá & Dự đoán: Tính điểm F1-score, so sánh các model và chứa hàm inference pipeline (demo).
│
├── models/                 # Chứa model đã train xong
│   ├── ...
│   ├── ...
│   └── ...     
│
├── app/                    # Code Web App (Gradio)
│   └── app.py             # File chạy web
│
├── .gitignore              
├── requirements.txt        # Danh sách thư viện cần cài
└── README.md               # Giới thiệu đồ án
```


## Hướng dẫn cài đặt

1. **Clone repository:**
   ```bash
   git clone https://github.com/Zile228/vn-medical-re.git
   cd vn-medical-re
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy ứng dụng Demo:**
   ```bash
   streamlit run app/app.py
   ```

## 📂 Dữ liệu
Dữ liệu được thu thập từ Vinmec, HelloBacsi và gán nhãn thủ công bằng Label Studio.

## 👥 Thành viên
- Thái Hoài An - 31231025020
- Nguyễn Thị Thùy Dương - 31231022904
- Nguyễn Duy Tân - 31231023384
- Lê Vy - 31231022128
```
