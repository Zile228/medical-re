# Vietnamese Medical Relation Extraction (VnMedical-RE)

Đồ án trích xuất quan hệ thực thể (Relation Extraction) trong văn bản y tế tiếng Việt, tập trung vào các mối quan hệ giữa **Bệnh**, **Triệu chứng**, **Nguyên nhân**, **Chẩn đoán** và **Điều trị**.

## Tính năng chính
- **NER (Named Entity Recognition):** Nhận diện thực thể y tế.
- **RE (Relation Extraction):** Phân loại quan hệ (Gây_ra, Điều_trị_bằng, Có_triệu_chứng, Chẩn_đoán_bằng).
- **Application:** Web tra cứu và tóm tắt hồ sơ bệnh án tự động.

## Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.8+
- **Model:** Logistic Regression, SVM, Random Forest, MLP (Deep Learning).
- **Embeddings:** TF-IDF, Word2Vec, ViHealthBERT.
- **App:** Streamlit.

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
├── src/                    # Source code chính (file .py)
│   ├── preprocessing.py    # Code xử lý JSON -> CSV, tạo marker [S], [O]
│   ├── vectorizer.py       # Code tạo vector (TF-IDF, W2V, BERT)
│   └── train.py            # Code huấn luyện model và lưu file .pkl
│
├── models/                 # Chứa model đã train xong
│   ├── svm_model.pkl
│   ├── logreg_model.pkl
│   └── vectorizer.pkl      
│
├── app/                    # Code Web App (Streamlit)
│   └── main.py             # File chạy web
│
├── .gitignore              
├── requirements.txt        # Danh sách thư viện cần cài
└── README.md               # Giới thiệu đồ án
```


## Hướng dẫn cài đặt

1. **Clone repository:**
   ```bash
   git clone https://github.com/username-cua-ban/vn-medical-re.git
   cd vn-medical-re
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy ứng dụng Demo:**
   ```bash
   streamlit run app/main.py
   ```

## 📂 Dữ liệu
Dữ liệu được thu thập từ Vinmec, HelloBacsi và gán nhãn thủ công bằng Label Studio.

## 👥 Thành viên
- Thái Hoài An - 31231025020
- Nguyễn Thị Thùy Dương - 31231022904
- Nguyễn Duy Tân - 31231023384
- Lê Vy - 31231022128
```
