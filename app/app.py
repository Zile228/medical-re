import gradio as gr
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import re
import sys
import os
import joblib
import torch
import itertools
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Thêm src vào path để import module
sys.path.append(str(SRC_DIR))

# IMPORT MODULE NỘI BỘ
try:
    from utils import add_markers, load_re_model_resources
    # Import các hàm cốt lõi từ evaluate
    from evaluate import (
        load_ner_model_unified, 
        predict_ner_general, 
        predict_relation_hybrid
    )
except ImportError as e:
    print(f"Lỗi Import: {e}")
    print("Bạn cần đảm bảo file src/evaluate.py và src/utils.py tồn tại và đúng cấu trúc.")
    sys.exit(1)

# CẤU HÌNH MODEL
NER_MODEL_PATH = MODEL_DIR / "ner_spacy_model" 
RE_MODEL_FILENAME = "MLPDeepLearning_bert_hybrid.pkl" 
RE_MODEL_NAME_PURE = "MLPDeepLearning"
VEC_NAME = "bert"

# KHO TÀI NGUYÊN TOÀN CỤC
RESOURCES = {
    "ner_model": None, "ner_tokenizer": None, "ner_device": None,
    "re_clf": None, "re_le": None, "re_scaler": None,
    "bert_tokenizer": None, "bert_model": None, "re_device": None
}

GRAPH_STORE = {"G": nx.DiGraph(), "knowledge_base": {}}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BẢNG MÀU CHO CÁC THỰC THỂ (Palette Y Tế Hiện Đại)
TYPE_COLORS = {
    "Bệnh": "#ef4444",          # Đỏ tươi
    "Triệu chứng": "#3b82f6",   # Xanh dương
    "Nguyên nhân": "#f59e0b",   # Vàng cam
    "Chẩn đoán": "#10b981",     # Xanh lá
    "Điều trị": "#8b5cf6",      # Tím
    "Unknown": "#9ca3af"        # Xám
}


def initialize_system():
    print(f" ĐANG KHỞI TẠO HỆ THỐNG (DEVICE: {DEVICE})")
    
    # 1. Load NER
    print(f"    Load NER từ: {NER_MODEL_PATH}")
    ner_model, ner_tok, ner_dev = load_ner_model_unified(str(NER_MODEL_PATH))
    RESOURCES["ner_model"] = ner_model
    RESOURCES["ner_tokenizer"] = ner_tok
    RESOURCES["ner_device"] = ner_dev

    # 2. Load RE Classifier & Label Encoder
    print(f"   Load RE Model: {RE_MODEL_FILENAME}")
    try:
        clf_path = MODEL_DIR / RE_MODEL_FILENAME
        le_path = MODEL_DIR / "label_encoder_hybrid.pkl"
        scaler_path = MODEL_DIR / "scaler_bert_hybrid.pkl"
        
        RESOURCES["re_clf"] = joblib.load(clf_path)
        RESOURCES["re_le"] = joblib.load(le_path)
        RESOURCES["re_scaler"] = joblib.load(scaler_path) if scaler_path.exists() else None
        
    except Exception as e:
        print(f" Lỗi load RE components: {e}")
        return

    # 3. Load PhoBERT Vectorizer
    if VEC_NAME == 'bert':
        print("    Load PhoBERT Embeddings...")
        bert_re_name = "vinai/phobert-base"
        tokenizer = AutoTokenizer.from_pretrained(bert_re_name)
        model = AutoModel.from_pretrained(bert_re_name)
        
        # Thêm token đặc biệt (Khớp với quy trình huấn luyện)
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
        model.to(DEVICE)
        model.eval()
        
        RESOURCES["bert_tokenizer"] = tokenizer
        RESOURCES["bert_model"] = model
        RESOURCES["re_device"] = torch.device(DEVICE)

    print(" HỆ THỐNG ĐÃ SẴN SÀNG PHỤC VỤ!")

# Gọi khởi tạo ngay
initialize_system()


def segment_text(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\n)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def process_text_pipeline(input_text):
    if RESOURCES["ner_model"] is None:
        return nx.DiGraph(), pd.DataFrame(), []

    sentences = segment_text(input_text)
    all_relations = []
    knowledge_base = {} 
    G = nx.DiGraph()
    
    # Lấy resources ra biến cục bộ
    ner_model = RESOURCES["ner_model"]
    ner_tok = RESOURCES["ner_tokenizer"]
    ner_dev = RESOURCES["ner_device"]
    
    re_clf = RESOURCES["re_clf"]
    re_le = RESOURCES["re_le"]
    re_scaler = RESOURCES["re_scaler"]
    bert_tok = RESOURCES["bert_tokenizer"]
    bert_mdl = RESOURCES["bert_model"]
    re_dev = RESOURCES["re_device"]

    print(f"\n Đang xử lý: {input_text[:50]}...")

    for sent in sentences:
        # 1. NER Prediction
        ents = predict_ner_general(sent, ner_model, ner_dev, ner_tok)
        
        if not ents: continue
        
        # Build Nodes (SỬA LẠI ĐOẠN NÀY)
        for e in ents:
            # ID dùng chữ thường để gộp trùng lặp
            node_id = e['text'].lower().strip()
            # Label dùng text gốc để hiển thị đẹp
            display_label = e['text'].strip()
            
            # Nếu node chưa có, hoặc node cũ đang hiển thị ngắn hơn node mới 
            # (ví dụ cũ là "tiểu đường", mới là "Bệnh tiểu đường" -> cập nhật lại label cho đẹp)
            if not G.has_node(node_id):
                G.add_node(node_id, type=e['label'], label=display_label)
            else:
                current_label = G.nodes[node_id].get('label', '')
                if len(display_label) > len(current_label):
                    G.nodes[node_id]['label'] = display_label

            # Xây dựng Knowledge Base (Logic cũ vẫn đúng vì bạn đã dùng key lower)
            if e['label'] == 'Bệnh':
                if node_id not in knowledge_base:
                    knowledge_base[node_id] = {
                        "display_name": display_label, 
                        "causes": set(), "symptoms": set(), 
                        "treatments": set(), "diagnosis": set()
                    }
                else:
                    # Cập nhật display name nếu tìm thấy tên dài/đẹp hơn
                    if len(display_label) > len(knowledge_base[node_id]["display_name"]):
                        knowledge_base[node_id]["display_name"] = display_label

        # 2. RE Prediction
        # Lọc trùng lặp thực thể trước khi chạy RE (như đã làm ở evaluate.py)
        # Để tránh chạy RE cho ("Tiểu đường", "ABC") và ("tiểu đường", "ABC")
        unique_ents = []
        seen = set()
        for e in ents:
            if e['text'].lower() not in seen:
                seen.add(e['text'].lower())
                unique_ents.append(e)

        if len(unique_ents) >= 2:
            for subj, obj in itertools.permutations(unique_ents, 2):
                pred_label = predict_relation_hybrid(
                    sent, subj, obj, 
                    vec_model=None, 
                    clf_model=re_clf, scaler=re_scaler, le=re_le, 
                    vec_name=VEC_NAME, device=re_dev, 
                    bert_tokenizer=bert_tok, bert_embed_model=bert_mdl
                )

                if pred_label != "No_relation":
                    # Chuẩn hóa ID để tạo cạnh
                    s_id = subj['text'].lower().strip()
                    o_id = obj['text'].lower().strip()
                    
                    # Lấy label đẹp hiện tại từ graph để lưu vào bảng
                    s_label = G.nodes[s_id]['label'] if G.has_node(s_id) else subj['text']
                    o_label = G.nodes[o_id]['label'] if G.has_node(o_id) else obj['text']

                    print(f"    Tìm thấy: {s_label} --[{pred_label}]--> {o_label}")
                    
                    all_relations.append({
                        "Chủ thể": s_label, "Loại chủ thể": subj['label'],
                        "Quan hệ": pred_label,
                        "Đối tượng": o_label, "Loại đối tượng": obj['label']
                    })
                    
                    # Tạo cạnh dựa trên ID chữ thường
                    G.add_edge(s_id, o_id, relation=pred_label)

                    # Update Knowledge Base
                    if pred_label == 'Có_triệu_chứng' and subj['label'] == 'Bệnh' and s_id in knowledge_base:
                        knowledge_base[s_id]['symptoms'].add(o_label)
                    elif pred_label == 'Gây_ra':
                        if obj['label'] == 'Bệnh' and o_id in knowledge_base: 
                            knowledge_base[o_id]['causes'].add(s_label)
                        elif subj['label'] == 'Bệnh' and s_id in knowledge_base: 
                            knowledge_base[s_id]['symptoms'].add(f"Biến chứng: {o_label}")
                    elif pred_label == 'Điều_trị_bằng' and subj['label'] == 'Bệnh' and s_id in knowledge_base:
                        knowledge_base[s_id]['treatments'].add(o_label)
                    elif pred_label == 'Chẩn_đoán_bằng' and subj['label'] == 'Bệnh' and s_id in knowledge_base:
                        knowledge_base[s_id]['diagnosis'].add(o_label)

    GRAPH_STORE["G"] = G
    GRAPH_STORE["knowledge_base"] = knowledge_base
    
    df = pd.DataFrame(all_relations)
    if not df.empty: 
        df = df[['Chủ thể', 'Quan hệ', 'Đối tượng']]
    else: 
        df = pd.DataFrame(columns=['Chủ thể', 'Quan hệ', 'Đối tượng'])
    
    return G, df, list(knowledge_base.keys())


def draw_graph(G):
    if G.number_of_nodes() == 0: 
        fig = go.Figure()
        fig.update_layout(title="Chưa có dữ liệu để hiển thị")
        return fig
        
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    # Vẽ cạnh (Edges)
    edge_x, edge_y, edge_txt = [], [], []
    for e in G.edges(data=True):
        x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_txt.append(e[2].get('relation', ''))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, 
        line=dict(width=1, color='#94a3b8'), 
        hoverinfo='text', 
        text=edge_txt, 
        mode='lines'
    )
    
    # Vẽ node (Nodes)
    node_x, node_y, node_txt, node_col, hover_text = [], [], [], [], []
    for n in G.nodes(data=True):
        # n[0] là ID (chữ thường), n[1] là attributes
        node_id = n[0]
        attrs = n[1]
        
        if node_id not in pos: continue # Phòng hờ lỗi
        
        x, y = pos[node_id]
        node_x.append(x); node_y.append(y)
        
        typ = attrs.get('type', 'Unknown')
        # Lấy label hiển thị (chữ hoa đẹp) thay vì lấy node_id
        display_text = attrs.get('label', node_id) 
        
        node_txt.append(display_text)
        node_col.append(TYPE_COLORS.get(typ, '#999'))
        hover_text.append(f"<b>{display_text}</b><br>Loại: {typ}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, 
        mode='markers+text', 
        text=node_txt, 
        textposition="top center", 
        hoverinfo="text",
        hovertext=hover_text,
        marker=dict(
            color=node_col, 
            size=25,
            line=dict(width=2, color='white') 
        ),
        textfont=dict(size=11, color='#333')
    )
    
    layout = go.Layout(
        showlegend=False, 
        margin=dict(b=20,l=20,r=20,t=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)', 
        hovermode='closest'
    )
    return go.Figure(data=[edge_trace, node_trace], layout=layout)

def answer_question(disease_key, question_type):
    kb = GRAPH_STORE["knowledge_base"]
    if not disease_key or disease_key not in kb: return " Vui lòng chọn một bệnh để tra cứu."
    
    data = kb[disease_key]
    mapping = {
        "Nguyên nhân": data['causes'],
        "Triệu chứng": data['symptoms'],
        "Điều trị": data['treatments'],
        "Chẩn đoán": data['diagnosis']
    }
    
    items = mapping.get(question_type, [])
    
    if not items: 
        return f"ℹ Không tìm thấy thông tin về **{question_type}** của bệnh này trong văn bản."
    
    # Format câu trả lời đẹp
    result = f"### {question_type} của {data['display_name']}\n"
    result += "\n".join([f"- {i}" for i in items])
    return result


def run_app():
    # Tùy chỉnh CSS
    custom_css = """
    .gradio-container {font-family: 'Segoe UI', sans-serif;}
    h1 {color: #0f766e; text-align: center; margin-bottom: 0;}
    .description {text-align: center; color: #64748b; font-size: 1.1em; margin-bottom: 20px;}
    """
    
    theme = gr.themes.Soft(
        primary_hue="teal", 
        secondary_hue="blue",
        radius_size="md"
    )

    with gr.Blocks(theme=theme, css=custom_css, title="Medical Knowledge Graph") as demo:
        
        # HEADER
        gr.Markdown("# HỆ THỐNG TRÍCH XUẤT TRI THỨC Y KHOA")
        gr.Markdown("*Công nghệ: BERT Hybrid Model • Named Entity Recognition • Relation Extraction*", elem_classes=["description"])
        
        with gr.Row():
            # CỘT TRÁI: INPUT & ĐIỀU KHIỂN
            with gr.Column(scale=4, min_width=300):
                gr.Markdown("###  Nhập liệu")
                inp = gr.Textbox(
                    label="Văn bản y khoa", 
                    placeholder="Nhập đoạn văn mô tả bệnh học...", 
                    lines=8, 
                    value="Bệnh tiểu đường type 2 thường do béo phì và lười vận động gây ra. Triệu chứng của bệnh tiểu đường type 2 gồm khát nước, mệt mỏi và sụt cân bất thường. Điều trị bệnh tiểu đường type 2 bằng thuốc Metformin và thay đổi lối sống."
                )
                btn = gr.Button(" Phân tích Tri thức", variant="primary", size="lg")
                
                # Phần Hỏi đáp nằm bên dưới Input cho gọn
                gr.Markdown("###  Tra cứu nhanh")
                with gr.Group():
                    dd = gr.Dropdown(label="Chọn bệnh đã trích xuất", interactive=True, choices=[])
                    rad = gr.Radio(
                        ["Nguyên nhân", "Triệu chứng", "Điều trị", "Chẩn đoán"], 
                        label="Bạn muốn biết gì?",
                        value="Triệu chứng"
                    )
                    btn_a = gr.Button(" Hỏi Hệ Thống")
                out_ans = gr.Markdown(label="Câu trả lời")

            # CỘT PHẢI: KẾT QUẢ VISUALIZATION
            with gr.Column(scale=6):
                gr.Markdown("###  Kết quả Phân tích")
                with gr.Tabs():
                    with gr.TabItem(" Đồ thị Tri thức"):
                        out_plot = gr.Plot(label="Knowledge Graph")
                        # Chú thích màu sắc
                        legend_html = """
                        <div style="display: flex; gap: 10px; font-size: 12px; justify-content: center; margin-top: 5px;">
                            <span style="color:#ef4444">● Bệnh</span>
                            <span style="color:#3b82f6">● Triệu chứng</span>
                            <span style="color:#f59e0b">● Nguyên nhân</span>
                            <span style="color:#10b981">● Chẩn đoán</span>
                            <span style="color:#8b5cf6">● Điều trị</span>
                        </div>
                        """
                        gr.HTML(legend_html)
                        
                    with gr.TabItem(" Bảng dữ liệu quan hệ"):
                        out_df = gr.Dataframe(
                            headers=["Chủ thể", "Quan hệ", "Đối tượng"],
                            datatype=["str", "str", "str"],
                            interactive=False
                        )

        # XỬ LÝ SỰ KIỆN
        def on_process(text):
            G, df, diseases = process_text_pipeline(text)
            kb = GRAPH_STORE["knowledge_base"]
            
            # Tạo list choices cho Dropdown
            choices = []
            if diseases:
                choices = [(kb[k]['display_name'], k) for k in diseases]
            
            # Mặc định chọn bệnh đầu tiên nếu có
            first_val = choices[0][1] if choices else None
            
            return df, draw_graph(G), gr.Dropdown(choices=choices, value=first_val, interactive=True)

        btn.click(fn=on_process, inputs=[inp], outputs=[out_df, out_plot, dd])
        btn_a.click(fn=answer_question, inputs=[dd, rad], outputs=[out_ans])

    print("Đang khởi động trình duyệt...")
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    run_app()