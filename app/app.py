import gradio as gr
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import sys
import re
from pathlib import Path

# CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

# Import Module từ evaluate.py
try:
    from evaluate import MedicalKnowledgePipeline, find_best_model_from_results
except ImportError as e:
    print(f"Lỗi Import: {e}")
    sys.exit(1)

NER_MODEL_PATH = MODEL_DIR / "ner_spacy_model"
GRAPH_STORE = {"G": nx.DiGraph(), "knowledge_base": {}}

# BẢNG MÀU
TYPE_COLORS = {
    "Bệnh": "#ef4444", "Triệu chứng": "#3b82f6", "Nguyên nhân": "#f59e0b",
    "Chẩn đoán": "#10b981", "Điều trị": "#8b5cf6", "Unknown": "#9ca3af"
}

def initialize_pipeline():
    """Khởi tạo Pipeline dựa trên model tốt nhất tìm thấy."""
    print("KHỞI TẠO HỆ THỐNG...")
    
    # 1. Tìm model tốt nhất (Ưu tiên Hybrid -> Supervised)
    best_model, best_vec = find_best_model_from_results(use_silver=True)
    is_silver = True
    
    if not best_model:
        print("Không tìm thấy model Hybrid, chuyển sang Supervised...")
        best_model, best_vec = find_best_model_from_results(use_silver=False)
        is_silver = False
        
    if not best_model:
        print("CRITICAL: Không tìm thấy bất kỳ model nào. Vui lòng chạy train_re.py trước.")
        return None

    print(f"Loading Pipeline: NER + RE[{best_model} - {best_vec}] (Hybrid={is_silver})")
    
    try:
        # Khởi tạo class từ evaluate.py
        pipeline = MedicalKnowledgePipeline(
            ner_model_path=str(NER_MODEL_PATH),
            re_model_name=best_model,
            vec_name=best_vec,
            use_silver=is_silver
        )
        return pipeline
    except Exception as e:
        print(f"Lỗi khởi tạo Pipeline: {e}")
        return None

# Khởi tạo biến toàn cục
PIPELINE = initialize_pipeline()

def segment_text(text):
    """Tách câu đơn giản."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\n)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def process_text_ui(input_text):
    if PIPELINE is None:
        return nx.DiGraph(), pd.DataFrame(), []

    sentences = segment_text(input_text)
    all_relations = []
    knowledge_base = {} 
    G = nx.DiGraph()

    for sent in sentences:
        # Gọi hàm process_text của Pipeline (Trả về ents và relations)
        ents, rels = PIPELINE.process_text(sent)
        
        # 1. Xây dựng Nodes
        for e in ents:
            node_id = e['text'].lower().strip()
            display_label = e['text'].strip()
            
            if not G.has_node(node_id):
                G.add_node(node_id, type=e['label'], label=display_label)
            else:
                # Cập nhật label hiển thị nếu tìm thấy tên đầy đủ hơn
                current_label = G.nodes[node_id].get('label', '')
                if len(display_label) > len(current_label):
                    G.nodes[node_id]['label'] = display_label

            # Xây dựng KB sơ khởi cho Bệnh
            if e['label'] == 'Bệnh':
                if node_id not in knowledge_base:
                    knowledge_base[node_id] = {
                        "display_name": display_label, 
                        "causes": set(), "symptoms": set(), 
                        "treatments": set(), "diagnosis": set()
                    }

        # 2. Xây dựng Edges từ kết quả RE
        for r in rels:
            s_id = r['subject'].lower().strip()
            o_id = r['object'].lower().strip()
            pred_label = r['relation']
            
            # Lấy label đẹp từ Graph node
            s_label = G.nodes[s_id]['label'] if G.has_node(s_id) else r['subject']
            o_label = G.nodes[o_id]['label'] if G.has_node(o_id) else r['object']

            all_relations.append({
                "Chủ thể": s_label, "Quan hệ": pred_label, "Đối tượng": o_label
            })
            G.add_edge(s_id, o_id, relation=pred_label)

            # Update Knowledge Base logic
            if pred_label == 'Có_triệu_chứng' and r['subject_type'] == 'Bệnh' and s_id in knowledge_base:
                knowledge_base[s_id]['symptoms'].add(o_label)
            elif pred_label == 'Gây_ra':
                if r['object_type'] == 'Bệnh' and o_id in knowledge_base: 
                    knowledge_base[o_id]['causes'].add(s_label)
                elif r['subject_type'] == 'Bệnh' and s_id in knowledge_base: 
                    knowledge_base[s_id]['symptoms'].add(f"Biến chứng: {o_label}")
            elif pred_label == 'Điều_trị_bằng' and r['subject_type'] == 'Bệnh' and s_id in knowledge_base:
                knowledge_base[s_id]['treatments'].add(o_label)
            elif pred_label == 'Chẩn_đoán_bằng' and r['subject_type'] == 'Bệnh' and s_id in knowledge_base:
                knowledge_base[s_id]['diagnosis'].add(o_label)

    GRAPH_STORE["G"] = G
    GRAPH_STORE["knowledge_base"] = knowledge_base

    df = pd.DataFrame(all_relations)
    if df.empty: df = pd.DataFrame(columns=['Chủ thể', 'Quan hệ', 'Đối tượng'])
    return G, df, list(knowledge_base.keys())

def draw_graph(G):
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="Chưa có dữ liệu")
        return fig

    pos = nx.spring_layout(G, k=0.8, seed=42)
    edge_x, edge_y, edge_txt = [], [], []
    for e in G.edges(data=True):
        x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_txt.append(e[2].get('relation', ''))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color='#94a3b8'), 
        hoverinfo='text', text=edge_txt, mode='lines'
    )

    node_x, node_y, node_txt, node_col, hover_text = [], [], [], [], []
    for n in G.nodes(data=True):
        node_id = n[0]
        attrs = n[1]
        if node_id not in pos: continue
        x, y = pos[node_id]
        node_x.append(x); node_y.append(y)
        display_text = attrs.get('label', node_id)
        node_txt.append(display_text)
        node_col.append(TYPE_COLORS.get(attrs.get('type'), '#999'))
        hover_text.append(f"<b>{display_text}</b><br>Loại: {attrs.get('type')}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_txt, 
        textposition="top center", hoverinfo="text", hovertext=hover_text,
        marker=dict(color=node_col, size=25, line=dict(width=2, color='white')),
        textfont=dict(size=11, color='#333')
    )
    return go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, margin=dict(b=20,l=20,r=20,t=20)))

def answer_question(disease_key, question_type):
    kb = GRAPH_STORE["knowledge_base"]
    if not disease_key or disease_key not in kb: return "Vui lòng chọn một bệnh để tra cứu."
    data = kb[disease_key]
    mapping = {"Nguyên nhân": data['causes'], "Triệu chứng": data['symptoms'], "Điều trị": data['treatments'], "Chẩn đoán": data['diagnosis']}
    items = mapping.get(question_type, [])
    if not items: return f"Không tìm thấy thông tin về **{question_type}** của bệnh này."
    return f"### {question_type} của {data['display_name']}\n" + "\n".join([f"- {i}" for i in items])

def run_app():
    custom_css = ".gradio-container {font-family: 'Segoe UI', sans-serif;} h1 {color: #0f766e; text-align: center;}"
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="blue")

    with gr.Blocks(theme=theme, css=custom_css, title="Medical KG") as demo:
        gr.Markdown("# HỆ THỐNG TRÍCH XUẤT TRI THỨC Y KHOA")
        with gr.Row():
            with gr.Column(scale=4):
                inp = gr.Textbox(label="Văn bản", lines=5, value="Bệnh tiểu đường type 2 thường do béo phì và lười vận động gây ra. Triệu chứng của bệnh tiểu đường type 2 gồm khát nước, mệt mỏi và sụt cân bất thường. Điều trị bệnh tiểu đường type 2 bằng thuốc Metformin và thay đổi lối sống.")
                btn = gr.Button("Phân tích", variant="primary")
                with gr.Group():
                    dd = gr.Dropdown(label="Chọn bệnh", interactive=True)
                    rad = gr.Radio(["Nguyên nhân", "Triệu chứng", "Điều trị", "Chẩn đoán"], label="Câu hỏi", value="Triệu chứng")
                    btn_a = gr.Button("Hỏi")
                out_ans = gr.Markdown()
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.TabItem("Đồ thị"): out_plot = gr.Plot()
                    with gr.TabItem("Bảng"): out_df = gr.Dataframe()

        def on_process(text):
            G, df, diseases = process_text_ui(text)
            kb = GRAPH_STORE["knowledge_base"]
            choices = [(kb[k]['display_name'], k) for k in diseases]
            return df, draw_graph(G), gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

        btn.click(fn=on_process, inputs=[inp], outputs=[out_df, out_plot, dd])
        btn_a.click(fn=answer_question, inputs=[dd, rad], outputs=[out_ans])
    
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    run_app()