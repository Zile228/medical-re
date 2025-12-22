import json
import pandas as pd
import itertools
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Thêm đường dẫn để import được utils khi chạy script
sys.path.append(str(Path(__file__).resolve().parent))
from utils import BASE_DIR, DATA_DIR, add_markers

# Định nghĩa đường dẫn Input/Output
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw') 
PROCESSED_TRAIN_JSON = os.path.join(DATA_DIR, 'processed', 'train_tasks.json')
PROCESSED_TEST_JSON = os.path.join(DATA_DIR, 'processed', 'test_tasks.json')
PROCESSED_TRAIN_CSV = os.path.join(DATA_DIR, 'processed', 'train_data.csv')
PROCESSED_TEST_CSV = os.path.join(DATA_DIR, 'processed', 'test_data.csv')

def load_and_merge_json_files(directory):
    """Gộp các file JSON và khử trùng lặp nội dung."""
    json_files = list(Path(directory).glob('*.json'))
    json_files.sort()
    
    if not json_files:
        print(f"Cảnh báo: Không tìm thấy file JSON nào trong {directory}")
        return []

    merged_tasks = []
    seen_texts = set()
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for task in data:
                        text_content = task['data']['text'].strip()
                        if text_content in seen_texts: continue
                        
                        seen_texts.add(text_content)
                        new_task = task.copy()
                        new_task['id'] = len(merged_tasks) + 1 
                        merged_tasks.append(new_task)
        except Exception as e:
            print(f"Lỗi đọc file {file_path.name}: {e}")
            
    return merged_tasks

def process_tasks_to_pairs(tasks_list):
    """Chuyển đổi Task Label Studio sang dạng Cặp (Pairs) cho RE."""
    processed_samples = []
    
    for task in tasks_list:
        original_text = task.get('data', {}).get('text', "")
        if not original_text or not task.get('annotations'): continue
        
        result = task['annotations'][0]['result']
        entities = {}
        relations = []
        
        # 1. Parse JSON
        for item in result:
            if item['type'] == 'labels':
                if 'text' not in item['value'] or 'labels' not in item['value']:
                    continue
                    
                entities[item['id']] = {
                    'id': item['id'],
                    'text': item['value']['text'],
                    'start': item['value']['start'],
                    'end': item['value']['end'],
                    'label': item['value']['labels'][0]
                }
            elif item['type'] == 'relation':
                label_list = item.get('labels', [])
                
                if label_list:
                    relations.append({
                        'from_id': item['from_id'],
                        'to_id': item['to_id'],
                        'label': label_list[0]
                    })
        
        entity_ids = list(entities.keys())
        if len(entity_ids) < 2: continue
            
        for source_id, target_id in itertools.permutations(entity_ids, 2):
            relation_label = "No_relation"
            for rel in relations:
                if rel['from_id'] == source_id and rel['to_id'] == target_id:
                    relation_label = rel['label']
                    break
            
            subj = entities[source_id]
            obj = entities[target_id]
            
            # Gọi hàm add_markers từ utils
            marked_sentence = add_markers(
                original_text, 
                {'start': subj['start'], 'end': subj['end']},
                {'start': obj['start'], 'end': obj['end']},
                "S", "O" 
            )
            
            sample = {
                'original_sentence': original_text,
                'subject_text': subj['text'],
                'subject_type': subj['label'],
                'object_text': obj['text'],
                'object_type': obj['label'],
                'marked_sentence': marked_sentence,
                'relation_label': relation_label
            }
            processed_samples.append(sample)

    return pd.DataFrame(processed_samples)

def main():
    print("--- BẮT ĐẦU XỬ LÝ DỮ LIỆU ---")
    
    # 1. Gộp và lọc trùng
    all_tasks = load_and_merge_json_files(RAW_DATA_DIR)
    print(f"Tổng số câu sạch: {len(all_tasks)}")

    # 2. Chia tập Train/Test (80/20)
    train_tasks, test_tasks = train_test_split(all_tasks, test_size=0.2, random_state=42)
    
    # 3. Lưu JSON để dùng cho NER
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_JSON), exist_ok=True)
    with open(PROCESSED_TRAIN_JSON, 'w', encoding='utf-8') as f:
        json.dump(train_tasks, f, ensure_ascii=False, indent=2)
    with open(PROCESSED_TEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(test_tasks, f, ensure_ascii=False, indent=2)

    # 4. Tạo CSV cho RE
    print("Đang tạo dữ liệu cặp cho RE...")
    df_train = process_tasks_to_pairs(train_tasks)
    df_test = process_tasks_to_pairs(test_tasks)

    df_train.to_csv(PROCESSED_TRAIN_CSV, index=False, encoding='utf-8')
    df_test.to_csv(PROCESSED_TEST_CSV, index=False, encoding='utf-8')

    print("HOÀN TẤT XỬ LÝ")

if __name__ == "__main__":
    main()