import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from torch import nn
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import warnings

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent))
from utils import DATA_DIR, MODEL_DIR

PROCESSED_TRAIN_JSON = os.path.join(DATA_DIR, 'processed', 'train_tasks.json')
PROCESSED_TEST_JSON = os.path.join(DATA_DIR, 'processed', 'test_tasks.json')
MODEL_OUTPUT_DIR = os.path.join(MODEL_DIR, 'ner_bert_model')
BERT_MODEL_NAME = "tmnam/vihealthbert-w_unsup-SynPD"

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def get_all_original_labels(file_paths):
    original_labels = set()
    for path in file_paths:
        if not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'annotations' in item:
                    for res in item['annotations'][0]['result']:
                        if res['type'] == 'labels':
                            label = res['value']['labels'][0]
                            original_labels.add(label)
    return sorted(list(original_labels))

def setup_bio_mappings(original_labels):
    labels = ["O"]
    for lbl in original_labels:
        labels.append(f"B-{lbl}")
        labels.append(f"I-{lbl}")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label, labels

def convert_to_dataset_format(json_data):
    formatted_data = []
    for item in json_data:
        text = item['data']['text']
        spans = []
        if 'annotations' in item:
            for res in item['annotations'][0]['result']:
                if res['type'] == 'labels':
                    spans.append({
                        'start': res['value']['start'],
                        'end': res['value']['end'],
                        'label': res['value']['labels'][0]
                    })
        formatted_data.append({"id": str(item['id']), "text": text, "spans": spans})
    return formatted_data

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
        is_split_into_words=False
    )
    labels = []
    for i, text in enumerate(examples["text"]):
        spans = examples["spans"][i]
        input_ids = tokenized_inputs["input_ids"][i]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        label_ids = []
        idx_in_text = 0
        text_lower = text.lower()
        
        for token in tokens:
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token]:
                label_ids.append(-100)
                continue
            
            clean_token = token.replace("##", "").lower()
            while idx_in_text < len(text) and text[idx_in_text].isspace():
                idx_in_text += 1
            
            if text_lower.startswith(clean_token, idx_in_text):
                start_char = idx_in_text
                end_char = idx_in_text + len(clean_token)
                idx_in_text = end_char 
                
                token_label = "O"
                for span in spans:
                    if start_char >= span['start'] and start_char < span['end']:
                         if start_char == span['start']:
                             token_label = f"B-{span['label']}"
                         else:
                             token_label = f"I-{span['label']}"
                         break
                label_ids.append(label2id.get(token_label, 0))
            else:
                label_ids.append(label2id["O"])
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]
    report = classification_report(flat_labels, flat_preds, output_dict=True, zero_division=0)
    return {
        "precision": report['macro avg']['precision'],
        "recall": report['macro avg']['recall'],
        "f1": report['macro avg']['f1-score'],
        "accuracy": report['accuracy'],
    }

def calculate_class_weights(dataset, label_list, id2label):
    all_labels = []
    for item in dataset:
        for lbl in item['labels']:
            if lbl != -100:
                all_labels.append(lbl)

    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(all_labels), 
        y=all_labels
    )
    weights_vector = np.ones(len(label_list))
    unique_classes = np.unique(all_labels)
    for idx, w in zip(unique_classes, class_weights):
        weights_vector[idx] = w
    weights_vector = np.power(weights_vector, 0.5)
    return weights_vector

def main():
    print(f"BẮT ĐẦU HUẤN LUYỆN NER VỚI MODEL: {BERT_MODEL_NAME}")
    if not os.path.exists(PROCESSED_TRAIN_JSON):
        print("Lỗi: Không tìm thấy file dữ liệu train.")
        return

    original_labels = get_all_original_labels([PROCESSED_TRAIN_JSON, PROCESSED_TEST_JSON])
    label2id, id2label, label_list = setup_bio_mappings(original_labels)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(
        BERT_MODEL_NAME, 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    with open(PROCESSED_TRAIN_JSON, 'r', encoding='utf-8') as f:
        train_ds = Dataset.from_list(convert_to_dataset_format(json.load(f)))
    with open(PROCESSED_TEST_JSON, 'r', encoding='utf-8') as f:
        test_ds = Dataset.from_list(convert_to_dataset_format(json.load(f)))

    fn_kwargs = {"tokenizer": tokenizer, "label2id": label2id}
    tokenized_train = train_ds.map(tokenize_and_align_labels, batched=True, fn_kwargs=fn_kwargs)
    tokenized_test = test_ds.map(tokenize_and_align_labels, batched=True, fn_kwargs=fn_kwargs)
    class_weights = calculate_class_weights(tokenized_train, label_list, id2label)

    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=4e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20, 
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    def compute_metrics_wrapper(p):
        return compute_metrics(p, id2label)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        class_weights=class_weights,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper
    )

    print("Đang huấn luyện...")
    trainer.train()
    print(f"Lưu model tại: {MODEL_OUTPUT_DIR}")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    metrics = trainer.evaluate()
    print("Kết quả đánh giá trên tập test:", metrics)

if __name__ == "__main__":
    main()