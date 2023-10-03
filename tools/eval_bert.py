# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import cuda
from torch.nn import DataParallel
from tqdm import tqdm
import os
from IPython.display import clear_output, display

# Load the model and tokenizer
model_name = "/home/pgajo/working/pt_models/incel-bert-base-multilingual-cased-1000k_multi_finetuned1_hate_speech_metrics_id_23"
# model_name = "/home/pgajo/working/pt_models/bert-base-multilingual-cased_finetuned1_hate_speech_metrics_id_17"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
df_test = pd.read_csv('/home/pgajo/working/data/datasets/Italian/Il_forum_dei_brutti/IFD-IT-500.csv')
display(df_test.head())
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

class TestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def evaluate(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

device = torch.device("cuda" if cuda.is_available() else "cpu")
model.to(device)
model = DataParallel(model) if cuda.device_count() > 1 else model

texts = df_test['text'].tolist()
labels = df_test['hs'].tolist()
test_data = TestDataset(texts, labels, tokenizer)

test_data_loader = DataLoader(
    test_data,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    shuffle=False
)

results = evaluate(model, test_data_loader, device)
print("Results: ", results)


# %%



