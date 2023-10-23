# %%
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm.auto import tqdm

# load incelsis_5203 dataset
# df_filename = '/home/pgajo/working/incels/data/datasets/English/Incels.is/IFD-EN-5203.csv'
df_filename = '/home/pgajo/working/incels/data/datasets/Italian/Il_forum_dei_brutti/IFD-IT-500.csv'

df = pd.read_csv(df_filename)

# # only keep rows where 'incel_terms' is 1 or 0. set to None to keep all rows
using_incel_terms = None

df_train = df[df['data_type'] == 'train']
df_train_source = df_filename.split('/')[-1]
df_dev = df[df['data_type'] == 'dev']
df_dev_source = df_filename.split('/')[-1]
if using_incel_terms is not None:
    df = df[df['incel_terms'] == using_incel_terms]
df_test = df[df['data_type'] == 'test']
df_test_source = df_filename.split('/')[-1]

# Print the size of each split
print('train set size:', len(df_train))
print('dev set size:', len(df_dev))
print('test set size:', len(df_test))

df_train = df_train.sample(frac=1)[:]
df_dev = df_dev.sample(frac=1)[:]
df_test = df_test.sample(frac=1)[:]

# model_name = '/home/pgajo/working/incels/pt_models/incel-mbert-hs'
model_name = '/home/pgajo/working/incels/pt_models/incel-bert-base-multilingual-cased-1000k_multi_best_HF_finetuned__metrics_id_ECIR_val_f1_0.8894_test_f1_0.7244_2023-10-18_18-15-39'
# model_name = '/home/pgajo/working/incels/pt_models/incel-bert-base-multilingual-cased-1000k_multi_best_HF_finetuned__metrics_id_ECIR_val_f1_0.8868_test_f1_0.6666_2023-10-17_20-55-05'
# model_name = '/home/pgajo/working/incels/pt_models/incel-bert-base-multilingual-cased-1000k_multi_best_HF_finetuned__metrics_id_ECIR_val_f1_0.8835_test_f1_0.7158_2023-10-17_19-57-30'
# model_name = '/home/pgajo/working/incels/pt_models/incel-bert-base-multilingual-cased-1000k_multi_best_HF_finetuned__metrics_id_ECIR_val_f1_0.8882_test_f1_0.6818_2023-10-17_20-16-39'
# model_name = 'pgajo/incel-bert-base-uncased-1000k_english'
model_name_simple = model_name.split('/')[-1]

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

if len(df_train) > 0:
    encoded_data_train = tokenizer.batch_encode_plus(
        [el for el in tqdm(df_train.text.values)],
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',  # change pad_to_max_length to padding
        max_length=256,
        truncation=True,  # add truncation
        return_tensors='pt'
    )

    # Extract IDs, attention masks and labels from training dataset
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df_train.hs.values)

    # Create DataLoader instances for training and validation sets
    train_data = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32)
else:
    train_dataloader = []

if len(df_dev) > 0:
    encoded_data_val = tokenizer.batch_encode_plus(
        [el for el in tqdm(df_dev.text.values)],
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',  # change pad_to_max_length to padding
        max_length=256,
        truncation=True,  # add truncation
        return_tensors='pt'
    )
    # Extract IDs, attention masks and labels from validation dataset
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df_dev.hs.values)

    val_data = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    val_dataloader = DataLoader(val_data, batch_size=32)
else:
    val_dataloader = []

if len(df_test) > 0:
    encoded_data_test = tokenizer.batch_encode_plus(
        [el for el in tqdm(df_test.text.values)],
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',  # change pad_to_max_length to padding
        max_length=256,
        truncation=True,  # add truncation
        return_tensors='pt'
    )
    # Extract IDs, attention masks and labels from test dataset
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(df_test.hs.values)

    test_data = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    test_dataloader = DataLoader(test_data, batch_size=32)
else:
    test_dataloader = []

# Initialize the BERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    # num_labels=2,
    # output_attentions=False,
    # output_hidden_states=False
)

# # Initialize the Adam optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# parallelize the model
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

# Initialize a DataFrame to store metrics
metrics_df = pd.DataFrame(columns=[
    # 'prec_val',
    # 'rec_val',
    # 'f1_val',
    'prec_test',
    'rec_test',
    'f1_test',
    ])


import time
time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
csv_file_path = f'/home/pgajo/working/incels/data/metrics/clic-it-2023-review/{model_name_simple}_{time_str}_metrics.csv'

# Evaluation phase
model.eval()

# eval_preds, eval_labels = [], []
# for batch in tqdm(val_dataloader, desc='Evaluating'):
#     batch = tuple(b.to(device) for b in batch)
#     inputs = {'input_ids': batch[0],
#                 'attention_mask': batch[1],
#                 'labels': batch[2]}

#     with torch.inference_mode():
#         outputs = model(**inputs)

#     preds = torch.argmax(outputs[1], dim=1).flatten()
#     eval_preds.extend(preds.cpu().numpy())
#     eval_labels.extend(inputs['labels'].cpu().numpy())

# val_metrics = compute_metrics(eval_preds, eval_labels)
# print('val_metrics:', val_metrics)
# val_metrics_df = pd.DataFrame({
#     'prec_val': val_metrics['Precision'],
#     'rec_val': val_metrics['Recall'],
#     'f1_val': val_metrics['F1']
# }, index=[0])

# # concat val_metrics_df to metrics_df
# metrics_df = pd.concat([metrics_df, val_metrics_df], axis=0)

# make a copy of df and add a column for predictions
df_test_preds = df_test.copy()

test_preds, test_labels = [], []
batch_idx_start = 0  # to keep track of where to insert predictions in df_test_preds
for batch in tqdm(test_dataloader, desc='Testing'):
    batch = tuple(b.to(device) for b in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'labels': batch[2]}

    with torch.inference_mode():
        outputs = model(**inputs)

    preds = torch.argmax(outputs[1], dim=1).flatten()
    test_preds.extend(preds.cpu().numpy())
    test_labels.extend(inputs['labels'].cpu().numpy())


test_metrics = compute_metrics(test_preds, test_labels)
print('test_metrics:', test_metrics)
test_metrics_df = pd.DataFrame({
    'prec_test': test_metrics['Precision'],
    'rec_test': test_metrics['Recall'],
    'f1_test': test_metrics['F1']
}, index=[0])

df_test_preds['predictions'] = test_preds

import os
# save test df with predictions
predictions_dir = '/home/pgajo/working/incels/data/metrics/clic-it-2023-review/predictions'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

df_test_preds.to_csv(os.path.join(predictions_dir, f'{model_name_simple}_{time_str}_predictions.csv'), index=False)

# concat test_metrics_df to metrics_df
metrics_df = pd.concat([metrics_df, test_metrics_df], axis=0)

# Save metrics to a CSV file
metrics_df.to_csv(csv_file_path, index=False)
print('.csv saved to:', csv_file_path)
