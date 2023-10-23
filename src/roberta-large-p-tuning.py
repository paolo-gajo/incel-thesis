import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)
from tqdm.auto import tqdm

# Load the data from the CSV file
df = pd.read_csv('/home/pgajo/working/food/IFD-EN-5203.csv',
                #  nrows=100,
                 )

df_0_filename = '/home/pgajo/working/incels/data/datasets/English/Incels.is/ECIR/IFD-EN-5203_truncated_0.tsv'
df_0 = pd.read_csv(df_0_filename, sep='\t')
df_0['source'] = 'df' + df_0_filename[-5]
df_1_filename = '/home/pgajo/working/incels/data/datasets/English/Incels.is/ECIR/IFD-EN-5203_truncated_1.tsv'
df_1 = pd.read_csv(df_1_filename, sep='\t')
df_1['source'] = 'df' + df_1_filename[-5]
df_2_filename = '/home/pgajo/working/incels/data/datasets/English/Incels.is/ECIR/IFD-EN-5203_truncated_2.tsv'
df_2 = pd.read_csv(df_2_filename, sep='\t')
df_2['source'] = 'df' + df_2_filename[-5]

df_0_1 = pd.concat([df_0, df_1], ignore_index=True)
df_0_2 = pd.concat([df_0, df_2], ignore_index=True)
df_0_1_2 = pd.concat([df_0, df_1, df_2], ignore_index=True)

df_list = [df_0_1, df_0_2, df_0_1_2]

# Initialize the BERT tokenizer
# model_name = 'bert-base-uncased'
# model_name = 'bert-large-uncased'
# model_name = 'xlm-roberta-large'
model_name = 'roberta-large'
# model_name = 'Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit'
# model_name = 'Muennighoff/SGPT-125M-weightedmean-nli-bitfit'

tokenizer = AutoTokenizer.from_pretrained(model_name)
# if tokenizer._pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# Tokenize the 'text' column and get the attention masks
input_ids = []
attention_masks = []

for text in df['text']:
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df['hs'].values)

# Create a DataLoader
dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load the model
model_name = model_name
model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)

# Convert the model to PEFT
peft_config = PromptEncoderConfig(task_type='SEQ_CLS', num_virtual_tokens=20, encoder_hidden_size=128)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Initialize the Adam optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# model.config.pad_token_id = model.config.eos_token_id
# Parallelize the model
if torch.cuda.device_count() > 1:
    print('Use', torch.cuda.device_count(), 'GPUs!')
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
metrics_df = pd.DataFrame(columns=['Epoch', 'Phase', 'Accuracy', 'Precision', 'Recall', 'F1'])

csv_file_path = f'/home/pgajo/working/food/{model_name}_metrics.csv'

num_epochs = 5

for epoch in tqdm(range(num_epochs), desc='Epochs'):
    for phase in ['Training', 'Validation']:
        if phase == 'Training':
            model.train()
            dataloader = train_dataloader
        else:
            model.eval()
            dataloader = val_dataloader
            
        all_preds, all_labels = [], []
        
        for batch in tqdm(dataloader, desc=f'{phase} batches'):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            model.zero_grad()
            
            with torch.set_grad_enabled(phase == 'Training'):
                outputs = model(input_ids, attention_mask=input_mask, labels=labels)
                loss = outputs.loss.mean()
                
                if phase == 'Training':
                    loss.backward()
                    optimizer.step()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute and print the metrics
        epoch_metrics = compute_metrics(all_preds, all_labels)
        epoch_metrics['Epoch'] = epoch + 1
        epoch_metrics['Phase'] = phase
        print(f'Metrics for {phase} in Epoch {epoch}: {epoch_metrics}')

        metrics_df = metrics_df.append(epoch_metrics, ignore_index=True)

        # Save the metrics to a .csv file
        metrics_df.to_csv(csv_file_path, index=False)



