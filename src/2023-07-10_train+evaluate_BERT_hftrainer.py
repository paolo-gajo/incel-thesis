# %% [markdown]
# Load dependencies
# used to make train/dev/test partitions
from sklearn.model_selection import train_test_split
from typing import Dict
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from IPython.display import clear_output, display
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, log_loss
import random
import os
# import csv
from pgfuncs import tokenize_and_vectorize, pad_trunc, collect_expected, tokenize_and_vectorize_1dlist, collect_expected_1dlist, df_classification_report

from datetime import datetime
# timestamp for file naming
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
date_str = now.strftime("%Y-%m-%d")


def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %% Load data


# load incelsis_5203 dataset
df = pd.read_csv('/home/pgajo/working/incels/data/datasets/English/Incels.is/IFD-EN-5203_splits_incel_terms.csv')

# only keep rows where 'incel_terms' is 1

using_incel_terms = 0

df = df[df['incel_terms'] == using_incel_terms]

df_train = df[df['data_type'] == 'train_incelsis']
df_dev = df[df['data_type'] == 'dev_incelsis']
df_test = df[df['data_type'] == 'test_incelsis']

# Print the size of each split
print('Incels.is train set size:', len(df_train))
print('Incels.is dev set size:', len(df_dev))
print('Incels.is test set size:', len(df_test))

df_train = df_train.sample(frac=1)[:]
df_dev = df_dev.sample(frac=1)[:]
df_test = df_test.sample(frac=1)[:]

if using_incel_terms == 1:
    metrics_id = 'RANLP_review_metrics' + '_' + 'incel_terms'
elif using_incel_terms == 0:
    metrics_id = 'RANLP_review_metrics' + '_' + 'no_incel_terms'

print('Run ID:', metrics_id)
print('Train sets:')
print(df_train['data_type'].value_counts(normalize=False))
print('Train set length:', len(df_train), '\n')
print('Dev sets:')
print(df_dev['data_type'].value_counts(normalize=False))
print('Dev set length:', len(df_dev), '\n')
print('Test sets:')
print(df_test['data_type'].value_counts(normalize=False))
print('Test set length:', len(df_test), '\n')

# %% Model choice

model_name_list = [
    # monolingual models
    'bert-base-uncased',  # 0
    'roberta-base',  # 1
    '/home/pgajo/working/incels/pt_models/HateBERT',  # 2
    'Hate-speech-CNERG/bert-base-uncased-hatexplain', # 3
    '/home/pgajo/working/incels/pt_models/incel-bert-base-uncased-10k_english',  # 4
    '/home/pgajo/working/incels/pt_models/incel-bert-base-uncased-100k_english',  # 5
    '/home/pgajo/working/incels/pt_models/incel-bert-base-uncased-1000k_english',  # 6
    '/home/pgajo/working/incels/pt_models/incel-roberta-base-10k_english',  # 7
    '/home/pgajo/working/incels/pt_models/incel-roberta-base-100k_english',  # 8
    '/home/pgajo/working/incels/pt_models/incel-roberta-base-1000k_english',  # 9

    # multilingual models
    'bert-base-multilingual-cased',  # 10
    '/home/pgajo/working/incels/pt_models/incel-bert-base-multilingual-cased-10k_multi',  # 11
    '/home/pgajo/working/incels/pt_models/incel-bert-base-multilingual-cased-100k_multi',  # 12
    '/home/pgajo/working/incels/pt_models/incel-bert-base-multilingual-cased-1000k_multi',  # 13

    # fine-tuned models
    '/home/pgajo/working/incels/pt_models/bert-base-multilingual-cased_finetuned1_hate_speech_metrics_id_17',  # 14
]

model_name = model_name_list[6]

for i in range(10):

    # Filename bits
    # metrics_path_category = '/home/pgajo/working/incels/data/metrics/1_hate_speech'
    # metrics_path_category = '/home/pgajo/working/incels/data/metrics/2_1_misogyny'
    # metrics_path_category = '/home/pgajo/working/incels/data/metrics/2_2_racism'
    # metrics_path_category = '/home/pgajo/working/incels/data/metrics/3_hate_forecasting'
    metrics_path_category = '/home/pgajo/working/incels/data/metrics/review_metrics'

    if using_incel_terms == 1:
        metrics_save_path = f'{metrics_path_category}/incel_terms/'
        if not os.path.exists(metrics_save_path):
            os.mkdir(metrics_save_path)
    else:
        metrics_save_path = f'{metrics_path_category}/non_incel_terms/'
        if not os.path.exists(metrics_save_path):
            os.mkdir(metrics_save_path)

    model_name_simple = model_name.split('/')[-1]

    metrics_save_path_model = os.path.join(metrics_save_path, model_name_simple)
    print(metrics_save_path_model)
    # metrics_save_path_model = metrics_save_path + model_name_simple

    if not os.path.exists(metrics_save_path_model):
        os.mkdir(metrics_save_path_model)

    print('\n#####################################################\n',
        metrics_save_path_model,
        '\n#####################################################\n')

    set_seeds(seed_value=i)

    # reset time
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    date_str = now.strftime("%Y-%m-%d")

    # make unique filepath
    metrics_filename = str(metrics_id)+'_' + \
        model_name_simple+'_'+time_str+'_metrics.csv'
    metrics_csv_filepath = os.path.join(
        metrics_save_path_model, metrics_filename)
    print(metrics_csv_filepath)

    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # hatexplain needs modified output layer
    if model_name == 'Hate-speech-CNERG/bert-base-uncased-hatexplain':
        model.classifier = nn.Linear(model.config.hidden_size, 2)
        model.num_labels = 2
        print(model.eval())
        print(model.config)

    # Data pre-processing
    display(df_test)
    # Encode the training data using the tokenizer
    encoded_data_train = tokenizer.batch_encode_plus(
        [el for el in tqdm(df_train.text.values)],
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',  # change pad_to_max_length to padding
        max_length=256,
        truncation=True,  # add truncation
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        [el for el in tqdm(df_dev.text.values)],
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',  # change pad_to_max_length to padding
        max_length=256,
        truncation=True,  # add truncation
        return_tensors='pt'
    )

    encoded_data_test = tokenizer.batch_encode_plus(
        [el for el in tqdm(df_test.text.values)],
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
    # Extract IDs, attention masks and labels from validation dataset
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df_dev.hs.values)
    # Extract IDs, attention masks and labels from test dataset
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(df_test.hs.values)

    # # Model setup
    epochs = 4  # number of epochs
    # Define the size of each batch
    batch_size = 16  # number of examples to include in each batch

    # convert my train/dev/test pandas dataframes to huggingface-compatible datasets
    class CustomDataset(Dataset):
        def __init__(self, input_ids, attention_masks, labels):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_masks[idx], 'labels': self.labels[idx]}

    # make initial empty metrics dataframe
    df_metrics = pd.DataFrame(columns=['epoch', 'loss_train', 'eval_loss', 'eval_f1',
                            'eval_prec', 'eval_rec', 'test_loss', 'test_f1', 'test_prec', 'test_rec'])

    # custom compute metrics function
    def compute_metrics(eval_pred, metric_key_prefix="eval"):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary')

        return {
            f'{metric_key_prefix}_prec': precision,
            f'{metric_key_prefix}_rec': recall,
            f'{metric_key_prefix}_f1': f1
        }

    # Create the custom dataset instances
    train_dataset = CustomDataset(
        input_ids_train, attention_masks_train, labels_train)
    val_dataset = CustomDataset(
        input_ids_val, attention_masks_val, labels_val)
    test_dataset = CustomDataset(
        input_ids_test, attention_masks_test, labels_test)

    # write set identifiers for the pandas metrics dataframe
    df_metrics_train_set_string = ''
    for i, index in enumerate(df_train['data_type'].value_counts(normalize=False).index.to_list()):
        set_len = df_train['data_type'].value_counts(
            normalize=False).values[i]
        df_metrics_train_set_string += index+'('+str(set_len)+')'+'\n'

    df_metrics_dev_set_string = ''
    for i, index in enumerate(df_dev['data_type'].value_counts(normalize=False).index.to_list()):
        set_len = df_dev['data_type'].value_counts(
            normalize=False).values[i]
        df_metrics_dev_set_string += index+'('+str(set_len)+')'+'\n'

    df_metrics_test_set_string = ''
    for i, index in enumerate(df_test['data_type'].value_counts(normalize=False).index.to_list()):
        set_len = df_test['data_type'].value_counts(
            normalize=False).values[i]
        df_metrics_test_set_string += index+'('+str(set_len)+')'+'\n'

    # extend the huggingface Trainer class to make custom methods
    class CustomTrainer(Trainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None):
            val_output = self.predict(val_dataset)
            test_output = self.predict(test_dataset)
            val_metrics = compute_metrics(
                (val_output.predictions, val_output.label_ids), metric_key_prefix="val")
            test_metrics = compute_metrics(
                (test_output.predictions, test_output.label_ids), metric_key_prefix="test")
            df_metrics = pd.DataFrame(columns=[
                                    'epoch', 'val_f1', 'val_prec', 'val_rec', 'test_f1', 'test_prec', 'test_rec'])
            if self.state.epoch == None:
                current_epoch = -1
            else:
                current_epoch = self.state.epoch
            df_metrics = df_metrics.append({
                'epoch': current_epoch, # self.state.epoch,
                'val_f1': val_metrics['val_f1'],
                'val_prec': val_metrics['val_prec'],
                'val_rec': val_metrics['val_rec'],
                'test_f1': test_metrics['test_f1'],
                'test_prec': test_metrics['test_prec'],
                'test_rec': test_metrics['test_rec'],
            }, ignore_index=True)

            df_metrics['model'] = model_name_simple
            df_metrics['train_len'] = str(len(df_train))
            df_metrics['train_set(s)'] = df_metrics_train_set_string[:-1]
            df_metrics['dev_set(s)'] = df_metrics_dev_set_string[:-1]
            df_metrics['test_set(s)'] = df_metrics_test_set_string[:-1]
            df_metrics['run_id'] = metrics_id

            # make unique filepath
            metrics_filename = str(metrics_id)+'_' + \
                model_name_simple+'_'+time_str+'_metrics.csv'
            metrics_csv_filepath = os.path.join(
                metrics_save_path_model, metrics_filename)
            print(metrics_csv_filepath)

            # Save test metrics to CSV
            if not os.path.exists(metrics_csv_filepath):
                df_metrics.to_csv(metrics_csv_filepath, index=False)
            else:
                df_metrics.to_csv(metrics_csv_filepath,
                                mode='a', header=False, index=False)

            return val_metrics

        def log(self, logs: Dict[str, float]):
            # Call the original `log` method to preserve its functionality
            super().log(logs)

            # Calculate total steps
            total_steps = len(
                self.train_dataset) * self.args.num_train_epochs // self.args.per_device_train_batch_size
            if self.args.world_size > 1:
                total_steps = total_steps // self.args.world_size

            # Calculate the percentage of completed steps
            progress_percentage = 100 * self.state.global_step / total_steps

            # Print the custom message
            print("Global step:", self.state.global_step)
            print(
                f"Progress: {progress_percentage:.2f}% steps completed ({self.state.global_step}/{total_steps})")
            print(f"Current model: {model_name_simple}")
            print(f"Current run id: {metrics_id}")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',           # Output directory for model and predictions
        num_train_epochs=epochs,          # Number of epochs
        # Batch size per device during training
        per_device_train_batch_size=batch_size,
        # Batch size per device during evaluation
        per_device_eval_batch_size=batch_size,
        warmup_steps=0,                  # Linear warmup over warmup_steps
        weight_decay=0.01,               # Weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=100,               # Log every X updates steps
        evaluation_strategy='epoch',     # Evaluate every epoch
        save_strategy='no',              # Do not save checkpoint after each epoch
        # load_best_model_at_end=True,     # Load the best model when finished training (best on dev set)
        metric_for_best_model='f1',      # Use f1 score to determine the best model
        greater_is_better=True,           # The higher the f1 score, the better
    )

    # define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
        eps=1e-8,
    )

    # instantiate trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # pass the new optimizer to the trainer
        optimizers=(optimizer, None),
    )

    model_path = '/home/pgajo/working/incels/pt_models'
    model_name_ft = model_name_simple + '_' + 'finetuned' + metrics_path_category.split('/')[-1] + '_' + 'metrics_id_' + str(metrics_id)
    model_save_path = os.path.join(model_path, model_name_ft)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print('###################################')
    print('Saving model to: ', model_save_path)
    print('###################################')

    # Train the model
    trainer.train()

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)