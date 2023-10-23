# Load dependencies
# used to make train/dev/test partitions
from sklearn.model_selection import train_test_split
from typing import Dict
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, TrainerCallback, TrainerControl, TrainerState
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
print(os.environ)
from datetime import datetime
from torch.utils.data import DataLoader

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
df_filename_incels = '/home/pgajo/working/incels/data/datasets/English/Incels.is/IFD-EN-5203.csv'
df_incels = pd.read_csv(df_filename_incels)

# only keep rows where 'incel_terms' is 1 or 0. set to None to keep all rows
using_incel_terms = None

df_incels_train = df_incels[df_incels['data_type'] == 'train']
df_incels_train_source = 'incels_5203_train'
df_incels_dev = df_incels[df_incels['data_type'] == 'dev']
df_incels_dev_source = 'incels_5203_dev'
if using_incel_terms is not None:
    df_incels = df_incels[df_incels['incel_terms'] == using_incel_terms]
df_incels_test = df_incels[df_incels['data_type'] == 'test']
df_incels_test_source = 'incels_5203_test'

print('incelsis_5203 dataset train set size:', len(df_incels_train))
print('incelsis_5203 dataset dev set size:', len(df_incels_dev))    
print('incelsis_5203 dataset test set size:', len(df_incels_test))

# load the ami20 set
file_path_csv_ami20_train = '/home/pgajo/working/incels/data/datasets/Italian/AMI_2020/AMI2020_TrainingSet/AMI2020_training_raw_hs.tsv'

df_ami20 = pd.read_csv(file_path_csv_ami20_train, index_col=None, sep='\t')
# df_ami20.columns = ['id', 'text', 'hs']

# display(df_ami20)
df_ami20 = df_ami20.fillna('')
df_ami20['data_type'] = 'ami20'

# split the ami20 set into training and dev sets, 70:30

df_ami20_dev_size = 0.3

df_train_ami20, df_dev_ami20 = train_test_split(
    df_ami20, test_size=df_ami20_dev_size, random_state=42)
df_train_ami20_data_type_string = 'df_ami20_train'
df_train_ami20['data_type'] = df_train_ami20_data_type_string
df_dev_ami20_data_type_string = 'df_ami20_dev'
df_dev_ami20['data_type'] = df_dev_ami20_data_type_string
print('ami20 dataset dev+test split size:', df_ami20_dev_size)
print('ami20 dataset train set size:', len(df_train_ami20))
print('ami20 dataset dev set size:', len(df_dev_ami20))

# make train, dev and test sets by concatenating the incelsis_5203 and ami20 sets
df_train = pd.concat([df_incels_train, df_train_ami20])
df_dev = pd.concat([df_incels_dev, df_dev_ami20])

# get the test set from IFS-IT
df_filename_ifs_it = '/home/pgajo/working/incels/data/datasets/Italian/Il_forum_dei_brutti/IFD-IT-500.csv'
df_test = pd.read_csv(df_filename_ifs_it)

# Print the size of each split
print('train set size:', len(df_train))
print('dev set size:', len(df_dev))
print('test set size:', len(df_test))

# get dataset names from both the incelsis_5203 and ami sets for train and dev and the ifs-it set for test
df_train_source = df_incels_train_source + ', ' + df_train_ami20_data_type_string
df_dev_source = df_incels_dev_source + ', ' + df_dev_ami20_data_type_string
df_test_source = df_test_source = df_filename_ifs_it.split('/')[-1]

df_train = df_train.sample(frac=1)[:]
df_dev =     df_dev.sample(frac=1)[:]
df_test =   df_test.sample(frac=1)[:]

if using_incel_terms == 1:
    partition_name = 'incel_terms'
elif using_incel_terms == 0:
    partition_name = 'no_incel_terms'
else:
    partition_name = 'full'

metrics_id = 'ECIR'

experiment_name = 'bestsearch'

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
    'pgajo/incel-bert-base-uncased-1000k_english',  # 9

    # multilingual models
    'bert-base-multilingual-cased',  # 10
    '/home/pgajo/working/incels/pt_models/incel-bert-base-multilingual-cased-10k_multi',  # 11
    '/home/pgajo/working/incels/pt_models/incel-bert-base-multilingual-cased-100k_multi',  # 12
    'pgajo/incel-bert-base-multilingual-cased-1000k_multi',  # 13

    # fine-tuned models
    '/home/pgajo/working/incels/pt_models/bert-base-multilingual-cased_finetuned1_hate_speech_metrics_id_17',  # 14

    # large models
    'roberta-large',  # 15
    'bert-large-uncased',  # 16
]

model_name = model_name_list[13]

for i in range(5):
    set_seeds(seed_value = i)
    # reset time
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    date_str = now.strftime("%Y-%m-%d")

    # Filename bits
    metrics_path_category = f'/home/pgajo/working/incels/results/{metrics_id}_{time_str}/'

    if not os.path.exists(metrics_path_category):
        os.makedirs(metrics_path_category)

    if using_incel_terms == 1:
        metrics_save_path = f'{metrics_path_category}/incel_terms/'
        if not os.path.exists(metrics_save_path):
            os.mkdir(metrics_save_path)
    elif using_incel_terms == 0:
        metrics_save_path = f'{metrics_path_category}/non_incel_terms/'
        if not os.path.exists(metrics_save_path):
            os.mkdir(metrics_save_path)
    else:
        metrics_save_path = f'{metrics_path_category}/full/'
        if not os.path.exists(metrics_save_path):
            os.mkdir(metrics_save_path)

    model_name_simple = model_name.split('/')[-1]

    metrics_save_path_model = os.path.join(metrics_save_path, model_name_simple)

    if not os.path.exists(metrics_save_path_model):
        os.mkdir(metrics_save_path_model)

    print('###################################')
    print(metrics_save_path_model)
    print('###################################')
    
    # make unique filepath
    metrics_filename = f"{metrics_id}_{model_name_simple}_{time_str}_metrics.csv"
    metrics_csv_filepath = os.path.join(metrics_save_path_model, metrics_filename)

    print('###################################')
    print('Saving csv to: ', metrics_csv_filepath)
    print('###################################')

    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # hatexplain needs modified output layer
    if model_name == 'Hate-speech-CNERG/bert-base-uncased-hatexplain':
        model.classifier = nn.Linear(model.config.hidden_size, 2)
        model.num_labels = 2

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
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

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
        def log(self, logs: Dict[str, float]):
            # Call the original `log` method to preserve its functionality
            super().log(logs)

            # Calculate total steps
            # total_steps = len(
            #     self.train_dataset) * self.args.num_train_epochs // self.args.per_device_train_batch_size
            # if self.args.world_size > 1:
            #     total_steps = total_steps // self.args.world_size

            # # Calculate the percentage of completed steps
            # progress_percentage = 100 * self.state.global_step / total_steps

            # # Print the custom message
            # print("Global step:", self.state.global_step)
            # print(
            #     f"Progress: {progress_percentage:.2f}% steps completed ({self.state.global_step}/{total_steps})")
            # print(f"Current model: {model_name_simple}")
            # print(f"Current run id: {metrics_id}")
    
    class PrinterCallback(TrainerCallback):
        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            current_epoch = state.epoch
            val_output = trainer.predict(val_dataset)
            test_output = trainer.predict(test_dataset)
            val_metrics = compute_metrics(
                (val_output.predictions, val_output.label_ids), metric_key_prefix="val")
            test_metrics = compute_metrics(
                (test_output.predictions, test_output.label_ids), metric_key_prefix="test")

            df_dict = {
            'epoch' : current_epoch,
            'val_f1' : val_metrics['val_f1'],
            'val_prec' : val_metrics['val_prec'],
            'val_rec' : val_metrics['val_rec'],
            'test_f1' : test_metrics['test_f1'],
            'test_prec' : test_metrics['test_prec'],
            'test_rec' : test_metrics['test_rec'],
            'model' : model_name_simple,
            'train_len' : str(len(df_train)),
            'train_set(s)' : df_metrics_train_set_string[:-1],
            'train_source' : df_train_source,
            'dev_set(s)' : df_metrics_dev_set_string[:-1],
            'dev_source' : df_dev_source,
            'test_set(s)' : df_metrics_test_set_string[:-1],
            'test_source' : df_test_source,
            'run_id' : metrics_id,
            'partition' : partition_name,
            'experiment' : experiment_name,
            }

            # make df_metrics from df_dict

            df_metrics = pd.DataFrame.from_dict(df_dict, orient='index').T

            print('###################################')
            print('df_metrics:\n', df_metrics)

            # Save test metrics to CSV
            if not os.path.exists(metrics_csv_filepath):
                df_metrics.to_csv(metrics_csv_filepath, index=False)
            else:
                df_metrics.to_csv(metrics_csv_filepath, mode='a', header=False, index=False)
            
    # Define training arguments
    training_args = TrainingArguments(
        output_dir = '/home/pgajo/working/incels/pt_models/checkpoints',           # Output directory for model and predictions
        num_train_epochs = epochs,          # Number of epochs
        # Batch size per device during training
        per_device_train_batch_size = batch_size,
        # Batch size per device during evaluation
        per_device_eval_batch_size = batch_size,
        warmup_steps = 0,                  # Linear warmup over warmup_steps
        weight_decay = 0.01,               # Weight decay
        logging_dir = '/home/pgajo/working/incels/logs',            # Directory for storing logs
        logging_steps = 100,               # Log every X updates steps
        evaluation_strategy = 'epoch',     # Evaluate every epoch
        save_strategy = 'epoch',              # Do not save checkpoint after each epoch
        load_best_model_at_end = True,     # Load the best model when finished training (best on dev set)
        metric_for_best_model = 'f1',      # Use f1 score to determine the best model
        greater_is_better = True,          # The higher the f1 score, the better
    )

    # define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr = 2e-5,
        eps = 1e-8,
    )

    # instantiate trainer
    trainer = CustomTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_metrics,
        # pass the new optimizer to the trainer
        optimizers = (optimizer, None),
        callbacks = [PrinterCallback()],
    )



    # Train the model
    trainer.train()

    df_temp = pd.read_csv(metrics_csv_filepath)
    val_f1_max = df_temp['val_f1'].max()
    test_f1_max = df_temp['test_f1'].max()

    model_path = '/home/pgajo/working/incels/pt_models'
    model_name_ft = f"{model_name_simple}_best_HF_finetuned_{metrics_path_category.split('/')[-1]}_metrics_id_{str(metrics_id)}_val_f1_{str(val_f1_max)[:6]}_test_f1_{str(test_f1_max)[:6]}_{time_str}"
    model_save_path = os.path.join(model_path, model_name_ft)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print('###################################')
    print('Saving model to: ', model_save_path)
    print('###################################')

    # def get_raw_output(model, dataset):
    #     class_0_preds = []
    #     class_1_preds = []
    #     final_preds = []
    #     dataloader = DataLoader(dataset, batch_size=32)  # Use your own batch size
    #     model.eval()

    #     with torch.no_grad():
    #         for batch in tqdm(dataloader):
    #             input_ids = batch['input_ids'].to(device)
    #             attention_mask = batch['attention_mask'].to(device)
    #             outputs = model(input_ids, attention_mask=attention_mask)[0]
    #             outputs_np = outputs.cpu().numpy()
                
    #             class_0_preds.extend(outputs_np[:, 0].tolist())
    #             class_1_preds.extend(outputs_np[:, 1].tolist())
    #             final_preds.extend(np.argmax(outputs_np, axis=1).tolist())

    #     return class_0_preds, class_1_preds, final_preds

    # # Assume 'device' is defined (either "cuda" or "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # # Assume 'test_dataset' is defined and is compatible with the model
    # # Also assume 'metrics_path_category' is defined
    # class_0_preds, class_1_preds, final_preds = get_raw_output(model, test_dataset)

    # # Attach to DataFrame
    # df_test_copy = df_test.copy()
    # df_test_copy['class_0_pred'] = class_0_preds
    # df_test_copy['class_1_pred'] = class_1_preds
    # df_test_copy['pred'] = final_preds

    # # Save the test set with predictions to CSV
    # df_test_copy.to_csv(os.path.join(metrics_path_category, f'test_set_with_predictions_{time_str}.csv'), index=False)

    # Save the model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    del model