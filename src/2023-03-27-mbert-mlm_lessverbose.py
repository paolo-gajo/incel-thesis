sample_n_list = [5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
sample_n = sample_n_list[4]
print(sample_n)
language_option_list = ['english', 'italian', 'multi']
language_option = language_option_list[2]
print(language_option)

# monolingual
# model_name = 'bert-base-uncased'
# model_name = 'roberta-base'

# multilingual
model_name = 'bert-base-multilingual-cased'
# model_name = 'xlm-roberta-base'

# Replace these with the appropriate model and tokenizer names
new_model_name = 'incel-'+model_name+'-'+str(int(sample_n/1000))+'k_'+language_option

# Save the model and tokenizer to a directory
output_dir = "/home/pgajo/working/results"

import os
# Create the directory if it doesn't exist
model_path = os.path.join(output_dir,new_model_name)

if not os.path.exists(model_path):
    os.makedirs(model_path)
model_path

import torch
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from functools import partial
from typing import Dict

def load_custom_corpus(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna('')
    df = df[df['data_type'] == 'unknown']
    sentences = [sent for sent in df['text']]
    return {'text': sentences}

def tokenize_function(tokenizer, examples):
    return tokenizer((examples['text']), truncation=True, max_length=128, padding='max_length')

def dataset_map_with_progress(dataset, tokenize_func, batch_size=1000):
    num_processed = 0
    print_interval = 0.05
    dataset_size = len(dataset)
    tokenized_dataset = []

    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)
        batch = dataset[start_idx:end_idx]
        tokenized_batch = tokenize_func(batch)
        tokenized_dataset.append(tokenized_batch)

        num_processed += len(batch)
        progress = num_processed / dataset_size
        if progress >= print_interval:
            print(f"Progress: {progress * 100:.2f}%")
            print_interval += 0.05

    return Dataset.from_pandas(pd.concat(tokenized_dataset, ignore_index=True))

model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

file_path_en = '/home/pgajo/working/data/datasets/English/Incels.is/IFC-22-EN_updated.csv'
corpus_en = load_custom_corpus(file_path_en)
dataset_en = Dataset.from_dict(corpus_en)
tokenized_dataset_en = dataset_map_with_progress(dataset_en, partial(tokenize_function, tokenizer))
print(tokenized_dataset_en)

file_path_it = '/home/pgajo/working/data/datasets/Italian/Il_forum_dei_brutti/IFC-22-IT_updated.csv'
corpus_it = load_custom_corpus(file_path_it)
dataset_it = Dataset.from_dict(corpus_it)
tokenized_dataset_it = dataset_map_with_progress(dataset_it, partial(tokenize_function, tokenizer))
print(tokenized_dataset_it)

# shuffle and sample datasets

# Set a seed to ensure reproducibility when shuffling
seed = 42

from datasets import concatenate_datasets

if language_option == 'multi':
    shuffled_dataset_en = tokenized_dataset_en.shuffle(seed=seed)
    sampled_dataset_en = shuffled_dataset_en.select(range(sample_n))
    print('sampled_dataset_en\n',sampled_dataset_en)

    shuffled_dataset_it = tokenized_dataset_it.shuffle(seed=seed)
    sampled_dataset_it = shuffled_dataset_it.select(range(sample_n))
    print('sampled_dataset_it\n', sampled_dataset_it)

    # Assuming you have loaded the two datasets as `dataset1` and `dataset2`
    merged_dataset = concatenate_datasets([sampled_dataset_en, sampled_dataset_it])
    train_dataset = merged_dataset.shuffle(seed=seed)
    print('train_dataset_multi\n', train_dataset)
    
if language_option == 'english':
    
    shuffled_dataset_en = tokenized_dataset_en.shuffle(seed=seed)
    train_dataset = shuffled_dataset_en.select(range(sample_n))
    print('train_dataset_en\n', train_dataset)
    
if language_option == 'italian':
    
    shuffled_dataset_it = tokenized_dataset_it.shuffle(seed=seed)
    train_dataset = shuffled_dataset_it.select(range(sample_n))
    print('train_dataset_it\n', train_dataset)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
model = AutoModelForMaskedLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir='/home/pgajo/working/results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='/home/pgajo/working/logs/',
    logging_steps=100,
    report_to='none',
    disable_tqdm=False,
)

# extend the huggingface Trainer class to make custom methods
class CustomTrainer(Trainer):
    def log(self, logs: Dict[str, float]):
        # Call the original `log` method to preserve its functionality
        super().log(logs)

        # Calculate total steps
        total_steps = len(self.train_dataset) * self.args.num_train_epochs // self.args.per_device_train_batch_size
        if self.args.world_size > 1:
            total_steps = total_steps // self.args.world_size

        # Calculate the percentage of completed steps
        progress_percentage = 100 * self.state.global_step / total_steps

        # Print the custom message
        print("Global step:", self.state.global_step)
        print(f"MOM LOOK I'M LOGGING: {progress_percentage:.2f}% steps completed ({self.state.global_step}/{total_steps})")

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)


trainer.train(
    # resume_from_checkpoint = True
    )

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)




