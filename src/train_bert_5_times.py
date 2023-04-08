
# Load dependencies

import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig, AutoModelForSequenceClassification
import json
# used to make train/dev/test partitions
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from IPython.display import clear_output, display
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel, BertPreTrainedModel, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, log_loss
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler, SequentialSampler
import random
import os
import csv
from pgfuncs import tokenize_and_vectorize, pad_trunc, collect_expected, tokenize_and_vectorize_1dlist, collect_expected_1dlist, df_classification_report

from datetime import datetime
# timestamp for file naming
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
date_str = now.strftime("%Y-%m-%d")

# Load data


# load incelsis_5203 dataset
df_incelsis_5203 = pd.read_csv(
    '/home/pgajo/working/data/datasets/English/Incels.is/IFD-EN-5203_splits.csv')

df_train_incelsis_5203 = df_incelsis_5203[df_incelsis_5203['data_type']
                                          == 'train_incelsis']
df_dev_incelsis_5203 = df_incelsis_5203[df_incelsis_5203['data_type']
                                        == 'dev_incelsis']
df_test_incelsis_5203 = df_incelsis_5203[df_incelsis_5203['data_type']
                                         == 'test_incelsis']

# Print the size of each split
print('Incels.is train set size:', len(df_train_incelsis_5203))
print('Incels.is dev set size:', len(df_dev_incelsis_5203))
print('Incels.is test set size:', len(df_test_incelsis_5203))

# load fdb_250 dataset
df_fdb_250 = pd.read_csv(
    '/home/pgajo/working/data/datasets/Italian/Il_forum_dei_brutti/IFD-IT-250.csv')
df_fdb_250 = df_fdb_250[['hs', 'text']]
df_fdb_250
df_fdb_250['data_type'] = 'test_fdb_250'

print('Forum dei brutti test set size:', len(df_fdb_250))

# load the davidson set
file_path_csv_davidson = '/home/pgajo/working/data/datasets/English/hate-speech-and-offensive-language (davidson)/davidson_labeled_data.csv'
df_davidson = pd.read_csv(file_path_csv_davidson, index_col=None)
df_davidson = df_davidson[['hs', 'text']]
df_davidson['data_type'] = 'davidson'
df_davidson = df_davidson.sample(
    frac=1).reset_index(drop=True)  # shuffle the set
mask = df_davidson['hs'] >= 1

# Set those values to 1
df_davidson.loc[mask, 'hs'] = 1

# Split the data into training and test sets (70% for training, 30% for test)
df_train_davidson, df_test_davidson = train_test_split(
    df_davidson, test_size=0.3, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev_davidson, df_test_davidson = train_test_split(
    df_test_davidson, test_size=0.5, random_state=42)

# sample and get the same proportions of binary classes as the incels dataset

# Set the desired proportions of 1's and 0's in the sample
prop_1 = df_train_incelsis_5203['hs'].value_counts(normalize=True)[1]
df_len = len(df_train_incelsis_5203)
# Calculate the number of rows with 1s and 0s in the sample
num_hs_1 = int(df_len * prop_1)
num_hs_0 = df_len - num_hs_1

# Select rows with 1s and 0s separately, and concatenate the results
df_hs_1 = df_train_davidson[df_train_davidson['hs']
                            == 1].sample(n=num_hs_1, replace=True)
df_hs_0 = df_train_davidson[df_train_davidson['hs']
                            == 0].sample(n=num_hs_0, replace=True)
df_train_davidson_sample = pd.concat([df_hs_1, df_hs_0])

# Print the sample
# print(df_sample_davidson['hs'].value_counts(normalize=True))
# print(df_sample_davidson['hs'].value_counts(normalize=False))

# Print the sample
print('df_train_davidson_sample value_counts:')
print(df_train_davidson_sample['hs'].value_counts(normalize=False))
print()

# Print the size of each split
df_train_davidson['data_type'] = 'train_davidson'
df_dev_davidson['data_type'] = 'dev_davidson'
df_test_davidson['data_type'] = 'test_davidson'
print('Davidson full train set size:', len(df_train_davidson))
print('Davidson full dev set size:', len(df_dev_davidson))
print('Davidson full test set size:', len(df_test_davidson))

# load the hateval_2019_english set
file_path_csv_hateval_2019_english_train = '/home/pgajo/working/data/datasets/English/hateval2019_en/hateval2019_en_train_miso.csv'
file_path_csv_hateval_2019_english_dev = '/home/pgajo/working/data/datasets/English/hateval2019_en/hateval2019_en_dev_miso.csv'
file_path_csv_hateval_2019_english_test = '/home/pgajo/working/data/datasets/English/hateval2019_en/hateval2019_en_test_miso.csv'

df_train_hateval_2019_english = pd.read_csv(
    file_path_csv_hateval_2019_english_train, index_col=None)
df_dev_hateval_2019_english = pd.read_csv(
    file_path_csv_hateval_2019_english_dev, index_col=None)
df_test_hateval_2019_english = pd.read_csv(
    file_path_csv_hateval_2019_english_test, index_col=None)

# sample and get the same proportions of binary classes as the incels dataset

# Set the desired proportions of 1's and 0's in the sample
prop_1 = df_train_incelsis_5203['hs'].value_counts(normalize=True)[1]
df_len = len(df_train_incelsis_5203)
# Calculate the number of rows with 1s and 0s in the sample
num_hs_1 = int(df_len * prop_1)
num_hs_0 = df_len - num_hs_1

# Select rows with 1s and 0s separately, and concatenate the results
df_hs_1 = df_train_hateval_2019_english[df_train_hateval_2019_english['hs'] == 1].sample(
    n=num_hs_1, replace=True)
df_hs_0 = df_train_hateval_2019_english[df_train_hateval_2019_english['hs'] == 0].sample(
    n=num_hs_0, replace=True)
df_train_hateval_2019_english_sample = pd.concat([df_hs_1, df_hs_0])

# Print the sample
print('HatEval english sample value_counts:')
print(df_train_hateval_2019_english_sample['hs'].value_counts(normalize=False))
print()
df_train_hateval_2019_english['data_type'] = 'train_hateval_2019_english'
df_dev_hateval_2019_english['data_type'] = 'dev_hateval_2019_english'
df_test_hateval_2019_english['data_type'] = 'test_hateval_2019_english'
print('HatEval english full train set size:',
      len(df_train_hateval_2019_english))
print('HatEval english full dev set size:', len(df_dev_hateval_2019_english))
print('HatEval english full test set size:', len(df_test_hateval_2019_english))

# load the hateval_2019_spanish set
file_path_csv_hateval_2019_spanish_train = '/home/pgajo/working/data/datasets/Spanish/hateval2019_es/hateval2019_es_train.csv'
file_path_csv_hateval_2019_spanish_dev = '/home/pgajo/working/data/datasets/Spanish/hateval2019_es/hateval2019_es_dev.csv'
file_path_csv_hateval_2019_spanish_test = '/home/pgajo/working/data/datasets/Spanish/hateval2019_es/hateval2019_es_test.csv'

df_train_hateval_2019_spanish = pd.read_csv(
    file_path_csv_hateval_2019_spanish_train, index_col=None)
df_train_hateval_2019_spanish = df_train_hateval_2019_spanish.rename(columns={
                                                                     'HS': 'hs'})

df_dev_hateval_2019_spanish = pd.read_csv(
    file_path_csv_hateval_2019_spanish_dev, index_col=None)
df_dev_hateval_2019_spanish = df_dev_hateval_2019_spanish.rename(columns={
                                                                 'HS': 'hs'})

df_test_hateval_2019_spanish = pd.read_csv(
    file_path_csv_hateval_2019_spanish_test, index_col=None)
df_test_hateval_2019_spanish = df_test_hateval_2019_spanish.rename(columns={
                                                                   'HS': 'hs'})

# sample and get the same proportions of binary classes as the incels dataset

# Set the desired proportions of 1's and 0's in the sample
prop_1 = df_train_incelsis_5203['hs'].value_counts(normalize=True)[1]
df_len = len(df_train_incelsis_5203)
# Calculate the number of rows with 1s and 0s in the sample
num_hs_1 = int(df_len * prop_1)
num_hs_0 = df_len - num_hs_1

# Select rows with 1s and 0s separately, and concatenate the results
df_hs_1 = df_train_hateval_2019_spanish[df_train_hateval_2019_spanish['hs'] == 1].sample(
    n=num_hs_1, replace=True)
df_hs_0 = df_train_hateval_2019_spanish[df_train_hateval_2019_spanish['hs'] == 0].sample(
    n=num_hs_0, replace=True)
df_train_hateval_2019_spanish_sample = pd.concat([df_hs_1, df_hs_0])

# Print the sample
print('HatEval spanish sample value_counts:')
print(df_train_hateval_2019_spanish_sample['hs'].value_counts(normalize=False))
print()
df_train_hateval_2019_spanish['data_type'] = 'train_hateval_2019_spanish'
df_dev_hateval_2019_spanish['data_type'] = 'dev_hateval_2019_spanish'
df_test_hateval_2019_spanish['data_type'] = 'test_hateval_2019_spanish'
print('HatEval spanish full train set size:',
      len(df_train_hateval_2019_spanish))
print('HatEval spanish full dev set size:', len(df_dev_hateval_2019_spanish))
print('HatEval spanish full test set size:', len(df_test_hateval_2019_spanish))

# load the HateXplain dataset
filename_json = '/home/pgajo/working/data/datasets/English/HateXplain/Data/dataset.json'

# Open the JSON file
with open(filename_json, 'r') as f:
    # Load the JSON data into a Python dictionary
    dataset_json = json.load(f)


def post_majority_vote_choice(label_list):
    '''
    Returns the majority vote for a post in the HateXplain json dataset.
    '''
    label_dict = {}
    for i, post_label in enumerate(label_list):
        # print(i,post_label)
        if post_label not in label_dict:
            label_dict[post_label] = 1
        else:
            label_dict[post_label] += 1
    max_key = max(label_dict, key=label_dict.get)
    if label_dict[max_key] > 1:
        return max_key  # return the label key with the highest value if > 1


df_hatexplain_list = []
for key_post in dataset_json.keys():
    post = []
    labels_post = [key_annotators['label']
                   for key_annotators in dataset_json[key_post]['annotators']]  # get the list of labels
    label_majority = post_majority_vote_choice(
        labels_post)  # return the majority label
    if label_majority != None:  # the post_majority_vote_choice returns None if there is no majority label, i.e., they all have the same occurrences
        post.append(label_majority)  # append the label of the post
        # append the text tokens of the post
        post.append(' '.join(dataset_json[key_post]['post_tokens']))
        df_hatexplain_list.append(post)  # append the label-text pair
df_hatexplain = pd.DataFrame(df_hatexplain_list, columns=['hs', 'text'])
df_hatexplain_binary = df_hatexplain.loc[df_hatexplain['hs'] != 'offensive']
df_hatexplain_binary['hs'] = df_hatexplain_binary['hs'].replace(
    {'normal': 0, 'hatespeech': 1})

# Split the data into training and test sets (80% for training, 20% for test)
hatexplain_binary_devtest_size = 0.2
df_train_hatexplain_binary, df_test_hatexplain_binary = train_test_split(
    df_hatexplain_binary, test_size=hatexplain_binary_devtest_size, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev_hatexplain_binary, df_test_hatexplain_binary = train_test_split(
    df_test_hatexplain_binary, test_size=0.5, random_state=42)

df_train_hatexplain_binary['data_type'] = 'hatexplain_binary_train'
df_dev_hatexplain_binary['data_type'] = 'hatexplain_binary_dev'
df_test_hatexplain_binary['data_type'] = 'hatexplain_binary_test'
print('HateXplain binary dev+test split ratio:', hatexplain_binary_devtest_size)
print('HateXplain binary full train set size:', len(df_train_hatexplain_binary))
print('HateXplain binary full dev set size:', len(df_dev_hatexplain_binary))
print('HateXplain binary full test set size:', len(df_test_hatexplain_binary))

# load the stormfront dataset from "Hate speech dataset from a white supremacist forum"

df_stormfront_raw = pd.read_csv(
    '/home/pgajo/working/data/datasets/English/hate-speech-dataset-stormfront/annotations_metadata.csv')
df_stormfront_raw['label'] = df_stormfront_raw['label'].replace(
    {'noHate': 0, 'hate': 1})
df_stormfront_raw = df_stormfront_raw.rename(columns={'label': 'hs'})

post_dir = '/home/pgajo/working/data/datasets/English/hate-speech-dataset-stormfront/all_files'
dict_ids_labels = {}
dict_post_pairs_ws = []

for row in df_stormfront_raw.values.tolist():
    dict_ids_labels[row[0]] = row[4]
len(dict_ids_labels)
for filename in os.listdir(post_dir):
    with open(os.path.join(post_dir, filename), 'r') as file:
        # Read the contents of the file into a string variable
        file_contents = file.read()
        filename = filename[:-4]
    dict_post_pairs_ws.append(
        [dict_ids_labels[filename], file_contents, filename])
df_stormfront = pd.DataFrame(dict_post_pairs_ws, columns=[
                             'hs', 'text', 'filename'])
df_stormfront = df_stormfront[(
    df_stormfront['hs'] == 0) | (df_stormfront['hs'] == 1)]
df_stormfront['hs'] = df_stormfront['hs'].astype(int)

# Split the data into training and test sets (80% for training, 30% for test)
df_stormfront_devtest_size = 0.3
df_train_stormfront, df_test_stormfront = train_test_split(
    df_stormfront, test_size=df_stormfront_devtest_size, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev_stormfront, df_test_stormfront = train_test_split(
    df_test_stormfront, test_size=0.5, random_state=42)

df_train_stormfront['data_type'] = 'df_stormfront_train'
df_dev_stormfront['data_type'] = 'df_stormfront_dev'
df_test_stormfront['data_type'] = 'df_stormfront_test'
print('Stormfront dataset dev+test split size:', df_stormfront_devtest_size)
print('Stormfront dataset train set size:', len(df_train_stormfront))
print('Stormfront dataset dev set size:', len(df_dev_stormfront))
print('Stormfront dataset test set size:', len(df_test_stormfront))

# load the evalita18twitter set
file_path_csv_evalita18twitter_train = '/home/pgajo/working/data/datasets/Italian/haspeede_evalita/2018/TW-folder-20230313T173228Z-001/TW-folder/TW-train/haspeede_TW-train.tsv'

df_train_evalita18twitter = pd.read_csv(
    file_path_csv_evalita18twitter_train, sep='\t', names=['id', 'text', 'hs'])
df_train_evalita18twitter.columns = ['id', 'text', 'hs']
# display(df_train_evalita18twitter)
df_train_evalita18twitter['data_type'] = 'train_evalita18twitter'
print('evalita18twitter full train set size:', len(df_train_evalita18twitter))

# load the evalita18facebook set
file_path_csv_evalita18facebook_train = '/home/pgajo/working/data/datasets/Italian/haspeede_evalita/2018/FB-folder-20230313T173818Z-001/FB-folder/FB-train/haspeede_FB-train.tsv'

df_train_evalita18facebook = pd.read_csv(
    file_path_csv_evalita18facebook_train, sep='\t', names=['id', 'text', 'hs'])
df_train_evalita18facebook['data_type'] = 'train_evalita18facebook'
# display(df_train_evalita18facebook)
print('evalita18facebook full train set size:', len(df_train_evalita18facebook))

# load the evalita20 set
file_path_csv_evalita20_train = '/home/pgajo/working/data/datasets/Italian/haspeede_evalita/2020/haspeede2_dev/haspeede2_dev_taskAB.tsv'

df_train_evalita20 = pd.read_csv(
    file_path_csv_evalita20_train, sep='\t', index_col=None)
# display(df_train_evalita20)

print('evalita20 full train set size:', len(df_train_evalita20))

# load the offenseval_2020 dataset
# from datasets import load_dataset

# configs = ['ar', 'da', 'en', 'gr', 'tr']
# datasets = {}

# for config in configs:
#     datasets[config] = load_dataset("strombergnlp/offenseval_2020", config)


# Experiment setup

metrics_id = 0
# set to -1 for multigpu # Set the index of the CUDA device you want to use
device_index = -1

# experiment setup

# set problem type
prob_type = 'binary'

# set task name
task_name = 'incelsis'

# define dataset combinations
metrics_list_names = [
    # monolingual
    ['train_incelsis_5203', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 0
    ['train_incelsis_5203+train_davidson_sample',
        'dev_incelsis_5203', 'test_incelsis_5203'],  # 1
    ['train_incelsis_5203+train_hateval_2019_english_sample',
        'dev_incelsis_5203', 'test_incelsis_5203'],  # 2
    ['train_incelsis_5203+train_davidson_sample+train_hateval_2019_english_sample',
        'dev_incelsis_5203', 'test_incelsis_5203'],  # 3
    ['train_hateval_2019_english', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 4
    ['train_hateval_2019_english+train_davidson',
        'dev_incelsis_5203', 'test_incelsis_5203'],  # 5
    ['train_incelsis_5203', 'dev_hateval_2019_english',
        'test_hateval_2019_english'],  # 6
    ['train_davidson', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 7
    ['train_incelsis_5203', 'dev_davidson', 'test_davidson'],  # 8
    ['train_incelsis_5203+train_davidson+train_hateval_2019_english',
        'dev_davidson', 'test_davidson'],  # 9
    ['train_incelsis_5203+train_hateval_2019_english',
        'dev_incelsis_5203', 'test_incelsis_5203'],  # 10
    ['train_hatexplain_binary', 'hatexplain_binary_dev',
        'hatexplain_binary_test'],  # 11
    ['train_hatexplain_binary', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 12
    ['train_incelsis_5203+train_hatexplain_binary',
        'dev_incelsis_5203', 'test_incelsis_5203'],  # 13
    ['train_incelsis_5203+train_hatexplain_binary+train_hateval_2019_english',
        'dev_incelsis_5203', 'test_incelsis_5203'],  # 14
    ['train_incelsis_5203+train_stormfront',
        'dev_incelsis_5203', 'test_incelsis_5203'],  # 15
    ['train_incelsis_5203+train_stormfront+train_hateval_2019_english',
        'dev_incelsis_5203', 'test_incelsis_5203'],  # 16

    # multilingual
    ['train_incelsis_5203', 'dev_incelsis_5203', 'test_fdb_250'],  # 17
    ['train_incelsis_5203+train_hateval_2019_english',
        'dev_incelsis_5203', 'test_fdb_250'],  # 18
    ['train_incelsis_5203+train_hateval_2019_spanish',
        'dev_incelsis_5203', 'test_fdb_250'],  # 19
    ['train_incelsis_5203+train_hateval_2019_english+train_hateval_2019_spanish',
        'dev_incelsis_5203', 'test_fdb_250'],  # 20
    ['train_incelsis_5203+train_evalita18facebook',
        'dev_incelsis_5203', 'test_fdb_250'],  # 21
    ['train_incelsis_5203+train_evalita18twitter',
        'dev_incelsis_5203', 'test_fdb_250'],  # 22
    ['train_incelsis_5203+train_evalita18facebook+train_evalita18twitter',
        'dev_incelsis_5203', 'test_fdb_250'],  # 23
    ['train_incelsis_5203', 'dev_incelsis_5203', 'test_fdb_250'],  # 24
    ['train_incelsis_5203', 'dev_incelsis_5203', 'test_fdb_250'],  # 25

]

# set train datasets
df_train = pd.DataFrame()

if 'incelsis' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_incelsis_5203])

if 'davidson' in metrics_list_names[metrics_id][0]:
    if 'incelsis' in metrics_list_names[metrics_id][0] and 'sample' in metrics_list_names[metrics_id][0]:
        df_train = pd.concat([df_train, df_train_davidson_sample])
    else:
        df_train = pd.concat([df_train, df_train_davidson])

if 'hateval' in metrics_list_names[metrics_id][0]:
    if 'english' in metrics_list_names[metrics_id][0]:
        if 'incelsis' in metrics_list_names[metrics_id][0] and 'sample' in metrics_list_names[metrics_id][0]:
            df_train = pd.concat(
                [df_train, df_train_hateval_2019_english_sample])
        else:
            df_train = pd.concat([df_train, df_train_hateval_2019_english])
    if 'spanish' in metrics_list_names[metrics_id][0]:
        if 'incelsis' in metrics_list_names[metrics_id][0] and 'sample' in metrics_list_names[metrics_id][0]:
            df_train = pd.concat(
                [df_train, df_train_hateval_2019_english_sample])
        else:
            df_train = pd.concat([df_train, df_train_hateval_2019_spanish])

if 'train_hatexplain_binary' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_hatexplain_binary])

if 'train_stormfront' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_stormfront])

if 'train_evalita18facebook' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_evalita18facebook])

if 'train_evalita18twitter' in metrics_list_names[metrics_id][0]:
    df_train = pd.concat([df_train, df_train_evalita18twitter])

df_dev = pd.DataFrame()
# set dev datasets
if 'dev_incelsis_5203' in metrics_list_names[metrics_id][1]:
    df_dev = pd.concat([df_dev, df_dev_incelsis_5203])

if 'dev_davidson' in metrics_list_names[metrics_id][1]:
    df_dev = pd.concat([df_dev, df_dev_davidson])

if 'dev_hateval_2019' in metrics_list_names[metrics_id][1]:
    if 'english' in metrics_list_names[metrics_id][1]:
        df_dev = pd.concat([df_dev, df_dev_hateval_2019_english])
    if 'spanish' in metrics_list_names[metrics_id][1]:
        df_dev = pd.concat([df_dev, df_dev_hateval_2019_spanish])

if 'dev_hatexplain_binary' in metrics_list_names[metrics_id][1]:
    df_dev = pd.concat([df_dev, df_dev_hatexplain_binary])

if 'dev_stormfront' in metrics_list_names[metrics_id][1]:
    df_dev = pd.concat([df_dev, df_dev_stormfront])

# set test datasets
if 'test_incelsis_5203' in metrics_list_names[metrics_id][2]:
    df_test = df_test_incelsis_5203

if 'test_davidson' in metrics_list_names[metrics_id][2]:
    df_test = df_test_davidson

if 'test_hateval_2019' in metrics_list_names[metrics_id][2]:
    if 'english' in metrics_list_names[metrics_id][2]:
        df_test = df_test_hateval_2019_english
    if 'spanish' in metrics_list_names[metrics_id][2]:
        df_test = df_test_hateval_2019_spanish

if 'test_hatexplain_binary' in metrics_list_names[metrics_id][2]:
    df_test = df_test_hatexplain_binary

if 'test_stormfront' in metrics_list_names[metrics_id][2]:
    df_test = df_test_stormfront

if 'test_fdb_250' in metrics_list_names[metrics_id][2]:
    df_test = df_fdb_250

df_train = df_train.sample(frac=1)[:100]
df_dev = df_dev.sample(frac=1)

print('Run ID:', metrics_id)
print('Train sets:')
print(df_train['data_type'].value_counts(normalize=False))
print('Train set length:', len(df_train), '\n')
print('Dev sets:')
print(df_dev['data_type'].value_counts(normalize=False))
print('Train set length:', len(df_dev), '\n')
print('Test sets:')
print(df_test['data_type'].value_counts(normalize=False))
print('Train set length:', len(df_dev), '\n')


# Model choice


model_name_list = [
    'bert-base-uncased',
    'roberta-base',
    '/home/pgajo/working/pt_models/HateBERT',
    '/home/pgajo/working/pt_models/incel-bert-10k',
    '/home/pgajo/working/pt_models/incel-bert-100k',
    '/home/pgajo/working/pt_models/incel-bert-1M',
    # '/home/pgajo/working/pt_models/incel-bert',
    'Hate-speech-CNERG/bert-base-uncased-hatexplain',
    'bert-base-multilingual-cased',
    # '/home/pgajo/working/pt_models/incel-mbert-10k',
    # '/home/pgajo/working/pt_models/incel-mbert-100k',
    # '/home/pgajo/working/pt_models/incel-mbert-1M',
    # '/home/pgajo/working/pt_models/incel-mbert',
    '/home/pgajo/working/pt_models/incel-roberta-base-10k',
    '/home/pgajo/working/pt_models/incel-roberta-base-100k',
    '/home/pgajo/working/pt_models/incel-roberta-base-1M',
    'xlm-roberta-base',
]

for model_name in model_name_list[0:1]:
    for i in range(5):
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        date_str = now.strftime("%Y-%m-%d")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model_name_simple = model_name.split('/')[-1]

        # Data pre-processing
        # Encode the training data using the tokenizer
        encoded_data_train = tokenizer.batch_encode_plus(
            # text to encode, wrapped in a tqdm progress bar to show progress
            [el for el in tqdm(df_train.text.values)],
            # add special tokens to mark the beginning and end of each sentence
            add_special_tokens=True,
            # generate attention masks to distinguish padding from actual tokens
            return_attention_mask=True,
            pad_to_max_length=True,  # pad each sentence to the maximum length
            max_length=256,  # set the maximum length of each sentence to 256
            return_tensors='pt'  # return PyTorch tensors
        )

        # Encode the validation data using the tokenizer
        encoded_data_val = tokenizer.batch_encode_plus(
            # text to encode, wrapped in a tqdm progress bar to show progress
            [el for el in tqdm(df_dev.text.values)],
            # add special tokens to mark the beginning and end of each sentence
            add_special_tokens=True,
            # generate attention masks to distinguish padding from actual tokens
            return_attention_mask=True,
            pad_to_max_length=True,  # pad each sentence to the maximum length
            max_length=256,  # set the maximum length of each sentence to 256
            return_tensors='pt'  # return PyTorch tensors
        )

        # Encode the validation data using the tokenizer
        encoded_data_test = tokenizer.batch_encode_plus(
            # text to encode, wrapped in a tqdm progress bar to show progress
            [el for el in tqdm(df_test.text.values)],
            # add special tokens to mark the beginning and end of each sentence
            add_special_tokens=True,
            # generate attention masks to distinguish padding from actual tokens
            return_attention_mask=True,
            pad_to_max_length=True,  # pad each sentence to the maximum length
            max_length=256,  # set the maximum length of each sentence to 256
            return_tensors='pt'  # return PyTorch tensors
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

        # Create train and validation dataset from extracted features
        from torch.utils.data import TensorDataset
        dataset_train = TensorDataset(
            input_ids_train, attention_masks_train, labels_train)
        dataset_val = TensorDataset(
            input_ids_val, attention_masks_val, labels_val)
        dataset_test = TensorDataset(
            input_ids_test, attention_masks_test, labels_test)
        print("Train set length: {}\nDev set length: {}\nTest set length: {}".format(
            len(dataset_train), len(dataset_val), len(dataset_test)))

        # Define the size of each batch
        batch_size = 16  # number of examples to include in each batch

        # Load training dataset
        dataloader_train = DataLoader(
            dataset_train,  # training dataset to load
            # randomly sample examples from the training dataset
            sampler=RandomSampler(dataset_train),
            batch_size=batch_size  # set the batch size to the defined value
        )

        # Load valuation dataset
        dataloader_val = DataLoader(
            dataset_val,  # valuation dataset to load
            # randomly sample examples from the valuation dataset
            sampler=RandomSampler(dataset_val),
            batch_size=batch_size  # set the batch size to the defined value
        )

        # Load test dataset
        dataloader_test = DataLoader(
            dataset_test,  # testuation dataset to load
            # randomly sample examples from the valuation dataset
            sampler=RandomSampler(dataset_test),
            batch_size=batch_size  # set the batch size to the defined value
        )

        # Model setup
        epochs = 4  # number of epochs

        # Define model optimizer -> Adam
        optimizer = AdamW(
            model.parameters(),  # optimize the parameters of the model
            lr=1e-5,  # set the learning rate to 1e-5
            eps=1e-8  # set the epsilon value to 1e-8
        )

        # Define model scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,  # the optimizer to use
                                                    num_warmup_steps=0,  # number of warmup steps
                                                    num_training_steps=len(dataloader_train)*epochs)  # number of total training steps

        # Define random seeds
        seed_val = 17  # set the seed value to 17
        # Set the seed value for the random number generators in different modules
        # set the seed value for the random module's random number generator
        random.seed(seed_val)
        # set the seed value for NumPy's random number generator
        np.random.seed(seed_val)
        # set the seed value for PyTorch's CPU random number generator
        torch.manual_seed(seed_val)
        # set the seed value for PyTorch's GPU random number generators (if available)
        torch.cuda.manual_seed_all(seed_val)

        # GPU setup (1,2,both)

        # Set the device
        if device_index in [0, 1]:
            device = torch.device(f"cuda:{device_index}")
            model.to(device)
            print(device)
            multi_gpu = 0
        else:
            device = torch.device(f"cuda")
            model.to(device)
            from torch.nn import DataParallel
            model = DataParallel(model)
            multi_gpu = 1

        # Returns the F1 score computed on the predictions

        def f1_score_func(preds, labels, problem_type):
            if problem_type == 'binary':
                average_metric = 'binary'
                preds_flat = np.argmax(preds, axis=1).flatten()
            elif problem_type == 'multiclass':
                average_metric = 'macro'
                preds_flat = np.argmax(preds, axis=1).flatten()
            else:
                raise ValueError(
                    'Invalid problem_type argument. Use either "binary" or "multiclass".')
            labels_flat = labels.flatten()
            return f1_score(labels_flat, preds_flat, average=average_metric)

        # Returns the precision computed on the predictions
        def prec_func(preds, labels, problem_type):
            if problem_type == 'binary':
                average_metric = 'binary'
                preds_flat = np.argmax(preds, axis=1).flatten()
            elif problem_type == 'multiclass':
                average_metric = 'macro'
                preds_flat = np.argmax(preds, axis=1).flatten()
            else:
                raise ValueError(
                    'Invalid problem_type argument. Use either "binary" or "multiclass".')
            labels_flat = labels.flatten()
            return precision_score(labels_flat, preds_flat, average=average_metric)

        # Returns the recall computed on the predictions
        def recall_func(preds, labels, problem_type):
            if problem_type == 'binary':
                average_metric = 'binary'
                preds_flat = np.argmax(preds, axis=1).flatten()
            elif problem_type == 'multiclass':
                average_metric = 'macro'
                preds_flat = np.argmax(preds, axis=1).flatten()
            else:
                raise ValueError(
                    'Invalid problem_type argument. Use either "binary" or "multiclass".')
            labels_flat = labels.flatten()
            return recall_score(labels_flat, preds_flat, average=average_metric)

        # Evaluates the model using the validation set
        def evaluate(dataloader_val, setting='', multi_gpu=0):
            model.eval()
            loss_val_total = 0
            predictions, true_vals = [], []

            for batch in dataloader_val:
                batch = tuple(b.to(device) for b in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[2]
                          }

                with torch.no_grad():
                    outputs = model(**inputs)

                loss = outputs[0]
                if multi_gpu:  # do the mean of the two losses if i'm using 2 GPUs
                    loss = loss.mean()

                logits = outputs[1]
                # loss.mean() when training with multiple gpus, multiple batches at a time, giving multiple losses at a time
                loss_val_total += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = inputs['labels'].cpu().numpy()
                predictions.append(logits)
                true_vals.append(label_ids)

            loss_val_avg = loss_val_total / len(dataloader_val)

            predictions = np.concatenate(predictions, axis=0)
            true_vals = np.concatenate(true_vals, axis=0)
            if len(predictions[0]) > 2 and setting == 'binary':
                predictions = predictions[:, :len(predictions[0])-1]
            return loss_val_avg, predictions, true_vals

        # Define save path

        # filename bits
        multilingual = 0
        if multilingual:
            metrics_save_path = '/home/pgajo/working/data/metrics/metrics_multilingual'
        else:
            metrics_save_path = '/home/pgajo/working/data/metrics/metrics_monolingual'

        metrics_save_path_model = os.path.join(
            metrics_save_path, model_name_simple)

        if not os.path.exists(metrics_save_path_model):
            os.mkdir(metrics_save_path_model)

        metrics_filename = str(metrics_id)+'_' + \
            model_name_simple+'_'+time_str+'_metrics.csv'
        metrics_csv_filepath = os.path.join(
            metrics_save_path_model, metrics_filename)
        print(metrics_csv_filepath)

        # Train model
        # Write set identifiers for the pandas metrics dataframe
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

        # train
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

        df_metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_f1', 'val_prec',
                                  'val_rec', 'test_loss', 'test_f1', 'test_prec', 'test_rec'])
        for epoch in range(1, epochs + 1):
            model.train()  # set the model in training mode

            loss_train_total = 0  # initialize the total training loss

            # Create a progress bar for the training dataloader
            progress_bar = tqdm(dataloader_train,
                                desc=model_name_simple +
                                ' - Epoch {:1d}'.format(epoch),
                                leave=False,
                                disable=False
                                )

            # Loop over the batches in the training dataloader
            for batch in progress_bar:
                model.zero_grad()  # reset the gradients to 0 for each batch
                # move the batch to the device (e.g. GPU)
                batch = tuple(b.to(device) for b in batch)
                inputs = {
                    'input_ids': batch[0],  # input_ids are the token ids
                    # attention_mask masks the padding tokens
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }  # the true labels of the input

                outputs = model(**inputs)
                loss = outputs[0]  # the first element of outputs is the loss

                if multi_gpu:
                    loss = loss.mean()

                loss_train_total += loss.item()  # accumulate the training loss

                loss.backward()  # backpropagate the loss through the model to compute gradients

                # clip the gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()  # update the model parameters using the computed gradients
                scheduler.step()  # update the learning rate using the learning rate scheduler
                # update the progress bar to show the current loss
                progress_bar.set_postfix(
                    {'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            loss_train_avg = loss_train_total / len(dataloader_train)

            val_loss, pred_val, true_values_val = evaluate(
                dataloader_val, prob_type, multi_gpu)  # to check overtraining (or overfitting)
            val_f1 = f1_score_func(pred_val, true_values_val, prob_type)
            dev_prec = prec_func(pred_val, true_values_val, prob_type)
            val_rec = recall_func(pred_val, true_values_val, prob_type)

            test_loss, pred_test, true_values_test = evaluate(
                dataloader_test, prob_type, multi_gpu)  # to check overtraining (or overfitting)
            test_f1 = f1_score_func(pred_test, true_values_test, prob_type)
            test_prec = prec_func(pred_test, true_values_test, prob_type)
            test_rec = recall_func(pred_test, true_values_test, prob_type)

            df_metrics = df_metrics.append({
                'epoch': int(epoch),
                'train_loss': loss_train_avg,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'val_prec': dev_prec,
                'val_rec': val_rec,
                'test_loss': test_loss,
                'test_f1': test_f1,
                'test_prec': test_prec,
                'test_rec': test_rec
            }, ignore_index=True)

            df_metrics['model'] = model_name_simple
            df_metrics['train_len'] = str(len(df_train))
            df_metrics['train_set(s)'] = df_metrics_train_set_string[:-1]
            df_metrics['dev_set(s)'] = df_metrics_dev_set_string[:-1]
            df_metrics['test_set(s)'] = df_metrics_test_set_string[:-1]
            df_metrics['run_id'] = metrics_id

            # print('Run ID:', metrics_id)
            # print('Train sets:')
            # print(df_train['data_type'].value_counts(normalize = False))
            # print('Train set length:', len(df_train), '\n')
            # print('Dev sets:')
            # print(df_dev['data_type'].value_counts(normalize = False))
            # print('Dev set length:', len(df_dev), '\n')
            # print('Test sets:')
            # print(df_test['data_type'].value_counts(normalize = False))
            # print('Test set length:', len(df_test), '\n')

            display(df_metrics)

        # Save metrics
        # metrics_csv_filepath

        # Save the DataFrame to a CSV file
        df_metrics.to_csv(metrics_csv_filepath, index=False)

        # Move the model to CPU
        model = model.cpu()

        # Move the tensors to CPU
        input_ids_train = input_ids_train.cpu()
        attention_masks_train = attention_masks_train.cpu()
        labels_train = labels_train.cpu()

        input_ids_val = input_ids_val.cpu()
        attention_masks_val = attention_masks_val.cpu()
        labels_val = labels_val.cpu()

        input_ids_test = input_ids_test.cpu()
        attention_masks_test = attention_masks_test.cpu()
        labels_test = labels_test.cpu()

        # Delete the tensors
        del input_ids_train
        del attention_masks_train
        del labels_train

        del input_ids_val
        del attention_masks_val
        del labels_val

        del input_ids_test
        del attention_masks_test
        del labels_test

        # Empty the GPU cache
        torch.cuda.empty_cache()
