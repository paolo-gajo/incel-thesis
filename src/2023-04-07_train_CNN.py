
# %% Load dependencies

import pandas as pd
import numpy as np
from tqdm import tqdm
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

# Keras imports
import keras_tuner as kt
from keras.optimizers import Adam
from keras_tuner.tuners import Hyperband
import json

from keras import backend as K
from keras.models import Sequential  # Base Keras NN model
# Convolution layer and pooling
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Activation, MaxPooling1D, Flatten
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from datetime import datetime
# timestamp for file naming
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
date_str = now.strftime("%Y-%m-%d")

# %% Keras Tuner function definitions

# Functions used for evaluation metrics

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Keras Tuner functions

def build_loaded_hypermodel(hp):  # to use with json config
    # Loss functions: binary_crossentropy
    hp_loss = best_hps['values']['loss_function']
    # Kernel sizes: from 1 to 5
    hp_kernel = best_hps['values']['kernel_size']
    # Number of filters: from 50 to 250, with a step of 25 (e.g. it can be 75, 100 etc.)
    hp_filters = best_hps['values']['conv_filters']
    # Learning rates for the optimizer: 0.002, 0.001, 0.0001
    hp_learning_rate = best_hps['values']['adam_learning_rate']
    # Number of units in the Dense layer: from 32 to 512, with a step of 32
    hp_dense_units = best_hps['values']['dense_units']
    # Dropout value: 0.05, 0.1, 0.2, 0.3
    hp_dropout = best_hps['values']['dropout_value']
    # Intermediate layers: from 1 to 3 sections of Dense layers with Dropout
    hp_layers = best_hps['values']['num_intermediate_layers']

    model_hp = Sequential()
    model_hp.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel, padding='same',
                    activation='relu', strides=1, input_shape=(maxlen, embedding_dims)))
    model_hp.add(GlobalMaxPooling1D())

    for i in range(hp_layers):
        model_hp.add(Dense(units=hp_dense_units, activation='relu'))
        model_hp.add(Dropout(hp_dropout))

    model_hp.add(Activation('relu'))
    model_hp.add(Dense(1, activation='sigmoid'))
    # model_hp.add(Dense(1,activation=hp_output_activation)) # for regression with raw relu and squeezed into sigmoid

    model_hp.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                        loss=hp_loss, metrics=['acc', f1_m, precision_m, recall_m])

    return model_hp

# Defining the hyperparameters to explore, based on the CNN from before

def build_hypermodel(hp):
    # Loss functions: binary_crossentropy
    hp_loss = hp.Choice('loss_function', values=['binary_crossentropy'])
    # Kernel sizes: from 1 to 3
    hp_kernel = hp.Choice('kernel_size', values=[1, 2, 3, 4, 5])
    # Number of filters: from 50 to 250, with a step of 25 (e.g. it can be 75, 100 etc.)
    hp_filters = hp.Int('conv_filters', min_value=50, max_value=250, step=25)
    # Learning rates for the optimizer: 0.002, 0.001, 0.0001
    hp_learning_rate = hp.Choice(
        'adam_learning_rate', values=[0.002, 0.001, 0.0001])
    # Number of units in the Dense layer: from 32 to 512, with a step of 32
    hp_dense_units = hp.Int('dense_units', min_value=32,
                            max_value=512, step=32)
    # Dropout value: 0.05, 0.1, 0.2, 0.3
    hp_dropout = hp.Choice('dropout_value', values=[0.05, 0.1, 0.2, 0.3])
    # Intermediate layers: from 1 to 3 sections of Dense layers with Dropout
    hp_layers = hp.Int('num_intermediate_layers', 1, 3)

    model_hp = Sequential()
    model_hp.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel, padding='same',
                    activation='relu', strides=1, input_shape=(maxlen, embedding_dims)))
    model_hp.add(GlobalMaxPooling1D())

    for i in range(hp_layers):
        model_hp.add(Dense(units=hp_dense_units, activation='relu'))
        model_hp.add(Dropout(hp_dropout))

    model_hp.add(Activation('relu'))
    model_hp.add(Dense(1, activation='sigmoid'))

    model_hp.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                        loss=hp_loss, metrics=['acc', f1_m, precision_m, recall_m])

    return model_hp

# %% Load data

# Load the pre-trained word2vec model
from gensim.models.word2vec import Word2Vec
w2v_model = Word2Vec.load('/home/pgajo/working/VSM_incels.is/2023-02-08_W2V_IFC-22-en.model')

from sklearn.model_selection import train_test_split # used to make train/dev/test partitions

# load incelsis_5203 dataset
df_incelsis_5203 = pd.read_csv('/home/pgajo/working/data/datasets/English/Incels.is/IFD-EN-5203_splits.csv')

df_train_incelsis_5203 = df_incelsis_5203[df_incelsis_5203['data_type'] == 'train_incelsis']
df_dev_incelsis_5203 = df_incelsis_5203[df_incelsis_5203['data_type'] == 'dev_incelsis']
df_test_incelsis_5203 = df_incelsis_5203[df_incelsis_5203['data_type'] == 'test_incelsis']

# Print the size of each split
print('Incels.is train set size:', len(df_train_incelsis_5203))
print('Incels.is dev set size:', len(df_dev_incelsis_5203))
print('Incels.is test set size:', len(df_test_incelsis_5203))

# load fdb_500 dataset
df_fdb_500 = pd.read_csv('/home/pgajo/working/data/datasets/Italian/Il_forum_dei_brutti/IFD-IT-500.csv')
df_fdb_500 = df_fdb_500[['hs','text']]
df_fdb_500
df_fdb_500['data_type']='test_fdb_500'

print('Forum dei brutti test set size:', len(df_fdb_500))

# load the davidson set
file_path_csv_davidson = '/home/pgajo/working/data/datasets/English/hate-speech-and-offensive-language (davidson)/davidson_labeled_data.csv'
df_davidson = pd.read_csv(file_path_csv_davidson, index_col=None)
df_davidson = df_davidson[['hs','text']]
df_davidson['data_type']='davidson'
df_davidson = df_davidson.sample(frac=1).reset_index(drop=True) # shuffle the set
mask = df_davidson['hs'] >= 1

# Set those values to 1
df_davidson.loc[mask, 'hs'] = 1

# Split the data into training and test sets (70% for training, 30% for test)
df_train_davidson, df_test_davidson = train_test_split(df_davidson, test_size=0.3, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev_davidson, df_test_davidson = train_test_split(df_test_davidson, test_size=0.5, random_state=42)

# sample and get the same proportions of binary classes as the incels dataset

# Set the desired proportions of 1's and 0's in the sample
prop_1 = df_train_incelsis_5203['hs'].value_counts(normalize=True)[1]
df_len=len(df_train_incelsis_5203)
# Calculate the number of rows with 1s and 0s in the sample
num_hs_1 = int(df_len * prop_1)
num_hs_0 = df_len - num_hs_1

# Select rows with 1s and 0s separately, and concatenate the results
df_hs_1 = df_train_davidson[df_train_davidson['hs'] == 1].sample(n=num_hs_1, replace=True)
df_hs_0 = df_train_davidson[df_train_davidson['hs'] == 0].sample(n=num_hs_0, replace=True)
df_train_davidson_sample = pd.concat([df_hs_1, df_hs_0])

# Print the sample
# print(df_sample_davidson['hs'].value_counts(normalize=True))
# print(df_sample_davidson['hs'].value_counts(normalize=False))

# Print the sample
print('df_train_davidson_sample value_counts:')
print(df_train_davidson_sample['hs'].value_counts(normalize=False))
print()

# Print the size of each split
df_train_davidson['data_type']='train_davidson'
df_dev_davidson['data_type']='dev_davidson'
df_test_davidson['data_type']='test_davidson'
print('Davidson full train set size:', len(df_train_davidson))
print('Davidson full dev set size:', len(df_dev_davidson))
print('Davidson full test set size:', len(df_test_davidson))

# load the hateval_2019_english set
file_path_csv_hateval_2019_english_train = '/home/pgajo/working/data/datasets/English/hateval2019_en/hateval2019_en_train_miso.csv'
file_path_csv_hateval_2019_english_dev = '/home/pgajo/working/data/datasets/English/hateval2019_en/hateval2019_en_dev_miso.csv'
file_path_csv_hateval_2019_english_test = '/home/pgajo/working/data/datasets/English/hateval2019_en/hateval2019_en_test_miso.csv'

df_train_hateval_2019_english = pd.read_csv(file_path_csv_hateval_2019_english_train, index_col = None)
df_dev_hateval_2019_english = pd.read_csv(file_path_csv_hateval_2019_english_dev, index_col = None)
df_test_hateval_2019_english = pd.read_csv(file_path_csv_hateval_2019_english_test, index_col = None)

# sample and get the same proportions of binary classes as the incels dataset

# Set the desired proportions of 1's and 0's in the sample
prop_1 = df_train_incelsis_5203['hs'].value_counts(normalize = True)[1]
df_len=len(df_train_incelsis_5203)
# Calculate the number of rows with 1s and 0s in the sample
num_hs_1 = int(df_len * prop_1)
num_hs_0 = df_len - num_hs_1

# Select rows with 1s and 0s separately, and concatenate the results
df_hs_1 = df_train_hateval_2019_english[df_train_hateval_2019_english['hs'] == 1].sample(n = num_hs_1, replace = True)
df_hs_0 = df_train_hateval_2019_english[df_train_hateval_2019_english['hs'] == 0].sample(n = num_hs_0, replace = True)
df_train_hateval_2019_english_sample = pd.concat([df_hs_1, df_hs_0])

# Print the sample
print('HatEval english sample value_counts:')
print(df_train_hateval_2019_english_sample['hs'].value_counts(normalize = False))
print()
df_train_hateval_2019_english['data_type']='train_hateval_2019_english'
df_dev_hateval_2019_english['data_type']='dev_hateval_2019_english'
df_test_hateval_2019_english['data_type']='test_hateval_2019_english'
print('HatEval english full train set size:', len(df_train_hateval_2019_english))
print('HatEval english full dev set size:', len(df_dev_hateval_2019_english))
print('HatEval english full test set size:', len(df_test_hateval_2019_english))

# load the hateval_2019_spanish set
file_path_csv_hateval_2019_spanish_train = '/home/pgajo/working/data/datasets/Spanish/hateval2019_es/hateval2019_es_train.csv'
file_path_csv_hateval_2019_spanish_dev = '/home/pgajo/working/data/datasets/Spanish/hateval2019_es/hateval2019_es_dev.csv'
file_path_csv_hateval_2019_spanish_test = '/home/pgajo/working/data/datasets/Spanish/hateval2019_es/hateval2019_es_test.csv'

df_train_hateval_2019_spanish = pd.read_csv(file_path_csv_hateval_2019_spanish_train, index_col = None)
df_train_hateval_2019_spanish = df_train_hateval_2019_spanish.rename(columns={'HS': 'hs'})

df_dev_hateval_2019_spanish = pd.read_csv(file_path_csv_hateval_2019_spanish_dev, index_col = None)
df_dev_hateval_2019_spanish = df_dev_hateval_2019_spanish.rename(columns={'HS': 'hs'})

df_test_hateval_2019_spanish = pd.read_csv(file_path_csv_hateval_2019_spanish_test, index_col = None)
df_test_hateval_2019_spanish = df_test_hateval_2019_spanish.rename(columns={'HS': 'hs'})

# sample and get the same proportions of binary classes as the incels dataset

# Set the desired proportions of 1's and 0's in the sample
prop_1 = df_train_incelsis_5203['hs'].value_counts(normalize = True)[1]
df_len=len(df_train_incelsis_5203)
# Calculate the number of rows with 1s and 0s in the sample
num_hs_1 = int(df_len * prop_1)
num_hs_0 = df_len - num_hs_1

# Select rows with 1s and 0s separately, and concatenate the results
df_hs_1 = df_train_hateval_2019_spanish[df_train_hateval_2019_spanish['hs'] == 1].sample(n = num_hs_1, replace = True)
df_hs_0 = df_train_hateval_2019_spanish[df_train_hateval_2019_spanish['hs'] == 0].sample(n = num_hs_0, replace = True)
df_train_hateval_2019_spanish_sample = pd.concat([df_hs_1, df_hs_0])

# Print the sample
print('HatEval spanish sample value_counts:')
print(df_train_hateval_2019_spanish_sample['hs'].value_counts(normalize = False))
print()
df_train_hateval_2019_spanish['data_type']='train_hateval_2019_spanish'
df_dev_hateval_2019_spanish['data_type']='dev_hateval_2019_spanish'
df_test_hateval_2019_spanish['data_type']='test_hateval_2019_spanish'
print('HatEval spanish full train set size:', len(df_train_hateval_2019_spanish))
print('HatEval spanish full dev set size:', len(df_dev_hateval_2019_spanish))
print('HatEval spanish full test set size:', len(df_test_hateval_2019_spanish))

# load the HateXplain dataset
import json
filename_json = '/home/pgajo/working/data/datasets/English/HateXplain/Data/dataset.json'

# Open the JSON file
with open(filename_json, 'r') as f:
    # Load the JSON data into a Python dictionary
    dataset_json = json.load(f)

def post_majority_vote_choice(label_list):
    '''
    Returns the majority vote for a post in the HateXplain json dataset.
    '''
    label_dict={}
    for i,post_label in enumerate(label_list):
        # print(i,post_label)
        if post_label not in label_dict:
            label_dict[post_label]=1
        else:
            label_dict[post_label]+=1
    max_key = max(label_dict, key=label_dict.get)
    if label_dict[max_key]>1:
        return max_key # return the label key with the highest value if > 1

df_hatexplain_list = []
for key_post in dataset_json.keys():
    post = []
    labels_post = [key_annotators['label'] for key_annotators in dataset_json[key_post]['annotators']] # get the list of labels
    label_majority=post_majority_vote_choice(labels_post) # return the majority label
    if label_majority!=None: # the post_majority_vote_choice returns None if there is no majority label, i.e., they all have the same occurrences
        post.append(label_majority) # append the label of the post
        post.append(' '.join(dataset_json[key_post]['post_tokens'])) # append the text tokens of the post
        df_hatexplain_list.append(post) # append the label-text pair
df_hatexplain=pd.DataFrame(df_hatexplain_list, columns=['hs','text'])
df_hatexplain_binary = df_hatexplain.loc[df_hatexplain['hs'] != 'offensive']
df_hatexplain_binary['hs'] = df_hatexplain_binary['hs'].replace({'normal': 0, 'hatespeech': 1})
# df_hatexplain_binary
# Split the data into training and test sets (80% for training, 20% for test)
hatexplain_binary_devtest_size=0.2
df_train_hatexplain_binary, df_test_hatexplain_binary = train_test_split(df_hatexplain_binary, test_size=hatexplain_binary_devtest_size, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev_hatexplain_binary, df_test_hatexplain_binary = train_test_split(df_test_hatexplain_binary, test_size=0.5, random_state=42)

df_train_hatexplain_binary['data_type']='hatexplain_binary_train'
df_dev_hatexplain_binary['data_type']='hatexplain_binary_dev'
df_test_hatexplain_binary['data_type']='hatexplain_binary_test'
print('HateXplain binary dev+test split ratio:',hatexplain_binary_devtest_size)
print('HateXplain binary full train set size:', len(df_train_hatexplain_binary))
print('HateXplain binary full dev set size:', len(df_dev_hatexplain_binary))
print('HateXplain binary full test set size:', len(df_test_hatexplain_binary))

# load the stormfront dataset from "Hate speech dataset from a white supremacist forum"

df_stormfront_raw=pd.read_csv('/home/pgajo/working/data/datasets/English/hate-speech-dataset-stormfront/annotations_metadata.csv')
df_stormfront_raw['label'] = df_stormfront_raw['label'].replace({'noHate': 0, 'hate': 1})
df_stormfront_raw = df_stormfront_raw.rename(columns={'label': 'hs'})

post_dir='/home/pgajo/working/data/datasets/English/hate-speech-dataset-stormfront/all_files'
dict_ids_labels={}
dict_post_pairs_ws=[]

for row in df_stormfront_raw.values.tolist():
    dict_ids_labels[row[0]]=row[4]
len(dict_ids_labels)
for filename in os.listdir(post_dir):
    with open(os.path.join(post_dir, filename), 'r') as file:
        # Read the contents of the file into a string variable
        file_contents = file.read()
        filename=filename[:-4]
    dict_post_pairs_ws.append([dict_ids_labels[filename],file_contents,filename])
df_stormfront=pd.DataFrame(dict_post_pairs_ws, columns=['hs','text','filename'])
df_stormfront = df_stormfront[(df_stormfront['hs'] == 0) | (df_stormfront['hs'] == 1)]
df_stormfront['hs']=df_stormfront['hs'].astype(int)

# Split the data into training and test sets (80% for training, 30% for test)
df_stormfront_devtest_size=0.3
df_train_stormfront, df_test_stormfront = train_test_split(df_stormfront, test_size=df_stormfront_devtest_size, random_state=42)

# Split the test data into validation and test sets (50% for validation, 50% for test)
df_dev_stormfront, df_test_stormfront = train_test_split(df_test_stormfront, test_size=0.5, random_state=42)

df_train_stormfront['data_type']='df_stormfront_train'
df_dev_stormfront['data_type']='df_stormfront_dev'
df_test_stormfront['data_type']='df_stormfront_test'
print('Stormfront dataset dev+test split size:',df_stormfront_devtest_size)
print('Stormfront dataset train set size:', len(df_train_stormfront))
print('Stormfront dataset dev set size:', len(df_dev_stormfront))
print('Stormfront dataset test set size:', len(df_test_stormfront))

# load the evalita18twitter set
file_path_csv_evalita18twitter_train = '/home/pgajo/working/data/datasets/Italian/haspeede_evalita/2018/TW-folder-20230313T173228Z-001/TW-folder/TW-train/haspeede_TW-train.tsv'

df_train_evalita18twitter = pd.read_csv(file_path_csv_evalita18twitter_train, sep='\t', names=['id','text','hs'])
df_train_evalita18twitter.columns=['id','text','hs']
# display(df_train_evalita18twitter)
df_train_evalita18twitter['data_type'] = 'train_evalita18twitter'
print('evalita18twitter full train set size:', len(df_train_evalita18twitter))

# load the evalita18facebook set
file_path_csv_evalita18facebook_train = '/home/pgajo/working/data/datasets/Italian/haspeede_evalita/2018/FB-folder-20230313T173818Z-001/FB-folder/FB-train/haspeede_FB-train.tsv'

df_train_evalita18facebook = pd.read_csv(file_path_csv_evalita18facebook_train, sep='\t', names=['id','text','hs'])
df_train_evalita18facebook['data_type'] = 'train_evalita18facebook'
# display(df_train_evalita18facebook)
print('evalita18facebook full train set size:', len(df_train_evalita18facebook))

# load the evalita20 set
file_path_csv_evalita20_train = '/home/pgajo/working/data/datasets/Italian/haspeede_evalita/2020/haspeede2_dev/haspeede2_dev_taskAB.tsv'

df_train_evalita20 = pd.read_csv(file_path_csv_evalita20_train, sep='\t', index_col = None)
# display(df_train_evalita20)

print('evalita20 full train set size:', len(df_train_evalita20))

# # load the offenseval_2020 dataset
# from datasets import load_dataset

# configs = ['ar', 'da', 'en', 'gr', 'tr']
# datasets = {}

# for config in configs:
#     datasets[config] = load_dataset("strombergnlp/offenseval_2020", config)

# %% Experiment setup

# NN input settings for vectorization
maxlen = 100
embedding_dims = 300

# define dataset combinations
for j in range(1, 15):
    metrics_id = j

# Dataset combinations

    metrics_list_names = [
        # monolingual
        ['train_incelsis_5203', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 0
        ['train_incelsis_5203+train_davidson_sample', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 1
        ['train_incelsis_5203+train_hateval_2019_english_sample', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 2
        ['train_incelsis_5203+train_davidson_sample+train_hateval_2019_english_sample', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 3
        ['train_hateval_2019_english', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 4
        ['train_hateval_2019_english+train_davidson', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 5
        ['train_incelsis_5203', 'dev_hateval_2019_english', 'test_hateval_2019_english'],  # 6
        ['train_davidson', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 7
        ['train_incelsis_5203', 'dev_davidson', 'test_davidson'],  # 8
        ['train_incelsis_5203+train_davidson+train_hateval_2019_english', 'dev_davidson', 'test_davidson'],  # 9
        ['train_incelsis_5203+train_hateval_2019_english', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 10
        ['train_hatexplain_binary', 'dev_hatexplain_binary', 'test_hatexplain_binary'],  # 11
        ['train_hatexplain_binary', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 12
        ['train_incelsis_5203+train_hatexplain_binary', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 13
        ['train_incelsis_5203+train_hatexplain_binary+train_hateval_2019_english', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 14
        ['train_incelsis_5203+train_stormfront', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 15
        ['train_incelsis_5203+train_stormfront+train_hateval_2019_english', 'dev_incelsis_5203', 'test_incelsis_5203'],  # 16

        # multilingual
        ['train_incelsis_5203', 'dev_incelsis_5203', 'test_fdb_500'],  # 17
        ['train_incelsis_5203+train_hateval_2019_english', 'dev_incelsis_5203', 'test_fdb_500'],  # 18
        ['train_incelsis_5203+train_hateval_2019_spanish', 'dev_incelsis_5203', 'test_fdb_500'],  # 19
        ['train_incelsis_5203+train_hateval_2019_english+train_hateval_2019_spanish', 'dev_incelsis_5203', 'test_fdb_500'],  # 20
        ['train_incelsis_5203+train_evalita18facebook', 'dev_incelsis_5203', 'test_fdb_500'],  # 21
        ['train_incelsis_5203+train_evalita18twitter', 'dev_incelsis_5203', 'test_fdb_500'],  # 22
        ['train_incelsis_5203+train_evalita18facebook+train_evalita18twitter', 'dev_incelsis_5203', 'test_fdb_500'],  # 23
        ['train_incelsis_5203+train_evalita20', 'dev_incelsis_5203', 'test_fdb_500'],  # 24
        ['train_incelsis_5203+train_evalita18facebook+train_evalita18twitter+train_evalita20', 'dev_incelsis_5203', 'test_fdb_500'],  # 25
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
    
    if 'train_evalita20' in metrics_list_names[metrics_id][0]:
        df_train = pd.concat([df_train, df_train_evalita20])

    # set dev datasets
    df_dev = pd.DataFrame()

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

    if 'test_fdb_500' in metrics_list_names[metrics_id][2]:
        df_test = df_fdb_500

    if 'test_davidson' in metrics_list_names[metrics_id][2]:
        df_test = df_test_davidson

    if 'test_hateval_2019' in metrics_list_names[metrics_id][2]:
        if 'english' in metrics_list_names[metrics_id][2]:
            df_test = df_test_hateval_2019_english
        if 'spanish' in metrics_list_names[metrics_id][1]:
            df_test = df_test_hateval_2019_spanish

    if 'test_hatexplain_binary' in metrics_list_names[metrics_id][2]:
        df_test = df_test_hatexplain_binary
    

    df_train = df_train.sample(frac=1)[:]
    df_dev = df_dev.sample(frac=1)[:]
    df_test = df_test.sample(frac=1)[:]

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

    # %% Text pre-processing

    # Vectorize train/dev/test sets
    print('train')
    x_train = tokenize_and_vectorize_1dlist(df_train['text'], w2v_model.wv)
    x_train = pad_trunc(x_train, maxlen)
    x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
    y_train = np.array(df_train['hs'])
    print('dev')
    x_dev = tokenize_and_vectorize_1dlist(df_dev['text'], w2v_model.wv)
    x_dev = pad_trunc(x_dev, maxlen)
    x_dev = np.reshape(x_dev, (len(x_dev), maxlen, embedding_dims))
    y_dev = np.array(df_dev['hs'])
    print('test')
    x_test = tokenize_and_vectorize_1dlist(df_test['text'], w2v_model.wv)
    x_test = pad_trunc(x_test, maxlen)
    x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
    y_test = np.array(df_test['hs'])
    print('done')

    # %% Model choice
    model_name_simple = 'CNN'

    # Define save path
    metrics_path_category = '/home/pgajo/working/data/metrics/1_hate_speech'
    # metrics_path_category = '/home/pgajo/working/data/metrics/2_racism+misogyny'
    # metrics_path_category = '/home/pgajo/working/data/metrics/3_hate_forecasting'

    metrics_save_path = f'{metrics_path_category}/metrics_monolingual/'
    metrics_save_path_model = os.path.join(metrics_save_path, model_name_simple)

    if not os.path.exists(metrics_save_path_model):
        os.mkdir(metrics_save_path_model)

    metrics_filename = str(metrics_id)+'_'+model_name_simple + \
        '_'+time_str+'_metrics.csv'
    metrics_csv_filepath = os.path.join(metrics_save_path_model, metrics_filename)
    print(metrics_csv_filepath)

    threshold = 0.5

    # # Defining the hyperparameters to explore, based on the CNN from before

    # # Load the hyperparameters from file
    # with open(r'C:\Users\Paolo\My Drive\UNI_Google_Drive\NLP_Google_Drive\incels_2022-2023\best_models\CNN\hate_speech\2023-02-16_best_hyperparameters_CNN_incelsis.json', 'r') as f:
    #     best_hps = json.load(f)
    #     # best_hps=best_hps['values']
    # print(*["{}: {}".format(k, v) for k, v in best_hps.items()], sep="\n")


    # Define hyperparameter object
    hp = kt.HyperParameters() # Hyperparameters object

    # Hyperband is one of the optimization algorithms provided by Keras Tuner

    tuner = Hyperband(build_hypermodel,  # change this if you are using new tuned hyperparameters or if you're loading from json
                    # Objective to maximize
                    objective=kt.Objective('val_acc', direction='max'),
                    executions_per_trial=5,  # Number of models that should be built and fit for each trial
                    hyperband_iterations=1,  # The number of times the Hyperband algorithm is iterated over
                    max_epochs=10,
                    directory=os.path.normpath(date_str+'kt'),
                    project_name="keras_tuner_project",
                    overwrite=True)

    tuner.search(x_train, y_train, batch_size=128,  # Batch size -> another parameter that can be explored
                epochs=10,
                validation_data=(x_dev, y_dev),
                # Patience -> number of epochs with no improvement after which training will be stopped; I set it a bit higher to explore more configurations
                callbacks=[EarlyStopping('val_loss', patience=3)],
                verbose=2)


    # Train CNN model with optimal HPs

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]

    # Get the best model
    best_model = tuner.get_best_models(1)[0]

    # Show model summary
    best_model.summary()


    # Training loop

    # write set identifiers for the pandas metrics dataframe
    df_metrics_train_set_string = ''
    for i, index in enumerate(df_train['data_type'].value_counts(normalize=False).index.to_list()):
        set_len = df_train['data_type'].value_counts(normalize=False).values[i]
        df_metrics_train_set_string += index+'('+str(set_len)+')'+'\n'

    df_metrics_dev_set_string = ''
    for i, index in enumerate(df_dev['data_type'].value_counts(normalize=False).index.to_list()):
        set_len = df_dev['data_type'].value_counts(normalize=False).values[i]
        df_metrics_dev_set_string += index+'('+str(set_len)+')'+'\n'

    df_metrics_test_set_string = ''
    for i, index in enumerate(df_test['data_type'].value_counts(normalize=False).index.to_list()):
        set_len = df_test['data_type'].value_counts(normalize=False).values[i]
        df_metrics_test_set_string += index+'('+str(set_len)+')'+'\n'

    # train
    print('Run ID:', metrics_id)
    print('Train sets:')
    print(df_train['data_type'].value_counts(normalize=False))
    print('Train set length', len(df_train), '\n')
    print('Dev sets:')
    print(df_dev['data_type'].value_counts(normalize=False))
    print('Train set length', len(df_dev), '\n')
    print('Test sets:')
    print(df_test['data_type'].value_counts(normalize=False))
    print('Train set length', len(df_dev), '\n')

    df_metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'val_f1', 'val_prec',
                            'val_rec', 'test_loss', 'test_f1', 'test_prec', 'test_rec'])

    epochs = 3

    # Training loop
    for i in range(5):  # run the loop n times

        model = tuner.hypermodel.build(best_hps)

        # Train the model and get the history object
        history = model.fit(x_train, y_train,
                            validation_data=(x_dev, y_dev),
                            epochs=epochs,
                            batch_size=128,
                            verbose=1)

        # Get the validation loss for each epoch
        train_losses = history.history['loss']
        train_loss = train_losses[-1]
        val_losses = history.history['val_loss']
        val_loss = val_losses[-1]

        # Compute dev metrics
        pred_dev = [el[0] for el in model.predict(x_dev)]
        pred_dev = [(el > threshold).astype("int32") for el in pred_dev]

        # Calculate val_loss and dev_accuracy
        dev_metrics = model.evaluate(x_dev, y_dev, verbose=0)
        val_loss, dev_accuracy, *_ = dev_metrics

        val_f1 = f1_score(y_dev, pred_dev, average='binary', pos_label=1)
        val_prec = precision_score(y_dev, pred_dev)
        val_rec = recall_score(y_dev, pred_dev)

        # Compute test metrics
        pred_test = [el[0] for el in model.predict(x_test)]
        pred_test = [(el > threshold).astype("int32") for el in pred_test]

        # Calculate val_loss and dev_accuracy
        test_metrics = model.evaluate(x_test, y_test, verbose=0)
        test_loss, test_accuracy, *_ = test_metrics

        test_f1 = f1_score(y_test, pred_test, average='binary', pos_label=1)
        test_prec = precision_score(y_test, pred_test)
        test_rec = recall_score(y_test, pred_test)

        df_metrics = df_metrics.append({
            'epoch': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_prec': val_prec,
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

        clear_output(wait=True)

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

        display(df_metrics)

    # Calculate the average of the statistics over the 5 training iterations
    average_metrics = df_metrics.loc[:, 'train_loss':'test_rec'].mean(axis=0)

    # Create a new row with the average statistics
    average_row = pd.DataFrame(average_metrics).transpose()

    # Set the 'epoch' value to 'average' for the new row
    average_row['epoch'] = 'average'

    # Copy non-numeric columns from the last row of df_metrics to the average_row
    non_numeric_columns = ['model', 'train_len',
                        'train_set(s)', 'dev_set(s)', 'test_set(s)', 'run_id']
    average_row[non_numeric_columns] = df_metrics.loc[df_metrics.index[-1],
                                                    non_numeric_columns]

    # Append the average row to the df_metrics DataFrame
    # df_metrics = df_metrics.append(average_row, ignore_index=True)

    # Display the updated df_metrics DataFrame with the average row
    display(df_metrics)

    print(metrics_csv_filepath)

    # Save the DataFrame to a CSV file
    df_metrics.to_csv(metrics_csv_filepath, index=False)

    # Save HPs to json

    with open('_'.join(metrics_csv_filepath.split('_')[:-1])+'_hp.json', 'w') as f:
        json.dump(tuner.get_best_hyperparameters(1)[0].get_config(), f)





