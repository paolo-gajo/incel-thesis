from nltk.tokenize import TreebankWordTokenizer
import nltk
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report
import io

# Method to tokenise and vectorise all the training data
def tokenize_and_vectorize(dataset,word_vectors): #THE SHAPE OF THE DATASET NEEDS TO BE [(0,'text'),(1,'text'),...(0,'text')]
#     tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    zero_vector = []
    for _ in range(len(word_vectors['cat'])):
                zero_vector.append(0.0)
    j=0
    for sample in dataset:
        tokens = nltk.tokenize.casual_tokenize(sample[1].lower())
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass # No matching token in the w2v vocab
        if len(sample_vecs)==0:
            sample_vecs.append(zero_vector)
        vectorized_data.append(sample_vecs)

    return vectorized_data

def tokenize_and_vectorize_1dlist(dataset,word_vectors): #THE SHAPE OF THE DATASET NEEDS TO BE [('text'),('text'),...('text')]
#     tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    zero_vector = []
    for _ in range(len(word_vectors['cat'])):
        zero_vector.append(0.0)
    j=0
    for sample in dataset:
        tokens = nltk.tokenize.casual_tokenize(sample.lower())
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass # No matching token in the w2v vocab
        if len(sample_vecs)==0:
            sample_vecs.append(zero_vector)
        vectorized_data.append(sample_vecs)

    return vectorized_data

# Method to get the target labels
def collect_expected(dataset):
    """ Peel off the target values from the dataset """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected

def collect_expected_1dlist(dataset):
    """ Peel off the target values from the dataset """
    expected = []
    for sample in dataset:
        expected.append(sample)
    return expected

# Method to pad or truncate the input
# (notice that this code is quite verbose)
def pad_trunc(data, maxlen,label=''):
    """
    For a given dataset pad with zero vectors or truncate to maxlen
    """
    new_data = []
    # Create a vector of 0s the length of our word vectors
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in tqdm(data, desc='Padding/Truncating: '+label):
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            # Append the appropriate number 0 vectors to the list
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)

    return new_data

def df_classification_report(y_val, pred_bin_val_class, target_names, digits=2):
    # Generate classification report
    report_val = classification_report(y_val, pred_bin_val_class, target_names=target_names, digits=digits)

    # Convert report to a pandas dataframe
    df_report_val = pd.read_csv(io.StringIO(report_val), sep='\s{2,}', engine='python')

    # Define a new row as a list
    new_row = [None, None, df_report_val.iloc[2, 0], df_report_val.iloc[2, 1]]
    # Select the row with index position 3 (the fourth row) and assign the new row to it
    df_report_val.loc['accuracy'] = new_row
    
    return df_report_val