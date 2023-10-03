import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import cuda
from torch.nn import DataParallel
from tqdm import tqdm
import os

# Load the model and tokenizer
model_name = "/home/pgajo/working/pt_models/incel-bert-base-multilingual-cased-1000k_multi_finetuned1_hate_speech_metrics_id_23"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Check for GPU availability and use DataParallel for multiple GPUs
if cuda.is_available():
    device = 'cuda'
    if cuda.device_count() > 1:
        model = DataParallel(model)
else:
    device = 'cpu'

model.to(device)
    
# Function to make predictions
def predict(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs.to(device)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=1).cpu().numpy()
    return predictions

# Read and process the CSV file in chunks
input_csv = "/home/pgajo/working/data/datasets/Italian/Il_forum_dei_brutti/IFC-22-IT_updated.csv"
output_csv = os.path.splitext(input_csv)[0] + "_hs_silver_labels_test" + os.path.splitext(input_csv)[1]
chunksize = 16  # You can adjust this number based on your machine's memory

first_chunk = True
for chunk in tqdm(pd.read_csv(input_csv, chunksize=chunksize, nrows=1000), total=pd.read_csv(input_csv).shape[0]//chunksize):
    chunk.fillna('', inplace=True)
    texts = chunk['text'].tolist()
    hs_predictions = predict(texts, tokenizer, model, device)
    chunk['hs_predictions'] = hs_predictions
    
    # Save the processed chunk to the output CSV file
    if first_chunk:
        chunk.to_csv(output_csv, mode='w', index=False)
        first_chunk = False
    else:
        chunk.to_csv(output_csv, mode='a', index=False, header=False)

print("Predictions saved to", output_csv)
