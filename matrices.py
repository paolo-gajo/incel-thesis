import time
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv


# make list
# Specify the directory where the files (countries) are located
directory = "/home/pgajo/working/country_laws"

# Create an empty list to store the names of the remaining files (countries)
remaining_files = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        country_name = os.path.splitext(filename)[0]  # Remove the file extension
        print(country_name)
        remaining_files.append(country_name)

# Load the Universal Sentence Encoder model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

# Define a function to calculate the similarity scores
def calculate_similarity(sentence1, sentence2):
    if pd.isnull(sentence1) or pd.isnull(sentence2):
        return 0.0
    embeddings = model([sentence1, sentence2])
    return tf.squeeze(tf.matmul(tf.expand_dims(embeddings[0], 0), tf.expand_dims(embeddings[1], 1))).numpy()

# Define a function to create heatmap visualization for a pair of countries
def create_heatmap(country1, country2, output_path=None):
    # Read and process the CSV files for each country
    c1 = pd.read_csv(os.path.join(directory,f"{country1}.csv"))
    c2 = pd.read_csv(os.path.join(directory,f"{country2}.csv"))
    c1 = c1.dropna()
    c2 = c2.dropna()

    # Concatenate the datasets into a single DataFrame
    df = pd.concat([c1, c2])
    df = df.reset_index()

    # Create an empty similarity matrix
    num_sentences = len(df)
    similarity_matrix = np.zeros((num_sentences, num_sentences))

    # Calculate the similarity scores and populate the matrix
    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            sentence1 = df['Legislation'][i]
            sentence2 = df['Legislation'][j]
            similarity_score = calculate_similarity(sentence1, sentence2)
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score

    # Set the Seaborn style
    sns.set()

    # Create a heatmap visualization of the similarity matrix
    plt.figure()
    heatmap = sns.heatmap(similarity_matrix, cmap='plasma', fmt=".2f", cbar=True)
    heatmap.set_title(f'Sentence Similarity Heatmap: {country1} vs {country2}')
    heatmap.set_xlabel('Sentences')
    heatmap.set_ylabel('Sentences')

    # Save the plot if output path is provided
    if output_path:
        plt.savefig(output_path)

    plt.show()

# Specify the list of countries
country_list = remaining_files

# Specify the output directory
output_directory = "/home/pgajo/working/heatmap_plots/"

# Measure the execution time
start_time = time.time()

# Create heatmaps for each pair of countries
for i in range(len(country_list)):
    for j in range(i+1, len(country_list)):
        country1 = country_list[i]
        country2 = country_list[j]
        output_path = f"{output_directory}{country1}_{country2}_heatmap.png"
        create_heatmap(country1, country2, output_path)

# Calculate and print the execution time
execution_time = time.time() - start_time
print(f"Execution time: {execution_time} seconds")