# Information Gain

import numpy      as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

# Load parameters from config.csv
def config():
    config_params = pd.read_csv('config.csv', header=None)
    params = {
        'embedding_dim': int(config_params.iloc[0, 0]),
        'tau': int(config_params.iloc[1, 0]),           
        'num_symbols': int(config_params.iloc[2, 0]),   
        'top_k_relevantes': int(config_params.iloc[3, 0]),
        'sigma': float(config_params.iloc[4, 0]),
        'top_k_menos_redundantes': int(config_params.iloc[5, 0])
    }
    return params

def plot_information_gain(ig_scores, max_points=100):
    indices = [x[0] for x in ig_scores][:max_points]
    ig_values = [x[1] for x in ig_scores][:max_points]
    
    plt.figure(figsize=(10, 6))
    plt.bar(indices, ig_values, color='skyblue')
    plt.xlabel('Number of variable')
    plt.ylabel('Inform. Gain')
    plt.title('Ganancia Información')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()    

def select_top_k_variables(features, ig_scores, K):
    ig_scores_sorted = sorted(ig_scores, key=lambda x: x[1], reverse=True)
    top_k_indices = [x[0] for x in ig_scores_sorted[:K]]
    features_top_k = features[:, top_k_indices]
    return features_top_k, top_k_indices

def calculate_Hyx(features, labels, m, tau, c, B):
    Hyx = 0
    N = len(features)
    bin_edges = np.linspace(np.min(features), np.max(features), B + 1)
    for j in range(B):
        indices_j = np.where((features >= bin_edges[j]) & (features < bin_edges[j + 1]))[0]
        if len(indices_j) > 0:
            dj_samples = labels[indices_j]
            DE_dj = entropy_disp(dj_samples, m, tau, c)
            Hyx += (len(dj_samples) / N) * DE_dj
    return Hyx

def calculate_final_DE(probabilities, c, m):
    non_zero_probs = probabilities[probabilities > 0]
    DE = -np.sum(non_zero_probs * np.log2(non_zero_probs + 1e-10))
    r = c ** m
    nDE = DE / np.log2(r)
    return DE, nDE

def calculate_probabilities(frequencies, N, m, tau):
    total_occurrences = N - (m - 1) * tau
    if total_occurrences == 0:
        total_occurrences = 1
    return frequencies / total_occurrences

def calculate_frequencies(patterns, num_patterns):
    frequencies = np.zeros(num_patterns)
    unique, counts = np.unique(patterns, return_counts=True)
    for u, count in zip(unique, counts):
        frequencies[u - 1] = count
    return frequencies

def convert_to_patterns(symbols, c):
    powers = np.array([c ** i for i in range(symbols.shape[1])])
    return 1 + np.dot(symbols, powers)

def map_to_symbols(embedding_vectors, c):
    max_value = c - 1 
    if len(embedding_vectors.shape) == 1:
        embedding_vectors = embedding_vectors.reshape(-1, 1)
    symbols = np.round(embedding_vectors * max_value + 0.5).astype(int)
    # print(f"symbols.shape: {symbols.shape}")
    return np.clip(symbols, 0, max_value)

def create_embedding_vectors(data, m, tau):
    M = len(data) - (m - 1) * tau
    vectors_array = np.array([data[i:i + m * tau:tau] for i in range(M)])
    # print(vectors_array.ndim)
    return vectors_array

# Normalised by use sigmoidal
def norm_data_sigmoidal(data):
    mu = np.mean(data)
    sigma = np.std(data)
    normalized_data = 1 / (1 + np.exp(-(data - mu) / (sigma + 1e-10)))
    return normalized_data

# Dispersion entropy
def entropy_disp(data, m, tau, c):
    normalized_data = norm_data_sigmoidal(data)
    # with np.printoptions(threshold=np.inf):
    #     print(data)
    embedding_vectors = create_embedding_vectors(normalized_data, m, tau)
    symbols = map_to_symbols(embedding_vectors, c)
    patterns = convert_to_patterns(symbols, c)
    num_patterns = c ** m
    frequencies = calculate_frequencies(patterns, num_patterns)
    probabilities = calculate_probabilities(frequencies, len(data), m, tau)
    DE, nDE = calculate_final_DE(probabilities, c, m)
    return DE

def calculate_Hy(labels, m, tau, c):
    return entropy_disp(labels, m, tau, c)

#Information gain
def inform_gain(features, labels, m, tau, c):    
    Hy = calculate_Hy(labels, m, tau, c)
    # print(str(Hy)+" hy")
    ig_scores = []
    B = int(np.sqrt(len(features)))
    for i in range(features.shape[1]):
        Hyx = calculate_Hyx(features[:, i], labels, m, tau, c, B)
        # print(str(Hyx)+" hyx")
        ig = Hy - Hyx 
        # print(str(ig)+" ig")
        ig_scores.append((i, ig))
    return ig_scores

# Beginning ...
def main():
    start_time = time.time()
    params = config()

    output_directory = './output'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    m = params['embedding_dim']
    tau = params['tau']
    c = params['num_symbols']
    top_k_relevantes = params['top_k_relevantes']

    # print(m,tau,c,top_k_relevantes)
    
    data = pd.read_csv('./output/DataClass.csv', sep=',', header=None)
    # print(data.head(15))
    features = data.iloc[:, :-1].to_numpy() #Características
    labels = data.iloc[:, -1].to_numpy() #Clases
    # print("Características (features):", features.shape)
    # print("Etiquetas (labels):", labels.shape)  
    
    ig_scores = inform_gain(features, labels, m, tau, c)
    features_top_k, top_k_indices = select_top_k_variables(features, ig_scores, top_k_relevantes)

    # with np.printoptions(threshold=np.inf):
    #     print(features_top_k)
    
    pd.DataFrame(top_k_indices).to_csv('./output/Idx_variable.csv', header=False, index=False)
    pd.DataFrame(features_top_k).to_csv('./output/DataIG.csv', header=False, index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    plot_information_gain(ig_scores)
    
    print(f"Total Runtime ig.py: {elapsed_time:.2f} seconds")

       
if __name__ == '__main__':   
	 main()

