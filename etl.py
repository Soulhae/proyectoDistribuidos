#-------------------------------------------------------
# Load Data from File: KDDTrain.txt
#--------------------------------------------------------

import numpy as np
import pandas as pd
import random
import time
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

def preprocess_data(data):
    categorical_cols = [1, 2, 3]
    for col in categorical_cols:
        data[col] = pd.factorize(data[col])[0]
    
    dos_attacks = {'neptune', 'teardrop', 'smurf', 'pod', 'back', 'land', 
                   'apache2', 'processtable', 'mailbomb', 'udpstorm'}
    probe_attacks = {'ipsweep', 'portsweep', 'nmap', 'satan', 'saint', 'mscan'}

    data.iloc[:, -2] = data.iloc[:, -2].apply(
        lambda x: 1 if x == 'normal' else
                  2 if x in dos_attacks else
                  3 if x in probe_attacks else 0
    )

    data = data.iloc[:, :-2].join(data.iloc[:, -1]).join(data.iloc[:, -2])

    return data

def select_samples_by_index(class_file, idx_file):
    class_data = pd.read_csv(class_file, header=None)
    
    indices = pd.read_csv(idx_file, header=None).iloc[:, 0]
    
    num_rows = len(class_data)
    
    valid_indices = indices[indices < num_rows].astype(int)
    
    selected_samples = class_data.iloc[valid_indices]
    
    return selected_samples

def select_random_samples(class_file, M):
    class_data = pd.read_csv(class_file, header=None)
    
    num_rows = len(class_data)
    M = min(M, num_rows)
    
    selected_samples = class_data.sample(n=M, random_state=42)
    
    return selected_samples
    
# Beginning ...
def main():
    start_time = time.time()
    params = config()

    output_directory = './output'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    raw_data = pd.read_csv('./input/KDDTrain.txt', sep=',', header=None)
    raw_data = preprocess_data(raw_data)
    raw_data.to_csv('./output/Data.csv', index=False, header=False)
    
    class1 = raw_data[raw_data.iloc[:, -1] == 1]
    class2 = raw_data[raw_data.iloc[:, -1] == 2]
    class3 = raw_data[raw_data.iloc[:, -1] == 3]

    class1.to_csv('./output/class1.csv', index=False, header=False)
    class2.to_csv('./output/class2.csv', index=False, header=False)
    class3.to_csv('./output/class3.csv', index=False, header=False)

    class1_file = './output/class1.csv'
    class2_file = './output/class2.csv'
    class3_file = './output/class3.csv'
    
    idx_class1_file = './input/idx_class1.csv'
    idx_class2_file = './input/idx_class2.csv'
    idx_class3_file = './input/idx_class3.csv'
    
    M = random.randint(1500, 2200)
    # print(M)
    
    selected_class1_idx = select_samples_by_index(class1_file, idx_class1_file)
    selected_class2_idx = select_samples_by_index(class2_file, idx_class2_file)
    selected_class3_idx = select_samples_by_index(class3_file, idx_class3_file)
    
    remaining_class1 = M - len(selected_class1_idx)
    remaining_class2 = M - len(selected_class2_idx)
    remaining_class3 = M - len(selected_class3_idx)
    
    if len(selected_class1_idx) < M:
        selected_class1_random = select_random_samples(class1_file, M - len(selected_class1_idx))
        selected_class1 = pd.concat([selected_class1_idx, selected_class1_random], ignore_index=True)
    else:
        selected_class1 = selected_class1_idx.head(M)

    if len(selected_class2_idx) < M:
        selected_class2_random = select_random_samples(class2_file, M - len(selected_class2_idx))
        selected_class2 = pd.concat([selected_class2_idx, selected_class2_random], ignore_index=True)
    else:
        selected_class2 = selected_class2_idx.head(M)

    if len(selected_class3_idx) < M:
        selected_class3_random = select_random_samples(class3_file, M - len(selected_class3_idx))
        selected_class3 = pd.concat([selected_class3_idx, selected_class3_random], ignore_index=True)
    else:
        selected_class3 = selected_class3_idx.head(M)
    
    # print(str(len(selected_class1)) + " Normal")
    # print(str(len(selected_class2)) + " DOS")
    # print(str(len(selected_class3)) + " Probe")
    # print(str(len(selected_class1)+len(selected_class2)+len(selected_class3)) + " Total muestras")

    all_selected_samples = pd.concat([selected_class1, selected_class2, selected_class3], ignore_index=True)
    
    all_selected_samples.to_csv('./output/DataClass.csv', index=False, header=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Total Runtime etl.py: {elapsed_time:.2f} seconds")

if __name__ == '__main__':   
    main()
