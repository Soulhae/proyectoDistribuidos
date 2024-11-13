# Kernel-PCA by use Gaussian function

import numpy as np
import pandas as pd
import os
import time
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

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

# Gaussian Kernel
def kernel_gauss(data, sigma):
    pairwise_sq_dists = squareform(pdist(data, 'sqeuclidean'))
    K = np.exp(-pairwise_sq_dists / (2 * sigma**2))
    return K
#Kernel-PCA
def kpca_gauss(data, sigma, k):
    K = kernel_gauss(data, sigma)

    N = K.shape[0]
    one_N = np.ones((N, N)) / N
    K_centered = K - one_N @ K - K @ one_N + one_N @ K @ one_N

    eigvals, eigvecs = eigh(K_centered)

    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvecs_top_k = eigvecs[:, :k]
    projected_data = K_centered @ eigvecs_top_k

    return projected_data
# 
def load_data():
    data = pd.read_csv('./output/DataIG.csv', sep=',', header=None)
    return data
# Beginning ...
def main():
    start_time = time.time()
    params = config()
    sigma = params['sigma']
    k = params['top_k_menos_redundantes']
    #print(sigma, k)

    output_directory = './output'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)			
    
    data = load_data().values[:3000]

    projected_data = kpca_gauss(data, sigma, k)

    pd.DataFrame(projected_data).to_csv('./output/DataKPCA.csv', header=False, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Total Runtime kpca.py: {elapsed_time:.2f} seconds")		

if __name__ == '__main__':   
	 main()

