# Kernel-PCA by use Gaussian function

import numpy as np
import pandas as pd
import time

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
    pairwise_sq_dists = np.sum(data**2, axis=1).reshape(-1, 1) + np.sum(data**2, axis=1) - 2 * np.dot(data, data.T)
    K = np.exp(-pairwise_sq_dists / (2 * sigma**2))
    return K
#Kernel-PCA
def kpca_gauss(data, sigma, k):
    K = kernel_gauss(data, sigma)

    N = K.shape[0]
    one_N = np.ones((N, N)) / N
    K_centered = K - one_N @ K - K @ one_N + one_N @ K @ one_N

    eigvals, eigvecs = np.linalg.eigh(K_centered)

    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvecs_top_k = eigvecs[:, :k]
    projected_data = K_centered @ eigvecs_top_k

    return projected_data
# 
def load_data():
    data = pd.read_csv('./DataIG.csv', sep=',', header=None)
    return data
# Beginning ...
def main():
    start_time = time.time()
    params = config()
    sigma = params['sigma']
    k = params['top_k_menos_redundantes']
    #print(sigma, k)			
    
    data = load_data().values[:3000]
    pd.DataFrame(data).to_csv('./Data.csv', header=False, index=False)

    projected_data = kpca_gauss(data, sigma, k)

    pd.DataFrame(projected_data).to_csv('./DataKPCA.csv', header=False, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Total Runtime kpca.py: {elapsed_time:.2f} seconds")		

if __name__ == '__main__':   
	 main()

