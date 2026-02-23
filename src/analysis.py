import numpy as np
from sklearn.decomposition import PCA

def run_pca(X, n_components=60):
    """
    Fit PCA on matrix X and return:
    - fitted PCA model
    - transformed data
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return pca, X_reduced
