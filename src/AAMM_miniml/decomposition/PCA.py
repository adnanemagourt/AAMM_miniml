import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Transformer

class PCA(Transformer):
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
    
    def fit(self, X):
        # Step 1: Standardize the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Step 3: Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Step 4: Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Step 5: Select the top k eigenvectors
        self.components_ = eigenvectors[:, :self.n_components]
    
    def transform(self, X):
        # Step 1: Standardize the data
        X_centered = X - self.mean_
        
        # Step 2: Project the data onto the principal components
        return np.dot(X_centered, self.components_)

# Example usage
if __name__ == "__main__":
    # Sample dataset
    X = np.array([[2.5, 2.4],
                  [0.5, 0.7],
                  [2.2, 2.9],
                  [1.9, 2.2],
                  [3.1, 3.0],
                  [2.3, 2.7],
                  [2.0, 1.6],
                  [1.0, 1.1],
                  [1.5, 1.6],
                  [1.1, 0.9]])
    
    # Create PCA instance
    pca = PCA(n_components=1)
    
    # Fit PCA to the data
    pca.fit(X)
    
    # Transform the data
    X_reduced = pca.transform(X)
    
    print("Reduced Data:\n", X_reduced)