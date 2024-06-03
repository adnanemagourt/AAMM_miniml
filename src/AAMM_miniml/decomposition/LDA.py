import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from _BaseClasses import Transformer

class LDA(Transformer):
    def __init__(self, n_components):
        self.n_components = n_components
        self.scalings_ = None
    
    def fit(self, X, y):
        # Compute the mean vectors for each class
        classes = np.unique(y)
        mean_vectors = []
        for cls in classes:
            mean_vectors.append(np.mean(X[y == cls], axis=0))
        
        # Compute the within-class scatter matrix
        S_W = np.zeros((X.shape[1], X.shape[1]))
        for cls, mv in zip(classes, mean_vectors):
            class_scatter = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == cls]:
                row, mv = row.reshape(X.shape[1], 1), mv.reshape(X.shape[1], 1)
                class_scatter += (row - mv).dot((row - mv).T)
            S_W += class_scatter
        
        # Compute the between-class scatter matrix
        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == classes[i], :].shape[0]
            mean_vec = mean_vec.reshape(X.shape[1], 1)
            overall_mean = overall_mean.reshape(X.shape[1], 1)
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        
        # Solve the eigenvalue problem for the matrix S_W^-1 S_B
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        
        # Sort eigenvectors by eigenvalues in descending order
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        
        # Select the top n_components eigenvectors (scalings)
        self.scalings_ = np.hstack([eig_pairs[i][1].reshape(X.shape[1], 1) for i in range(self.n_components)])
    
    def transform(self, X):
        return np.dot(X, self.scalings_)

# Example usage
if __name__ == "__main__":
    # Sample dataset with class labels
    data = {
        'Feature1': [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
        'Feature2': [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9],
        'Class': [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    X = df[['Feature1', 'Feature2']].values
    y = df['Class'].values
    
    # Create LDA instance
    lda = LDA(n_components=1)
    
    # Fit LDA to the data
    lda.fit(X, y)
    
    # Transform the data
    X_reduced_lda = lda.transform(X)
    
    print("LDA Reduced DataFrame:\n", X_reduced_lda)