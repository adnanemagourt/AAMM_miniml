import numpy as np
from collections import Counter
import sys
sys.path.append('..')
from _BaseClasses import Classifier

class KNNClassifier(Classifier):
    """
    K-nearest neighbors classifier.
    
    Parameters
    ----------
    
    n_neighbors : int, default=5
        Number of neighbors to use by default.
    metric : {'euclidean', 'manhattan'}, default='euclidean'
        The distance metric to use
    
    Attributes
    ----------
    
    X_train : ndarray
        The training input samples.
    y_train : ndarray
        The target values.
    n_neighbors : int
        Number of neighbors to use by default.
    metric : {'euclidean', 'manhattan'}
        The distance metric to use.
    
    """
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, X):
        if self.metric == 'euclidean':
            distances = np.sqrt(((self.X_train - X[:, np.newaxis]) ** 2).sum(axis=2))
        elif self.metric == 'manhattan':
            distances = np.abs(self.X_train - X[:, np.newaxis]).sum(axis=2)
        else:
            raise ValueError("Unsupported metric. Choose 'euclidean' or 'manhattan'.")
        return distances

    def predict(self, X):
        distances = self._compute_distances(X)
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbors_labels = self.y_train[neighbors_indices]
        
        y_pred = np.array([Counter(neighbors).most_common(1)[0][0] for neighbors in neighbors_labels])
        return y_pred

    def predict_proba(self, X):
        distances = self._compute_distances(X)
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbors_labels = self.y_train[neighbors_indices]

        proba = []
        unique_labels = np.unique(self.y_train)
        for neighbors in neighbors_labels:
            counts = Counter(neighbors)
            proba.append([counts.get(label, 0) / self.n_neighbors for label in unique_labels])
        return np.array(proba)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)