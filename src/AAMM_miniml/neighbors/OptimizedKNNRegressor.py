import numpy as np
from scipy.spatial import KDTree
import sys
sys.path.append('..')
from _BaseClasses import Regressor

class OptimizedKNNRegressor(Regressor):
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.tree = None
        self.y_train = None

    def fit(self, X, y):
        if self.metric not in ['euclidean', 'manhattan']:
            raise ValueError("Unsupported metric. Choose 'euclidean' or 'manhattan'.")
        self.tree = KDTree(X)
        self.y_train = y

    def _compute_distances(self, X):
        if self.metric == 'euclidean':
            distances, indices = self.tree.query(X, k=self.n_neighbors)
        elif self.metric == 'manhattan':
            distances, indices = self.tree.query(X, k=self.n_neighbors)
        return distances, indices

    def predict(self, X):
        _, neighbors_indices = self._compute_distances(X)
        neighbors_labels = self.y_train[neighbors_indices]
        
        y_pred = np.mean(neighbors_labels, axis=1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)