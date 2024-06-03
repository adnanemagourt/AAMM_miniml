import numpy as np
from collections import Counter
from scipy.spatial import KDTree
import sys
sys.path.append('..')
from _BaseClasses import Classifier

class OptimizedKNNClassifier(Classifier):
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
            distances, indices = self.tree.query(X, k=self.n_neighbors, p=1)
        return distances, indices

    def predict(self, X):
        _, neighbors_indices = self._compute_distances(X)
        neighbors_labels = self.y_train[neighbors_indices]

        y_pred = np.array([Counter(neighbors).most_common(1)[0][0] for neighbors in neighbors_labels])
        return y_pred

    def predict_proba(self, X):
        _, neighbors_indices = self._compute_distances(X)
        neighbors_labels = self.y_train[neighbors_indices]

        proba = []
        unique_labels = np.unique(self.y_train)
        for neighbors in neighbors_labels:
            counts = Counter(neighbors)
            proba.append([counts.get(label, 0) / self.n_neighbors for label in unique_labels])
        return np.array(proba)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)