import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Regressor

class DecisionStumpRegressor(Regressor):
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        m, n = X.shape
        best_error = float('inf')

        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                left_value = np.mean(y[left_mask])
                right_value = np.mean(y[right_mask])

                predictions = np.where(left_mask, left_value, right_value)
                error = np.mean((predictions - y) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, X):
        left_mask = X[:, self.feature_index] <= self.threshold
        predictions = np.where(left_mask, self.left_value, self.right_value)
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)
