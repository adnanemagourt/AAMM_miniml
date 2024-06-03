import sys
sys.path.append('..')
import numpy as np
from sklearn.tree import DecisionTreeRegressor

import sys
sys.path.append('..')
from _BaseClasses import Regressor, MetaEstimator

class GradientBoosting(Regressor, MetaEstimator):
    """
    Gradient boosting for regression.

    Parameters
    ----------

    n_estimators : int, default=100
        The number of boosting stages.
    learning_rate : float, default=0.1
        The learning rate.
    max_depth : int, default=3
        The maximum depth of the individual trees.
    trees : list
        A list of fitted trees.
    initial_prediction : float
        The initial prediction.

    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initial_prediction = np.mean(y)
        residuals = y - self.initial_prediction

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
    
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)
