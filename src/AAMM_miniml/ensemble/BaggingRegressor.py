import numpy as np
from sklearn.base import clone

import sys
sys.path.append('..')
from _BaseClasses import Regressor, MetaEstimator

class BaggingRegressor(MetaEstimator, Regressor):
    """
    Bagging regressor.

    Parameters
    ----------

    base_estimator : object
        The base estimator.
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    random_state : int, default=None
        Seed for the random number generator.
    """
    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            model = clone(self.base_estimator)
            model.fit(X_sample, y_sample)
            self.models.append(model)
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)
    
    def score(self, X, y):
        # error
        return np.mean((self.predict(X) - y) ** 2)

