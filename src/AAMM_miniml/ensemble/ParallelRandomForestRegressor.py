import numpy as np
from joblib import Parallel, delayed
import sys
sys.path.append("..")
from tree.DecisionTreeRegressor import DecisionTreeRegressor

import sys
sys.path.append('..')
from _BaseClasses import Regressor, MetaEstimator


class ParallelRandomForestRegressor(MetaEstimator, Regressor):
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.trees = []

    def _sample(self, X, y):
        n_samples = X.shape[0]
        if self.bootstrap:
            indices = np.random.choice(n_samples, n_samples, replace=True)
        else:
            indices = np.arange(n_samples)
        return X[indices], y[indices]

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features

    def _fit_tree(self, X, y, features_indices):
        tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        X_sample, y_sample = self._sample(X, y)
        tree.fit(X_sample[:, features_indices], y_sample)
        return tree, features_indices

    def fit(self, X, y):
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)

        # Train trees in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_tree)(X, y, np.random.choice(n_features, max_features, replace=False))
            for _ in range(self.n_estimators)
        )

        self.trees = results

    def _predict_tree(self, tree, features_indices, X):
        return tree.predict(X[:, features_indices])

    def predict(self, X):
        tree_preds = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_tree)(tree, features_indices, X)
            for tree, features_indices in self.trees
        )
        tree_preds = np.array(tree_preds)  # Shape (n_estimators, n_samples)
        y_pred = np.mean(tree_preds, axis=0)
        return y_pred
    
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)



