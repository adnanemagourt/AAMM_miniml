import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Transformer

class SelectKBest(Transformer):
    """
    Feature selector that selects the k highest scoring features according to a scoring function.
    
    Parameters
    ----------
    
    score_func : callable
        The scoring function. It must take two arrays X and y, and return two arrays of scores and p-values.
    k : int or 'all', default=10
        Number of top features to select. If 'all', select all features.
    
    Attributes
    ----------
    
    scores_ : ndarray
        The scores of each feature.
    pvalues_ : ndarray
        The p-values of each feature.
    n_features_in_ : int
        Number of features in the input data.
    """
    def __init__(self, score_func, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.scores_, self.pvalues_ = self.score_func(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if not hasattr(self, 'scores_'):
            raise RuntimeError("You must fit the transformer before transforming data")
        if self.k == 'all':
            return X
        else:
            indices = np.argsort(self.scores_)[-self.k:]
            return X[:, indices]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        if not hasattr(self, 'scores_'):
            raise RuntimeError("You must fit the transformer before getting support mask")
        support_mask = np.zeros(self.n_features_in_, dtype=bool)
        if self.k == 'all':
            support_mask[:] = True
        else:
            top_indices = np.argsort(self.scores_)[-self.k:]
            support_mask[top_indices] = True
        if indices:
            return np.where(support_mask)[0]
        return support_mask

    def get_feature_names_out(self, input_features=None):
        support_mask = self.get_support()
        if input_features is None:
            return np.array([f"x{i}" for i in range(self.n_features_in_)])[support_mask]
        else:
            return np.array(input_features)[support_mask]
