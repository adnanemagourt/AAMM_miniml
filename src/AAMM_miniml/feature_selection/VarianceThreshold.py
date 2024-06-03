import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Transformer

class VarianceThreshold(Transformer):
    """
    Feature selector that removes all low-variance features.
    
    Parameters
    ----------
    
    threshold : float, default=0.0
        Features with a variance lower than this threshold will be removed.
        
    Attributes
    ----------
    
    variances_ : ndarray
        The variance of each feature.
    n_features_in_ : int
        Number of features in the input data.
    """
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.variances_ = np.var(X, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if not hasattr(self, 'variances_'):
            raise RuntimeError("You must fit the transformer before transforming data")
        support_mask = self.variances_ > self.threshold
        if not np.any(support_mask):
            raise ValueError("No feature in X meets the variance threshold")
        return X[:, support_mask]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        if not hasattr(self, 'variances_'):
            raise RuntimeError("You must fit the transformer before getting support mask")
        support_mask = self.variances_ > self.threshold
        if indices:
            return np.where(support_mask)[0]
        return support_mask

    def get_feature_names_out(self, input_features=None):
        support_mask = self.get_support()
        if input_features is None:
            return np.array([f"x{i}" for i in range(self.n_features_in_)])[support_mask]
        else:
            return np.array(input_features)[support_mask]
