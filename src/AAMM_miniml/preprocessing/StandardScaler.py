import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Transformer

class StandardScaler(Transformer):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    mean_ : ndarray
        The mean of each feature.
    std_ : ndarray
        The standard deviation of each feature.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X):
        # Calculate mean and standard deviation along each feature (column)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        # Check if the scaler has been fitted
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() before transform().")
        
        # Standardize each feature
        X_std = (X - self.mean_) / self.std_
        return X_std
    
    def fit_transform(self, X):
        # Fit to the data and transform it in one step
        return self.fit(X).transform(X)
