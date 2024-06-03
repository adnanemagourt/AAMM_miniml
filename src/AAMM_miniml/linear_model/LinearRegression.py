import numpy as np
import sys
sys.path.append('..')
from _BaseClasses import Regressor

class LinearRegression(Regressor):
    """
    Linear regression model.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    beta : ndarray
        The regression coefficients.
    """
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        # Step 1: Compute X^T X
        XT_X = np.dot(X.T, X)
        
        # Step 2: Compute X^T y
        XT_y = np.dot(X.T, y)
        
        # Step 3: Compute the inverse of X^T X
        XT_X_inv = np.linalg.inv(XT_X)
        
        # Step 4: Compute beta
        beta = np.dot(XT_X_inv, XT_y)
        
        self.beta = beta
    
    def predict(self, X):
        # Compute predictions
        y_pred = np.dot(X, self.beta)
        
        return y_pred
    
    def score(self, X, y):
        # Compute R^2
        y_pred = self.predict(X)
        SS_res = np.sum((y - y_pred) ** 2)
        SS_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - SS_res / SS_tot
        
        return r2


