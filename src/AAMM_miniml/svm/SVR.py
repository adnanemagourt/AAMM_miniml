import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Regressor

class SVR(Regressor):
    """A Support Vector Regressor.
    
    Parameters
    ----------
    learning_rate : float, default=0.001
        The learning rate.
    lambda_param : float, default=0.01
        The regularization parameter.
    n_iters : int, default=1000
        The number of iterations.
    epsilon : float, default=0.1
        The epsilon-tube within which no penalty is associated in the training loss function.
        
    Attributes
    ----------
    lr : float
        The learning rate.
    lambda_param : float
        The regularization parameter.
    n_iters : int
        The number of iterations.
    epsilon : float
        The epsilon-tube within which no penalty is associated in the training loss function.
    w : ndarray
        The weights.
    b : float
        The bias.
        
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, epsilon=0.1):

        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                prediction = np.dot(x_i, self.w) - self.b
                error = y[idx] - prediction

                if abs(error) <= self.epsilon:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.sign(error) * x_i)
                    self.b -= self.lr * (-np.sign(error))

    def predict(self, X):
        return np.dot(X, self.w) - self.b

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)


