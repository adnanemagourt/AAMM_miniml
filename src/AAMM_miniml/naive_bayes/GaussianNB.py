import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Classifier
     
class GaussianNB(Classifier):
    """
    Gaussian Naive Bayes classifier.
    
    Parameters
    ----------
    
    None
    
    Attributes
    ----------
    
    y_classes : ndarray
        The unique classes in the target.
    x_classes : list    
        The unique classes in the features.
    phi_y : ndarray
        The prior probabilities of the classes.
    u : ndarray 
        The means of the features for each class.
    var_x : ndarray
        The variances of the features for each class.
    """
    def fit(self, X, y, epsilon = 1e-10):
        self.y_classes, y_counts = np.unique(y, return_counts=True)
        self.x_classes = np.array([np.unique(x) for x in X.T])
        self.phi_y = 1.0 * y_counts/y_counts.sum()
        self.u = np.array([X[y==k].mean(axis=0) for k in self.y_classes])
        self.var_x = np.array([X[y==k].var(axis=0)  + epsilon for k in self.y_classes])
        return self
    
    def predict(self, X):
        return np.apply_along_axis(lambda x: self.compute_probs(x), 1, X)
    
    def compute_probs(self, x):
        probs = np.array([self.compute_prob(x, y) for y in range(len(self.y_classes))])
        return self.y_classes[np.argmax(probs)]
    
    def compute_prob(self, x, y):
        c = 1.0 /np.sqrt(2.0 * np.pi * (self.var_x[y]))
        return np.prod(c * np.exp(-1.0 * np.square(x - self.u[y]) / (2.0 * self.var_x[y])))
    
    def score(self, X, y):
        return (self.predict(X) == y).mean()