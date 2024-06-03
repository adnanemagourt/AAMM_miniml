import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Classifier

class DecisionStumpClassifier(Classifier):
    """
    A decision stump is a machine learning model consisting of a one-level decision tree.
    
    Parameters:
    -----------
    None
    
    Attributes:
    -----------
    threshold: float
        The threshold that the feature should be compared with.
    feature_index: int
        The index of the feature that the threshold is compared with.
    polarity: int
        The polarity of the prediction.
    alpha: float
        The weight of the decision stump.   
        """
    def __init__(self):
        """
        Initialize the DecisionStumpClassifier.

        Attributes:
        -----------
        threshold: float
            The threshold that the feature should be compared with.
        feature_index: int
            The index of the feature that the threshold is compared with.
        polarity: int
            The polarity of the prediction.
        alpha: float
            The weight of the decision stump.
        """
        self.threshold = None
        self.feature_index = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the DecisionStumpClassifier to the training data.

        Parameters:
        -----------
        X: numpy.ndarray
            The training data.
        y: numpy.ndarray
            The true labels.
        sample_weight: numpy.ndarray
            The weight of each sample.

        Returns:
        --------
        None
        """
        n_samples, n_features = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        
        min_error = float('inf')

        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                p = 1
                predictions = np.ones(n_samples)
                predictions[X_column < threshold] = -1

                error = np.sum(sample_weight[y != predictions])

                if error > 0.5:
                    error = 1 - error
                    p = -1

                if error < min_error:
                    min_error = error
                    self.polarity = p
                    self.threshold = threshold
                    self.feature_index = feature_index

    def predict(self, X):
        """
        Make predictions using the DecisionStumpClassifier.

        Parameters:
        -----------
        X: numpy.ndarray
            The data to make predictions for.

        Returns:
        --------
        predictions: numpy.ndarray
            The predictions for the data.
        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        predictions[X_column < self.threshold] = -1
        return self.polarity * predictions

    def score(self, X, y):
        """
        Calculate the accuracy of the DecisionStumpClassifier on the given data.

        Parameters:
        -----------
        X: numpy.ndarray
            The data to make predictions for.
        y: numpy.ndarray
            The true labels.

        Returns:
        --------
        score: float
            The accuracy of the DecisionStumpClassifier on the given data.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

