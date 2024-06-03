from scipy.stats import mode
import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Classifier, MetaEstimator

class VotingClassifier(MetaEstimator, Classifier):
    def __init__(self, estimators, voting='hard', weights=None):
        """
        Initialize the VotingClassifier.

        Parameters:
        - estimators (list): List of (name, estimator) tuples representing base classifiers.
        - voting (str): Voting strategy ('hard' for hard voting or 'soft' for soft voting).
        - weights (list, optional): Optional weights used in soft voting to combine predictions.
        """
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X_train, y_train):
        """
        Train all base classifiers in the ensemble.

        Parameters:
        - X_train (array-like): Training input samples.
        - y_train (array-like): Target values (class labels) for training.
        """
        for name, estimator in self.estimators:
            estimator.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the ensemble of base classifiers.

        Parameters:
        - X_test (array-like): Test input samples.

        Returns:
        - final_predictions (array-like): Predicted class labels.
        """
        predictions = []
        for name, estimator in self.estimators:
            predictions.append(estimator.predict(X_test))
        
        if self.voting == 'hard':
            # Perform hard voting: choose the mode (majority vote)
            final_predictions = mode(predictions)[0][0]
        elif self.voting == 'soft':
            # Perform soft voting: combine probabilities and select class with highest average probability
            weighted_sum_probs = np.zeros_like(predictions[0])
            for i, prob in enumerate(predictions):
                if self.weights is not None:
                    weighted_prob = prob * self.weights[i]
                else:
                    weighted_prob = prob
                weighted_sum_probs += weighted_prob
            
            final_predictions = np.argmax(weighted_sum_probs, axis=1)

        return final_predictions
    
    def score(self, X, y):
        """
        Compute the accuracy of the ensemble on a test set.

        Parameters:
        - X (array-like): Test input samples.
        - y (array-like): True class labels for the test set.

        Returns:
        - accuracy (float): Accuracy of the ensemble on the test set.
        """
        return np.mean(self.predict(X) == y)

