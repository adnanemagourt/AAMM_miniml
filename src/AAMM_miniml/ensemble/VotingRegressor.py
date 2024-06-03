from statistics import median

import sys
sys.path.append('..')
from _BaseClasses import Regressor, MetaEstimator

class VotingRegressor(MetaEstimator, Regressor):
    """
    A voting regressor that aggregates the predictions of multiple regressors.
    
    Parameters
    ----------
    
    estimators : list
        A list of tuples where each tuple contains a name and a regressor.
    voting : {'hard', 'soft'}, default='hard'
        The method to use for aggregation.
        - 'hard': uses the majority rule voting.
        - 'soft': predicts the class label based
                  on the argmax of the sums of the predicted probabilities.
    weights : list, default=None
        Optional weights to apply to the predictions.
    
    """
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X_train, y_train):
        for name, regressor in self.estimators:
            regressor.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = []
        for name, regressor in self.estimators:
            predictions.append(regressor.predict(X_test))
        
        if self.voting == 'hard':
            final_predictions = median(predictions)
        elif self.voting == 'soft':
            if self.weights is None:
                self.weights = [1] * len(self.estimators)  # Use equal weights if not specified
            weighted_predictions = [w * pred for w, pred in zip(self.weights, predictions)]
            final_predictions = sum(weighted_predictions) / sum(self.weights)

        return final_predictions
    
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)