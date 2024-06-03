import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import MetaEstimator, Classifier

class AdaBoost(MetaEstimator, Classifier):
    def __init__(self, weak_classifier, n_classifiers):
        """
        Parameters
        ----------
        weak_classifier : class
            The weak classifier to be used in the ensemble. It should be a class that inherits from Classifier.
        n_classifiers : int
            The number of weak classifiers to be used in the ensemble.
        
        Attributes
        ----------
        weak_classifier : class
            The weak classifier to be used in the ensemble. It should be a class that inherits from Classifier. 
        n_classifiers : int
            The number of weak classifiers to be used in the ensemble.
        alphas : list
            The list of weights for each weak classifier.
        classifiers : list
            The list of weak classifiers.
        weights : ndarray
            The weights of the samples.
            """
        self.weak_classifier = weak_classifier
        self.n_classifiers = n_classifiers

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alphas = []
        self.classifiers = []
        self.weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_classifiers):
            clf = self.weak_classifier(max_depth=1)  # Using a decision stump as a weak classifier
            clf.fit(X, y, sample_weight=self.weights)
            y_pred = clf.predict(X)

            # Compute the error
            error = np.sum(self.weights * (y_pred != y)) / np.sum(self.weights)

            if error > 0.5:  # Skip this classifier if it's worse than random guessing
                continue

            # Compute alpha
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update weights
            self.weights *= np.exp(-alpha * y * y_pred)
            self.weights /= np.sum(self.weights)

            # Save classifier and alpha
            self.classifiers.append(clf)
            self.alphas.append(alpha)

    def predict(self, X):
        clf_preds = [alpha * clf.predict(X) for alpha, clf in zip(self.alphas, self.classifiers)]
        return np.sign(np.sum(clf_preds, axis=0))
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

