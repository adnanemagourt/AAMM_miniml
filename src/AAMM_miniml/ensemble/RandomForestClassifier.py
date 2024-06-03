import sys
sys.path.append("..")
from tree.DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np
from collections import Counter

import sys
sys.path.append('..')
from _BaseClasses import Classifier, MetaEstimator


class RandomForestClassifier(MetaEstimator, Classifier):
    """
    Random forest classifier.
    
    Parameters
    ----------
    
    n_estimators : int, default=100
        The number of trees in the forest.
    max_features : {'sqrt', 'log2', int, float}, default='sqrt' 
        The number of features to consider when looking for the best split:
        - If 'sqrt', then max_features=sqrt(n_features).
        - If 'log2', then max_features=log2(n_features).
        - If int, then max_features=n.
        - If float, then max_features=feature_fraction * n.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    trees : list
        A list of fitted trees.
    """
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.trees = []

    def _sample(self, X, y):
        n_samples = X.shape[0]
        if self.bootstrap:
            indices = np.random.choice(n_samples, n_samples, replace=True)
        else:
            indices = np.arange(n_samples)
        return X[indices], y[indices]

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features

    def fit(self, X, y):
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._sample(X, y)

            # Randomly select features
            features_indices = np.random.choice(n_features, max_features, replace=False)
            tree.fit(X_sample[:, features_indices], y_sample.reshape(-1, 1))

            self.trees.append((tree, features_indices))

    def _predict_tree(self, tree, features_indices, X):
        return tree.predict(X[:, features_indices])

    def predict(self, X):
        tree_preds = np.array([self._predict_tree(tree, features_indices, X) for tree, features_indices in self.trees])

        # Initialize a list to store the final predicted values
        y_pred = []

        # Perform voting for each sample
        for sample_preds in tree_preds.T:  # Transpose to iterate over predictions for each sample
            # Count occurrences of each prediction
            counts = Counter(sample_preds)
            # Get the most common prediction (mode)
            mode = counts.most_common(1)[0][0]
            # Append the mode to the final predictions
            y_pred.append(mode)

        return np.array(y_pred)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
