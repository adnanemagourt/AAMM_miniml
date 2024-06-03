import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as shuffle_arrays

import sys
sys.path.append('..')
from _BaseClasses import Splitter

class StratifiedShuffleSplit(Splitter):
    def __init__(self, n_splits=10, test_size=None, train_size=None, random_state=None):
        """
        Stratified ShuffleSplit cross-validator.

        Parameters:
        n_splits : int, default=10
            Number of re-shuffling & splitting iterations.

        test_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.1.

        train_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
            If int, represents the absolute number of train samples.
            If None, the value is automatically set to the complement of the test size.

        random_state : int, RandomState instance, or None, default=None
            Controls the randomness of the training and testing indices produced.
            Pass an int for reproducible output across multiple function calls.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, X, y, groups=None):
        """
        Generate indices to split data into train and test sets.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        Yields:
        train_index : array-like
            The training set indices for that split.

        test_index : array-like
            The testing set indices for that split.
        """
        rng = check_random_state(self.random_state)
        n_samples = X.shape[0]

        if self.test_size is None and self.train_size is None:
            self.test_size = 0.1

        if self.test_size is None:
            if isinstance(self.train_size, float):
                self.test_size = 1.0 - self.train_size
            else:
                self.test_size = 0.1

        if isinstance(self.test_size, float):
            test_size = int(self.test_size * n_samples)
        else:
            test_size = self.test_size

        if isinstance(self.train_size, float):
            train_size = int(self.train_size * n_samples)
        else:
            train_size = n_samples - test_size

        for _ in range(self.n_splits):
            indices = np.arange(n_samples)
            indices = shuffle_arrays(indices, random_state=rng)

            train_index = indices[:train_size]
            test_index = indices[train_size:(train_size + test_size)]

            yield train_index, test_index


    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns:
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

