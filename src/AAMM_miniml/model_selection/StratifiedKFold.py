import numpy as np
from collections import defaultdict

import sys
sys.path.append('..')
from _BaseClasses import Splitter

class StratifiedKFold(Splitter):
    """
    Stratified K-Folds cross-validator.
    
    Provides train/test indices to split data in train/test sets.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
    random_state : int, default=None
        Seed for the random number generator.
    
    Attributes
    ----------
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle the data before splitting into batches.
    random_state : int
        Seed for the random number generator.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups=None):
        n_samples = len(y)
        y = np.array(y)
        classes, y_indices = np.unique(y, return_inverse=True)
        
        n_classes = len(classes)
        class_counts = np.bincount(y_indices)
        class_indices = [np.where(y_indices == i)[0] for i in range(n_classes)]

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            for indices in class_indices:
                rng.shuffle(indices)

        fold_counts = np.zeros(self.n_splits, dtype=int)
        fold_indices = defaultdict(list)
        
        for class_idx in range(n_classes):
            fold_sizes = np.full(self.n_splits, class_counts[class_idx] // self.n_splits, dtype=int)
            fold_sizes[:class_counts[class_idx] % self.n_splits] += 1

            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                fold_indices[fold_idx].extend(class_indices[class_idx][start:stop])
                fold_counts[fold_idx] += fold_size
                current = stop
        
        indices = np.arange(n_samples)
        for fold_idx in range(self.n_splits):
            test_idx = np.array(fold_indices[fold_idx])
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx
