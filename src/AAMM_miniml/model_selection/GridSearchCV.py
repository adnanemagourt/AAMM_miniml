import numpy as np
import itertools
import copy
from collections import defaultdict

import sys
sys.path.append('..')
from _BaseClasses import MetaEstimator

class GridSearchCV(MetaEstimator):
    """
    Exhaustive search over specified parameter values for an estimator.
    
    Parameters
    ----------
    estimator : object
        The base estimator.
    param_grid : dict
        The dictionary of parameters and their possible values.
    scoring : callable, default=None
        The scoring function.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    refit : bool, default=True
        Refit the best estimator with the entire dataset.
    cv : int, default=5
        Number of folds for cross-validation.
    verbose : int, default=0
        Controls the verbosity.
    pre_dispatch : str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel execution.
    error_score : float, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    return_train_score : bool, default=False
        Whether to include training scores.
    """
    def __init__(self, estimator, param_grid, scoring=None, n_jobs=None, refit=True, cv=5, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def _generate_param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        for v in itertools.product(*values):
            yield dict(zip(keys, v))

    def _split_data(self, X, y):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.cv, n_samples // self.cv, dtype=int)
        fold_sizes[:n_samples % self.cv] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

    def _score(self, estimator, X, y):
        if self.scoring is None:
            return estimator.score(X, y)
        else:
            return self.scoring(estimator, X, y)

    def fit(self, X, y):
        self.cv_results_ = defaultdict(list)
        self.best_score_ = -np.inf
        self.best_params_ = None
        self.best_estimator_ = None

        param_combinations = list(self._generate_param_combinations())
        
        for params in param_combinations:
            scores = []
            for train_indices, test_indices in self._split_data(X, y):
                train_X, train_y = X[train_indices], y[train_indices]
                test_X, test_y = X[test_indices], y[test_indices]
                
                estimator = copy.deepcopy(self.estimator)
                estimator.set_params(**params)
                estimator.fit(train_X, train_y)
                
                score = self._score(estimator, test_X, test_y)
                scores.append(score)

            mean_score = np.mean(scores)
            self.cv_results_['params'].append(params)
            self.cv_results_['mean_test_score'].append(mean_score)
            self.cv_results_['std_test_score'].append(np.std(scores))
            self.cv_results_['rank_test_score'].append(-1)  # Placeholder for rank, to be updated later
            
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_estimator_ = copy.deepcopy(estimator)

        # Update ranks based on mean test score
        sorted_indices = np.argsort(self.cv_results_['mean_test_score'])[::-1]
        for rank, idx in enumerate(sorted_indices, start=1):
            self.cv_results_['rank_test_score'][idx] = rank

        # Refit the best estimator on the entire dataset if refit is True
        if self.refit:
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    iris = load_iris()
    X, y = iris.data, iris.target
    param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 4, 6]}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid_search.fit(X, y)
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
