import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

import sys
sys.path.append('..')
from _BaseClasses import Regressor, MetaEstimator

class Stacking(MetaEstimator, Regressor):
    """
    Stacked generalization.
    
    Parameters
    ----------
    
    base_models : list
        A list of base models.
    meta_model : object
        The meta-model that aggregates the predictions of the base models.
    n_folds : int, default=5    
        Number of folds.
    """
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    
    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_idx], y[train_idx])
                self.base_models_[i].append(instance)
                meta_features[holdout_idx, i] = instance.predict(X[holdout_idx])
        
        self.meta_model_.fit(meta_features, y)
        return self
    
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)
    
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)