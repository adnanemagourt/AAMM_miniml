import numpy as np
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances
from sklearn.impute import MissingIndicator

import sys
sys.path.append('..')
from _BaseClasses import Transformer



class KNNImputer(Transformer):
    """
    K-nearest neighbors imputer.
    
    Parameters
    ----------
    
    missing_values : int, float, str, np.nan, default=np.nan
        The placeholder for the missing values.
    n_neighbors : int, default=5
        Number of neighbors to use by default.
    weights : {'uniform', 'distance', callable}, default='uniform'
        Weight function used in prediction.
        - 'uniform': uniform weights. All points in each neighborhood are weighted equally.
        - 'distance': weight points by the inverse of their distance.
        - callable: a user-defined function that accepts an array of distances and returns an array of the same shape containing the weights.
    metric : {'nan_euclidean'}, default='nan_euclidean'
        The distance metric to use.
    copy : bool, default=True
        Whether to create a copy of the input data.
    add_indicator : bool, default=False
        Whether to add a missing indicator.
    keep_empty_features : bool, default=False
        Whether to keep features with all missing values.
    """
    def __init__(self, missing_values=np.nan, n_neighbors=5, weights='uniform', 
                 metric='nan_euclidean', copy=True, add_indicator=False, 
                 keep_empty_features=False):
        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64, force_all_finite='allow-nan')
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
        
        if self.add_indicator:
            self.indicator_ = MissingIndicator(missing_values=self.missing_values)
            self.indicator_.fit(X)
        
        return self

    def transform(self, X):
        X = check_array(X, dtype=np.float64, force_all_finite='allow-nan')
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Number of features in X does not match number of features in fit data.")
        
        if self.copy:
            X = X.copy()

        mask_missing_values = np.isnan(X)
        row_missing = mask_missing_values.any(axis=1)
        col_missing = mask_missing_values.any(axis=0)

        for i in range(X.shape[0]):
            if row_missing[i]:
                valid_distances = np.full(X.shape[0], np.inf)
                for j in range(X.shape[0]):
                    if i != j and not row_missing[j]:
                        valid_mask = ~mask_missing_values[i] & ~mask_missing_values[j]
                        if np.any(valid_mask):
                            valid_distances[j] = np.linalg.norm(X[i, valid_mask] - X[j, valid_mask])
                
                neighbors_idx = np.argsort(valid_distances)[:self.n_neighbors]
                for j in range(X.shape[1]):
                    if mask_missing_values[i, j]:
                        neighbor_values = X[neighbors_idx, j]
                        valid_neighbor_mask = ~np.isnan(neighbor_values)
                        neighbor_values = neighbor_values[valid_neighbor_mask]
                        if len(neighbor_values) == 0:
                            continue
                        
                        if self.weights == 'uniform':
                            X[i, j] = np.mean(neighbor_values)
                        elif self.weights == 'distance':
                            valid_distances = valid_distances[neighbors_idx][valid_neighbor_mask]
                            weights = 1 / valid_distances
                            X[i, j] = np.average(neighbor_values, weights=weights)
                        else:
                            valid_distances = valid_distances[neighbors_idx][valid_neighbor_mask]
                            weights = self.weights(valid_distances)
                            X[i, j] = np.average(neighbor_values, weights=weights)

        if self.add_indicator:
            indicator_mask = self.indicator_.transform(X)
            X = np.hstack((X, indicator_mask))

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                return self.feature_names_in_
            return [f"x{i}" for i in range(self.n_features_in_)]
        return input_features

    def get_metadata_routing(self):
        # Example implementation (actual implementation may vary)
        return {"metadata_routing": "This is a placeholder for metadata routing"}

    def get_params(self, deep=True):
        return {
            'missing_values': self.missing_values,
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric,
            'copy': self.copy,
            'add_indicator': self.add_indicator,
            'keep_empty_features': self.keep_empty_features
        }

    def set_output(self, transform=None):
        # Placeholder for set_output API example
        return self

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self








