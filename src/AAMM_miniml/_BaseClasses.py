from abc import ABC, abstractmethod
from re import A
import numpy as np

class Estimator(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,), default=None
            The target values.
            
        Returns
        -------
        self : object
            Returns self."""
        pass
    
    def get_params(self) -> dict:
        """
        Get parameters for this estimator.
        Returns
        -------
        params : dict
            Parameter names mapped to their values."""
        return self.__dict__

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects (such as pipelines). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's possible to update each component of a nested
        object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

class Transformer(Estimator):
    @abstractmethod
    def transform(self, X) -> np.ndarray:
        """Transform the input data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features')
            The output samples."""
        pass

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit to data, then transform it.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,), default=None
            The target values.
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features')
            The transformed samples."""
        self.fit(X, y)
        return self.transform(X)

class Predictor(Estimator):
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predict the target for the provided data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted target values."""
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        Returns
        -------
        self : object
            Returns self.
            """
        pass

    def fit_predict(self, X, y) -> np.ndarray:
        """Fit to data, then predict it.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        y : array-like of shape (n_samples,)
            The target values.
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted target values."""
        self.fit(X, y)
        return self.predict(X)

    @abstractmethod
    def score(self, X, y) -> float:
        """"
        Return the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y."""
        pass

class Regressor(Predictor, ABC):
    pass

class Classifier(Predictor, ABC):
    pass

class MetaEstimator(Estimator, ABC):
    pass

class Splitter(ABC):
    
    @abstractmethod
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,), default=None
            The target data.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
            """
        pass

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            The input data.
        y : array-like of shape (n_samples,), default=None
            The target data.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into train/test set.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
            """
        pass


class ModelSelector(MetaEstimator, Predictor, ABC):
    best_score_: float
    best_params_: dict
    best_estimator_: Estimator

