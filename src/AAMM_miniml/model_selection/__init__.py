import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as shuffle_arrays
from _helpers import _num_samples
from .StratifiedShuffleSplit import StratifiedShuffleSplit
# from sklearn.model_selection import StratifiedKFold
from .StratifiedKFold import StratifiedKFold
import warnings
from .KFold import KFold
from sklearn.exceptions import FitFailedWarning
from metrics import *






def train_test_split(X, y, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """
    Split arrays or matrices into random train and test subsets.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        Input samples.

    y : array-like, shape (n_samples,)
        The target variable.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
        If int, represents the absolute number of train samples.
        If None, the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance, or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as the class labels.

    Returns:
    X_train : array-like
        The training input samples.

    X_test : array-like
        The testing input samples.

    y_train : array-like
        The training target values.

    y_test : array-like
        The testing target values.
    """
    random_state = check_random_state(random_state)
    n_samples = _num_samples(X)

    if stratify is not None:
        cv = StratifiedShuffleSplit(test_size=test_size, train_size=train_size, random_state=random_state)
        train_idx, test_idx = next(cv.split(X, stratify))
    else:
        if shuffle:
            X, y = shuffle_arrays(X, y, random_state=random_state)

        if test_size is None and train_size is None:
            test_size = 0.25

        if test_size is None:
            if isinstance(train_size, float):
                test_size = 1.0 - train_size
            else:
                test_size = 0.25

        if isinstance(test_size, float):
            test_size = round(test_size * n_samples)

        if isinstance(train_size, float):
            train_size = round(train_size * n_samples)

        test_size = int(test_size)
        train_size = n_samples - test_size

        train_idx = slice(None, train_size)
        test_idx = slice(train_size, None)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test




# Mapping from string to function
SCORERS = {
    'accuracy': accuracy_score,
    'f1': f1_score,
    'mean_squared_error': mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
}

def cross_val_score(estimator_class, X, y=None, *, groups=None, scoring=None, cv=None, fit_params=None, error_score=np.nan):

    if not hasattr(estimator_class, 'fit') or not hasattr(estimator_class, 'predict'):
        raise ValueError("The estimator should have fit and predict methods.")
    
    if cv is None:
        cv = 5
    
    if isinstance(cv, int):
        if hasattr(estimator_class, 'is_classifier') and estimator_class.is_classifier and y is not None:
            cv_splitter = StratifiedKFold(n_splits=cv)
        else:
            cv_splitter = KFold(n_splits=cv)
    else:
        cv_splitter = cv
    
    if scoring is None or scoring not in SCORERS:
        scoring = 'accuracy'  # Default scoring function
    scorer = SCORERS[scoring]
    
    scores = []

    def fit_and_score(estimator_class, X_train, X_test, y_train, y_test, scorer, error_score):
        try:
            estimator = estimator_class()
            estimator.fit(X_train, y_train, **(fit_params if fit_params is not None else {}))
            y_pred = estimator.predict(X_test)
            score = scorer(y_test, y_pred)
        except Exception as e:
            if error_score == 'raise':
                raise
            warnings.warn(f"Estimator fit failed. The score will be set to {error_score}. Details: {str(e)}", FitFailedWarning)
            score = error_score
        return score

    for train_index, test_index in cv_splitter.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        score = fit_and_score(estimator_class, X_train, X_test, y_train, y_test, scorer, error_score)
        scores.append(score)

    return np.array(scores)





