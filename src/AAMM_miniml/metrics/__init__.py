import numpy as np
import warnings

def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    """
    Compute the accuracy score.

    Parameters:
    y_true : 1d array-like
        Ground truth (correct) labels.
    y_pred : 1d array-like
        Predicted labels, as returned by a classifier.
    normalize : bool, default=True
        If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns:
    score : float
        If normalize == True, return the fraction of correctly classified samples (float),
        else returns the number of correctly classified samples (int).
    """

    # Convert inputs to numpy arrays for efficient computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if sample weights are provided and are valid
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weight length must be equal to the length of y_true")

    # Compute number of correctly classified samples
    correct = np.sum(y_true == y_pred)

    # Apply sample weights if provided
    if sample_weight is not None:
        weighted_correct = np.sum(sample_weight * (y_true == y_pred))
        score = weighted_correct if not normalize else weighted_correct / np.sum(sample_weight)
    else:
        score = correct if not normalize else correct / len(y_true)

    return score



def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', zero_division='warn'):
    """
    Compute the F1 score.

    Parameters:
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    labels : array-like, default=None
        The set of labels to include when average != 'binary'.
    pos_label : int, float, bool or str, default=1
        The class to report if average='binary'.
    average : {'micro', 'macro', 'samples', 'weighted', 'binary'}, default='binary'
        Type of averaging performed on the data.
    zero_division : {'warn', 0.0, 1.0, np.nan}, default='warn'
        Sets the value to return when there is a zero division.

    Returns:
    f1_score : float or array of float
        F1 score of the positive class in binary classification or weighted average of
        the F1 scores of each class for the multiclass task.
    """

    # Convert inputs to numpy arrays for efficient computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Determine unique labels and their counts
    unique_labels = np.unique(y_true)
    n_labels = len(unique_labels)

    if labels is None:
        labels = unique_labels

    if average == 'binary':
        if labels is None or pos_label not in labels:
            labels = [pos_label]

    f1_scores = []

    # Calculate F1 score for each label
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        # Calculate F1 score
        if tp == 0 and (zero_division == 'warn' or zero_division == 0.0):
            f1 = 0.0
        else:
            try:
                precision = tp / (tp + fp)
            except ZeroDivisionError:
                if zero_division == 'warn':
                    warnings.warn("precision is ill-defined and being set to 0.0 due to no predicted samples.", UserWarning)
                    precision = 0.0
                else:
                    precision = zero_division
            try:
                recall = tp / (tp + fn)
            except ZeroDivisionError:
                if zero_division == 'warn':
                    warnings.warn("recall is ill-defined and being set to 0.0 due to no predicted samples.", UserWarning)
                    recall = 0.0
                else:
                    recall = zero_division
            try:
                f1 = (2 * precision * recall) / (precision + recall)
            except ZeroDivisionError:
                if zero_division == 'warn':
                    warnings.warn("F1 score is ill-defined and being set to 0.0 due to no predicted samples.", UserWarning)
                    f1 = 0.0
                else:
                    f1 = zero_division

        f1_scores.append(f1)

    # Calculate average F1 score based on the 'average' parameter
    if average == 'micro':
        tp_sum = np.sum([np.sum((y_true == label) & (y_pred == label)) for label in labels])
        fp_sum = np.sum([np.sum((y_true != label) & (y_pred == label)) for label in labels])
        fn_sum = np.sum([np.sum((y_true == label) & (y_pred != label)) for label in labels])

        try:
            precision = tp_sum / (tp_sum + fp_sum)
        except ZeroDivisionError:
            if zero_division == 'warn':
                warnings.warn("precision is ill-defined and being set to 0.0 due to no predicted samples.", UserWarning)
                precision = 0.0
            else:
                precision = zero_division
        
        try:
            recall = tp_sum / (tp_sum + fn_sum)
        except ZeroDivisionError:
            if zero_division == 'warn':
                warnings.warn("recall is ill-defined and being set to 0.0 due to no predicted samples.", UserWarning)
                recall = 0.0
            else:
                recall = zero_division
        
        try:
            f1_macro = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            if zero_division == 'warn':
                warnings.warn("F1 score is ill-defined and being set to 0.0 due to no predicted samples.", UserWarning)
                f1_macro = 0.0
            else:
                f1_macro = zero_division

    elif average == 'macro':
        f1_macro = np.mean(f1_scores)

    elif average == 'weighted':
        supports = [np.sum(y_true == label) for label in labels]
        f1_macro = np.average(f1_scores, weights=supports)

    elif average == 'samples':
        f1_macro = np.mean(f1_scores)

    else:  # average == None or average == 'binary'
        f1_macro = f1_scores

    return f1_macro



def mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True):
    """
    Compute the mean squared error regression loss.

    Parameters:
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
        - 'raw_values' : Returns a full set of errors in case of multioutput input.
        - 'uniform_average' : Errors of all outputs are averaged with uniform weight.

    squared : bool, default=True
        If True returns MSE value, if False returns RMSE value (deprecated).

    Returns:
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    # Check if shapes are compatible
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if sample_weight is not None and sample_weight.shape != y_true.shape:
        raise ValueError("sample_weight must have the same shape as y_true.")

    # Compute squared errors
    if squared:
        errors = (y_true - y_pred) ** 2
    else:
        errors = np.sqrt((y_true - y_pred) ** 2)  # Compute RMSE (deprecated)

    # Apply sample weights if provided
    if sample_weight is not None:
        errors = errors * sample_weight

    # Calculate aggregated error based on multioutput type
    if multioutput == 'raw_values':
        # Return individual errors for each output
        return errors if y_true.ndim == 1 else errors.sum(axis=1)
    elif multioutput == 'uniform_average':
        # Compute mean of errors, possibly weighted
        if sample_weight is not None:
            return np.average(errors, weights=sample_weight)
        else:
            return np.mean(errors)
    else:
        raise ValueError("Invalid 'multioutput' parameter. Expected 'raw_values' or 'uniform_average'.")



def mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    Compute the mean absolute error regression loss.

    Parameters:
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
        - 'raw_values' : Returns a full set of errors in case of multioutput input.
        - 'uniform_average' : Errors of all outputs are averaged with uniform weight.

    Returns:
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the weighted average of all output errors is returned.
        MAE output is non-negative floating point. The best value is 0.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    # Check if shapes are compatible
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if sample_weight is not None and sample_weight.shape != y_true.shape:
        raise ValueError("sample_weight must have the same shape as y_true.")

    # Compute absolute errors
    errors = np.abs(y_true - y_pred)

    # Apply sample weights if provided
    if sample_weight is not None:
        errors = errors * sample_weight

    # Calculate aggregated error based on multioutput type
    if multioutput == 'raw_values':
        # Return individual errors for each output
        return errors if y_true.ndim == 1 else errors.sum(axis=1)
    elif multioutput == 'uniform_average':
        # Compute mean of errors, possibly weighted
        if sample_weight is not None:
            return np.average(errors, weights=sample_weight)
        else:
            return np.mean(errors)
    else:
        raise ValueError("Invalid 'multioutput' parameter. Expected 'raw_values' or 'uniform_average'.")



def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    Compute the root mean squared error regression loss.

    Parameters:
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values. Array-like value defines weights used to average errors.
        - 'raw_values' : Returns a full set of errors in case of multioutput input.
        - 'uniform_average' : Errors of all outputs are averaged with uniform weight.

    Returns:
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    # Compute squared errors
    squared_errors = (y_true - y_pred) ** 2

    # Apply sample weights if provided
    if sample_weight is not None:
        squared_errors = squared_errors * sample_weight

    # Compute mean of squared errors
    if multioutput == 'raw_values':
        # Return individual RMSE for each output
        rmse = np.sqrt(squared_errors) if y_true.ndim == 1 else np.sqrt(squared_errors.sum(axis=1))
    elif multioutput == 'uniform_average':
        # Compute mean RMSE, possibly weighted
        if sample_weight is not None:
            rmse = np.sqrt(np.average(squared_errors, weights=sample_weight))
        else:
            rmse = np.sqrt(np.mean(squared_errors))
    else:
        raise ValueError("Invalid 'multioutput' parameter. Expected 'raw_values' or 'uniform_average'.")

    return rmse












