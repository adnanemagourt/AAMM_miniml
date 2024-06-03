import numpy as np


def _num_samples(x):
    """
    Return the number of samples in the input array-like object `x`.

    Parameters:
    x : array-like
        Input array-like object (e.g., numpy array, list, dataframe).

    Returns:
    num_samples : int
        Number of samples in the input array-like object.
    """
    if hasattr(x, 'shape') and hasattr(x, '__array__'):
        # Handle numpy arrays and scipy sparse matrices
        return x.shape[0]
    elif hasattr(x, '__len__'):
        # Handle lists, tuples, and other iterable objects
        return len(x)
    else:
        raise ValueError("Unsupported type for determining number of samples")




