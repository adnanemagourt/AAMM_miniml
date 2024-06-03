import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Transformer

class LabelEncoder(Transformer):
    """
    Encode target labels with value between 0 and n_classes-1.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    classes_ : ndarray
        The unique classes found in the target array.
    """
    def __init__(self):
        self.classes_ = []

    def fit(self, y):
        # Find unique classes in the target array
        self.classes_ = np.unique(y)

    def transform(self, y):
        transformed_labels = np.searchsorted(self.classes_, y)
        return transformed_labels

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

if __name__ == "__main__":
    import pandas as pd

    # Create a sample DataFrame with a categorical target column
    data = pd.DataFrame({
        'fruit': ['apple', 'banana', 'apple', 'orange'],
        'Game' : ['Cricket', 'Football', 'Cricket', 'Tennis']
    })

    print("Original DataFrame:")
    print(data)

    # Extract the target column
    target = data['fruit'].values

    # Apply LabelEncoder
    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(target)

    # Add the encoded target back to the DataFrame
    data['encoded_fruit'] = encoded_target

    print("\nDataFrame with encoded target:")
    print(data)
