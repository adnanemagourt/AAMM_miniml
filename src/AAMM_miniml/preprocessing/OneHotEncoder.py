import numpy as np

import sys
sys.path.append('..')
from _BaseClasses import Transformer


class OneHotEncoder(Transformer):
    """
    One-hot encoder for categorical features.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    categories_ : list
        A list of unique categories for each feature.
    """
    def __init__(self):
        self.categories_ = []

    def fit(self, X):
        # Find unique categories in each column
        for i in range(X.shape[1]):
            unique_categories = np.unique(X[:, i])
            self.categories_.append(unique_categories)

    def transform(self, X):
        transformed_features = []
        for i in range(X.shape[1]):
            # Create binary features for each unique category
            categories = self.categories_[i]
            encoded_features = np.zeros((len(X), len(categories)), dtype=int)
            for j, category in enumerate(categories):
                encoded_features[:, j] = (X[:, i] == category).astype(int)
            transformed_features.append(encoded_features)
        return np.concatenate(transformed_features, axis=1)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    import pandas as pd

    # Create a sample DataFrame with categorical features
    data = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'red'],
        'shape': ['square', 'circle', 'triangle', 'square']
    })

    print("Original DataFrame:")
    print(data)

    # Extract categorical features
    categorical_features = data[['color', 'shape']].values

    # Apply OneHotEncoder
    encoder = OneHotEncoder()
    encoder.fit(categorical_features)
    encoded_features = encoder.transform(categorical_features)

    # Create a new DataFrame with encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.categories_[0].tolist() + encoder.categories_[1].tolist())

    print("\nEncoded DataFrame:")
    print(encoded_df)

