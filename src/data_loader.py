"""Data loading utilities."""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_data():
    """Load and return the Iris dataset as pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.target_names


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
