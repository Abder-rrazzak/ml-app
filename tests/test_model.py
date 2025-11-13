"""Tests for the ML model."""

import numpy as np
import pytest

from src.data_loader import load_iris_data, split_data
from src.model import IrisClassifier


def test_iris_classifier_initialization():
    """Test IrisClassifier initialization."""
    classifier = IrisClassifier()
    assert not classifier.is_trained


def test_iris_classifier_training():
    """Test model training."""
    df, _ = load_iris_data()
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = split_data(X, y)

    classifier = IrisClassifier()
    classifier.train(X_train, y_train)

    assert classifier.is_trained


def test_iris_classifier_prediction():
    """Test model prediction."""
    df, _ = load_iris_data()
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = split_data(X, y)

    classifier = IrisClassifier()
    classifier.train(X_train, y_train)

    predictions = classifier.predict(X_test)
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1, 2] for pred in predictions)


def test_prediction_without_training():
    """Test that prediction fails without training."""
    classifier = IrisClassifier()
    X_dummy = np.array([[1, 2, 3, 4]])

    with pytest.raises(ValueError):
        classifier.predict(X_dummy)
