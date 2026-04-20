import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics


@pytest.fixture
def sample_data():
    """Create small sample training data for testing."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


def test_train_model_returns_random_forest(sample_data):
    """
    Test that train_model returns a fitted RandomForestClassifier.
    """
    X, y = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_correct_shape(sample_data):
    """
    Test that inference returns a numpy array with the same
    number of predictions as input samples.
    """
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics_returns_valid_scores(sample_data):
    """
    Test that compute_model_metrics returns precision, recall,
    and F1 scores that are all between 0 and 1.
    """
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0