"""
Tests for Model Pools and Model Registry
"""

from backend.app.models.classification import get_classification_models
from backend.app.models.regression import get_regression_models
from backend.app.models.model_registry import ModelRegistry


def test_classification_models():
    models = get_classification_models()
    assert len(models) == 9
    for m in models:
        assert "name" in m
        assert hasattr(m["estimator"], "fit")


def test_regression_models():
    models = get_regression_models()
    assert len(models) == 10
    for m in models:
        assert "name" in m
        assert hasattr(m["estimator"], "fit")


def test_model_registry():
    registry = ModelRegistry()
    from sklearn.linear_model import LogisticRegression
    registry.register("classification", "my_lr", lambda: LogisticRegression())
    est = registry.get("classification", "my_lr")
    assert hasattr(est, "fit")
    assert "my_lr" in registry.list_models("classification")


def test_model_registry_contains_builtin_pools():
    registry = ModelRegistry()
    assert len(registry.list_models("classification")) == 9
    assert len(registry.list_models("regression")) == 10
    assert registry.get("binary_classification", "logistic_regression").__class__.__name__ == "LogisticRegression"
    assert registry.get("classification", "xgboost_classifier").__class__.__name__ == "XGBClassifier"
    assert registry.get("regression", "xgboost_regressor").__class__.__name__ == "XGBRegressor"
