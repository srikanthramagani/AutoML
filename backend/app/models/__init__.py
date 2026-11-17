"""
Models package.
Provides the model registry and candidate model pools for
classification and regression tasks.
"""
from .model_registry import ModelRegistry
from .classification import get_classification_models
from .regression import get_regression_models

__all__ = ["ModelRegistry", "get_classification_models", "get_regression_models"]
