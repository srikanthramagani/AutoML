"""
Preprocessing package.
Provides missing-value imputation, encoding, scaling, outlier handling,
and a pipeline builder that assembles these steps into a single
sklearn-compatible Pipeline object.
"""
from .missing_values import MissingValueHandler
from .encoding import CategoricalEncoder
from .scaling import FeatureScaler
from .outlier_handler import OutlierHandler
from .pipeline_builder import PreprocessingPipelineBuilder

__all__ = [
    "MissingValueHandler",
    "CategoricalEncoder",
    "FeatureScaler",
    "OutlierHandler",
    "PreprocessingPipelineBuilder",
]