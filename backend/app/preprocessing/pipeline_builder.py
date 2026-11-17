"""
PreprocessingPipelineBuilder
============================
Responsibility: Assemble a complete scikit-learn preprocessing Pipeline
tailored to the dataset profile and task type.
"""

from typing import List, Optional
from sklearn.pipeline import Pipeline

from ..data.profiler import DataProfile
from ..data.task_detector import TaskInfo
from .encoding import CategoricalEncoder
from .missing_values import MissingValueHandler
from .outlier_handler import OutlierHandler
from .scaling import FeatureScaler
from .time_features import TimeSeriesFeatureExtractor


class PreprocessingPipelineBuilder:
    """
    Build a scikit-learn preprocessing Pipeline from a DataProfile.
    """

    def __init__(
        self,
        include_outlier_handling: bool = True,
        scaler_strategy: str = "standard",
        encoder_strategy: str = "auto",
        max_onehot_cardinality: int = 15,
    ) -> None:
        self.include_outlier_handling = include_outlier_handling
        self.scaler_strategy = scaler_strategy
        self.encoder_strategy = encoder_strategy
        self.max_onehot_cardinality = max_onehot_cardinality

    def build(
        self,
        profile: Optional[DataProfile] = None,
        task_info: Optional[TaskInfo] = None,
    ) -> Pipeline:
        """
        Construct an unfitted scikit-learn Pipeline with all sequential steps:
        1. TimeSeriesFeatureExtractor (if time series forecasting)
        2. OutlierHandler (Tukey IQR clip on numeric features)
        3. MissingValueHandler (median numeric, mode categorical)
        4. CategoricalEncoder (OneHot / Ordinal)
        5. FeatureScaler (StandardScaler / RobustScaler)
        """
        steps = []

        is_time_series = task_info and task_info.task_type in ("time_series_forecasting", "forecasting", "time_series")
        if is_time_series:
            steps.append(("time_feature_extractor", TimeSeriesFeatureExtractor()))

        if self.include_outlier_handling:
            steps.append(("outlier_handler", OutlierHandler(strategy="clip")))

        steps.append(
            ("missing_value_handler", MissingValueHandler(
                numeric_strategy="median",
                categorical_strategy="most_frequent",
                add_indicator=False,
            ))
        )

        steps.append(
            ("categorical_encoder", CategoricalEncoder(
                strategy=self.encoder_strategy,
                max_onehot_cardinality=self.max_onehot_cardinality,
            ))
        )

        steps.append(
            ("feature_scaler", FeatureScaler(strategy=self.scaler_strategy))
        )

        return Pipeline(steps=steps)
