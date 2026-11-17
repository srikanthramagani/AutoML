"""
TaskDetector
============
Responsibility: Automatically determine the ML task type from the target column
and dataset profile.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from .profiler import DataProfile


class TaskDetectionError(Exception):
    """Raised when the task type cannot be determined reliably."""


@dataclass
class TaskInfo:
    """
    Result of task detection.
    """
    task_type: str  # 'binary_classification', 'multiclass_classification', 'regression'
    n_classes: Optional[int]
    target_column: str
    primary_metric: str
    is_imbalanced: bool = False


class TaskDetector:
    """
    Detect the ML task type from the target column.
    """

    _DEFAULT_METRICS = {
        "binary_classification": "roc_auc",
        "multiclass_classification": "f1_weighted",
        "regression": "r2",
    }

    def __init__(
        self,
        classification_threshold: int = 20,
        imbalance_ratio_threshold: float = 0.20,
    ) -> None:
        self.classification_threshold = classification_threshold
        self.imbalance_ratio_threshold = imbalance_ratio_threshold

    def detect(
        self,
        y: pd.Series,
        profile: Optional[DataProfile] = None,
        override_metric: Optional[str] = None,
    ) -> TaskInfo:
        """
        Detect the ML task type.
        """
        target_col = str(y.name) if y.name is not None else "target"
        clean_y = y.dropna()
        if len(clean_y) == 0:
            raise TaskDetectionError(f"Target column '{target_col}' contains only missing values.")

        n_unique = int(clean_y.nunique())

        if n_unique < 2:
            raise TaskDetectionError(
                f"Target column '{target_col}' must contain at least two distinct values."
            )

        task_type = self._infer_task(clean_y, n_unique)
        n_classes = None
        is_imbalanced = False

        if task_type in ("binary_classification", "multiclass_classification"):
            n_classes = n_unique
            is_imbalanced = self._check_imbalance(clean_y)

        if override_metric and override_metric != "auto":
            primary_metric = override_metric
        else:
            primary_metric = self._DEFAULT_METRICS[task_type]

        return TaskInfo(
            task_type=task_type,
            n_classes=n_classes,
            target_column=target_col,
            primary_metric=primary_metric,
            is_imbalanced=is_imbalanced,
        )

    def _infer_task(self, y: pd.Series, n_unique: int) -> str:
        # If non-numeric (string, object, category, bool)
        if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
            return "binary_classification" if n_unique == 2 else "multiclass_classification"

        # If floating point numbers with many unique values, it's regression
        if pd.api.types.is_float_dtype(y):
            # Check if all values are integers (e.g. 0.0, 1.0)
            is_all_int = np.all(np.mod(y.values, 1) == 0)
            if not is_all_int:
                return "regression"

        # Integer or whole number floats
        if n_unique == 2:
            return "binary_classification"
        if n_unique <= self.classification_threshold:
            return "multiclass_classification"

        return "regression"

    def _check_imbalance(self, y: pd.Series) -> bool:
        counts = y.value_counts(normalize=True)
        if len(counts) <= 1:
            return False
        return bool(counts.min() < self.imbalance_ratio_threshold)

    def describe(self, info: TaskInfo) -> str:
        lines = [
            f"  Detected Task       : {info.task_type.replace('_', ' ').title()}",
            f"  Target Column       : {info.target_column}",
            f"  Primary Metric      : {info.primary_metric}",
        ]
        if info.n_classes is not None:
            lines.append(f"  Target Classes      : {info.n_classes}")
        if info.is_imbalanced:
            lines.append("  Class Imbalance     : Detected (minority class < 20%)")
        return "\n".join(lines)
