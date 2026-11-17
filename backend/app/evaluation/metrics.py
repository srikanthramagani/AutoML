"""
MetricsCalculator
=================
Responsibility: Compute comprehensive evaluation metrics for classification and regression.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricsResult:
    """
    Evaluation metrics summary.
    """
    model_name: str
    task_type: str
    metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCalculator:
    """
    Compute holdout evaluation metrics.
    """

    def __init__(self, task_type: str) -> None:
        self.task_type = task_type

    def compute(
        self,
        model_name: str,
        y_true: Any,
        y_pred: Any,
        y_proba: Optional[Any] = None,
        n_features: Optional[int] = None,
    ) -> MetricsResult:
        """
        Compute all task-appropriate metrics.
        """
        if self.task_type in ("clustering", "anomaly_detection", "unsupervised"):
            from .unsupervised_metrics import UnsupervisedMetricsCalculator
            calc = UnsupervisedMetricsCalculator(task_type=self.task_type)
            res = calc.compute(model_name=model_name, X=y_true, labels=y_pred, scores=y_proba)
            return MetricsResult(
                model_name=res.model_name,
                task_type=res.task_type,
                metrics=res.metrics,
            )

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if "classification" in self.task_type:
            return self._classification_metrics(model_name, y_true_arr, y_pred_arr, y_proba)
        return self._regression_metrics(model_name, y_true_arr, y_pred_arr, n_features)

    def _classification_metrics(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
    ) -> MetricsResult:
        metrics: Dict[str, float] = {}

        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        for class_index, class_label in enumerate(np.unique(y_true)):
            metrics[f"recall_class_{class_index}"] = float(
                recall_score(y_true, y_pred, labels=[class_label], average="macro", zero_division=0)
            )

        try:
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            metrics["mcc"] = 0.0

        if y_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    proba_col = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] > 1 else y_proba
                    metrics["roc_auc"] = float(roc_auc_score(y_true, proba_col))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
                # Log Loss (cross-entropy)
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
            except Exception:
                pass

        return MetricsResult(
            model_name=model_name,
            task_type=self.task_type,
            metrics={k: round(v, 4) for k, v in metrics.items()},
        )

    def _regression_metrics(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: Optional[int],
    ) -> MetricsResult:
        metrics: Dict[str, float] = {}

        r2 = float(r2_score(y_true, y_pred))
        metrics["r2"] = r2

        if n_features is not None and len(y_true) > n_features + 1:
            n = len(y_true)
            p = n_features
            adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - p - 1))
            metrics["adjusted_r2"] = float(adj_r2)

        metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))
        metrics["max_error"] = float(max_error(y_true, y_pred))
        metrics["median_absolute_error"] = float(median_absolute_error(y_true, y_pred))

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))

        metrics["mae"] = mae
        metrics["mse"] = mse
        metrics["rmse"] = rmse

        # MAPE (Mean Absolute Percentage Error)
        try:
            mask = y_true != 0
            if np.any(mask):
                mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
                metrics["mape"] = mape
        except Exception:
            pass

        return MetricsResult(
            model_name=model_name,
            task_type=self.task_type,
            metrics={k: round(v, 4) for k, v in metrics.items()},
        )
