"""
Unsupervised Metrics Calculator
===============================
Responsibility: Compute clustering and anomaly detection evaluation metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass
class UnsupervisedMetricsResult:
    """
    Evaluation metrics summary for unsupervised models.
    """
    model_name: str
    task_type: str
    metrics: Dict[str, float] = field(default_factory=dict)


class UnsupervisedMetricsCalculator:
    """
    Compute intrinsic evaluation metrics for clustering and anomaly detection.
    """

    def __init__(self, task_type: str = "clustering") -> None:
        self.task_type = task_type

    def compute(
        self,
        model_name: str,
        X: Any,
        labels: Any,
        scores: Optional[Any] = None,
    ) -> UnsupervisedMetricsResult:
        """
        Compute metrics depending on task type (clustering vs anomaly detection).
        """
        X_arr = np.asarray(X)
        labels_arr = np.asarray(labels)

        if self.task_type in ("anomaly_detection", "outlier_detection"):
            return self._anomaly_metrics(model_name, X_arr, labels_arr, scores)
        return self._clustering_metrics(model_name, X_arr, labels_arr)

    def _clustering_metrics(
        self,
        model_name: str,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> UnsupervisedMetricsResult:
        metrics: Dict[str, float] = {}

        # Filter out noise points (-1 in DBSCAN) for silhouette if needed, or compute overall
        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        metrics["n_clusters"] = float(n_clusters)

        # Silhouette & Davies-Bouldin require at least 2 clusters and less than N samples
        if 1 < n_clusters < len(X):
            # If DBSCAN generated noise (-1), calculate valid silhouette on non-noise or whole
            valid_mask = labels != -1 if -1 in unique_labels and n_clusters >= 2 else np.ones(len(labels), dtype=bool)
            if np.sum(valid_mask) > n_clusters:
                # Subsample for very fast silhouette calculation if dataset is large
                subsample_size = min(2000, np.sum(valid_mask))
                metrics["silhouette_score"] = float(
                    silhouette_score(
                        X[valid_mask],
                        labels[valid_mask],
                        sample_size=subsample_size if len(X[valid_mask]) > 2000 else None,
                        random_state=42,
                    )
                )
                metrics["calinski_harabasz_score"] = float(
                    calinski_harabasz_score(X[valid_mask], labels[valid_mask])
                )
                metrics["davies_bouldin_score"] = float(
                    davies_bouldin_score(X[valid_mask], labels[valid_mask])
                )
            else:
                metrics["silhouette_score"] = -1.0
                metrics["calinski_harabasz_score"] = 0.0
                metrics["davies_bouldin_score"] = 999.0
        else:
            metrics["silhouette_score"] = -1.0
            metrics["calinski_harabasz_score"] = 0.0
            metrics["davies_bouldin_score"] = 999.0

        if -1 in unique_labels:
            metrics["noise_ratio"] = float(np.mean(labels == -1))

        return UnsupervisedMetricsResult(
            model_name=model_name,
            task_type="clustering",
            metrics={k: round(v, 4) for k, v in metrics.items()},
        )

    def _anomaly_metrics(
        self,
        model_name: str,
        X: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray],
    ) -> UnsupervisedMetricsResult:
        metrics: Dict[str, float] = {}
        # Scikit-learn anomaly detectors output 1 for inliers, -1 for outliers
        is_outlier = (labels == -1)
        contamination = float(np.mean(is_outlier))
        metrics["detected_anomaly_ratio"] = contamination
        metrics["inlier_count"] = float(np.sum(~is_outlier))
        metrics["outlier_count"] = float(np.sum(is_outlier))

        if scores is not None and len(scores) > 0:
            scores_arr = np.asarray(scores)
            metrics["mean_anomaly_score"] = float(np.mean(scores_arr))
            metrics["std_anomaly_score"] = float(np.std(scores_arr))

        return UnsupervisedMetricsResult(
            model_name=model_name,
            task_type="anomaly_detection",
            metrics={k: round(v, 4) for k, v in metrics.items()},
        )
