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
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
        ground_truth: Optional[Any] = None,
    ) -> UnsupervisedMetricsResult:
        """
        Compute metrics depending on task type (clustering vs anomaly detection).
        """
        X_arr = np.asarray(X)
        labels_arr = np.asarray(labels)

        if self.task_type in ("anomaly_detection", "outlier_detection"):
            return self._anomaly_metrics(model_name, X_arr, labels_arr, scores, ground_truth)
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
        ground_truth: Optional[Any] = None,
    ) -> UnsupervisedMetricsResult:
        metrics: Dict[str, float] = {}
        labels_arr = np.asarray(labels)
        # Scikit-learn anomaly detectors output 1 for inliers, -1 for outliers
        is_outlier = (labels_arr == -1)
        contamination = float(np.mean(is_outlier))
        metrics["detected_anomaly_ratio"] = contamination
        metrics["inlier_count"] = float(np.sum(~is_outlier))
        metrics["outlier_count"] = float(np.sum(is_outlier))
        metrics["evaluation_type_unsupervised_diagnostic"] = 1.0
        labels, counts = np.unique(labels_arr, return_counts=True)
        metrics["cluster_count_including_inliers"] = float(len(labels))
        for label, count in zip(labels, counts):
            metrics[f"label_{label}_count"] = float(count)

        if scores is not None and len(scores) > 0:
            scores_arr = np.asarray(scores)
            metrics["mean_anomaly_score"] = float(np.mean(scores_arr))
            metrics["std_anomaly_score"] = float(np.std(scores_arr))
            for quantile in (0.05, 0.25, 0.5, 0.75, 0.95):
                metrics[f"anomaly_score_q{int(quantile * 100):02d}"] = float(np.quantile(scores_arr, quantile))

        if ground_truth is not None:
            truth = np.asarray(ground_truth)
            if len(truth) != len(labels_arr):
                raise ValueError("Ground-truth anomaly labels and predictions must have equal length.")
            positive_labels = {"1", "true", "anomaly", "outlier", "-1"}
            truth_outlier = np.asarray([str(value).strip().lower() in positive_labels for value in truth], dtype=bool)
            metrics.pop("evaluation_type_unsupervised_diagnostic", None)
            metrics["precision"] = float(precision_score(truth_outlier, is_outlier, zero_division=0))
            metrics["recall"] = float(recall_score(truth_outlier, is_outlier, zero_division=0))
            metrics["f1"] = float(f1_score(truth_outlier, is_outlier, zero_division=0))
            metrics["confusion_matrix_tn"] = float(confusion_matrix(truth_outlier, is_outlier, labels=[False, True]).ravel()[0])
            metrics["confusion_matrix_fp"] = float(confusion_matrix(truth_outlier, is_outlier, labels=[False, True]).ravel()[1])
            metrics["confusion_matrix_fn"] = float(confusion_matrix(truth_outlier, is_outlier, labels=[False, True]).ravel()[2])
            metrics["confusion_matrix_tp"] = float(confusion_matrix(truth_outlier, is_outlier, labels=[False, True]).ravel()[3])
            if scores is not None and len(np.unique(truth_outlier)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(truth_outlier, scores))
                metrics["pr_auc"] = float(average_precision_score(truth_outlier, scores))

        return UnsupervisedMetricsResult(
            model_name=model_name,
            task_type="anomaly_detection",
            metrics={k: round(v, 4) for k, v in metrics.items()},
        )
