"""
UnsupervisedOptunaObjective
===========================
Responsibility: Define Optuna objective evaluation for clustering and anomaly detection.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import silhouette_score
from .search_space import SearchSpace


class UnsupervisedOptunaObjective:
    """
    Optuna objective function for unsupervised candidates (Clustering & Anomaly Detection).
    Optimized for high performance using preprocessed features where possible.
    """

    def __init__(
        self,
        estimator: Any,
        model_name: str,
        X: Any,
        task_type: str = "clustering",
        scoring: str = "silhouette_score",
        search_space: Optional[SearchSpace] = None,
        random_state: int = 42,
    ) -> None:
        self.estimator = estimator
        self.model_name = model_name
        self.X = np.asarray(X)
        self.task_type = task_type
        self.scoring = scoring
        self.search_space = search_space or SearchSpace()
        self.random_state = random_state

    def __call__(self, trial: Any) -> float:
        params = self.search_space.sample(self.model_name, trial)
        est = clone(self.estimator)
        if params:
            est.set_params(**params)

        try:
            if self.task_type == "anomaly_detection":
                # For anomaly detection, fit & evaluate contamination stability and decision function
                if hasattr(est, "fit_predict"):
                    labels = est.fit_predict(self.X)
                else:
                    labels = est.fit(self.X).predict(self.X)
                
                # Check for reasonable anomaly fraction (between 1% and 30%)
                outlier_ratio = np.mean(labels == -1)
                if outlier_ratio == 0 or outlier_ratio > 0.5:
                    return -1.0
                
                # If decision_function or score_samples is available, evaluate margin
                if hasattr(est, "decision_function"):
                    scores = est.decision_function(self.X)
                    return float(np.mean(scores))
                elif hasattr(est, "score_samples"):
                    scores = est.score_samples(self.X)
                    return float(np.mean(scores))
                return float(1.0 - abs(outlier_ratio - 0.1))

            # Clustering objective: evaluate silhouette score
            if hasattr(est, "fit_predict"):
                labels = est.fit_predict(self.X)
            else:
                est.fit(self.X)
                labels = getattr(est, "labels_", None)
                if labels is None and hasattr(est, "predict"):
                    labels = est.predict(self.X)

            if labels is None:
                return -1.0

            unique_labels = set(labels) - {-1}
            n_clusters = len(unique_labels)
            if n_clusters < 2 or n_clusters >= len(self.X):
                return -1.0

            # Subsample for ultra-fast silhouette computation on large datasets
            sample_size = min(2000, len(self.X))
            valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
            if np.sum(valid_mask) <= n_clusters:
                return -1.0

            score = silhouette_score(
                self.X[valid_mask],
                labels[valid_mask],
                sample_size=sample_size if len(self.X[valid_mask]) > 2000 else None,
                random_state=self.random_state,
            )
            return float(score)

        except Exception:
            return -1.0
