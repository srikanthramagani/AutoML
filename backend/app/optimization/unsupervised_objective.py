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
from sklearn.metrics import adjusted_rand_score
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
        self.direction = "maximize"
        self.study_name = model_name

    def __call__(self, trial: Any) -> float:
        params = self.search_space.sample(self.model_name, trial)
        est = clone(self.estimator)
        if params:
            est.set_params(**params)

        try:
            if self.task_type == "anomaly_detection":
                # Without labels, optimize repeat-fit partition stability only. This is
                # explicitly a diagnostic, not an accuracy estimate.
                from sklearn.metrics import adjusted_rand_score
                labels_by_seed = []
                rng = np.random.default_rng(self.random_state)
                sample_n = max(2, int(0.8 * len(self.X)))
                for _ in range(3):
                    candidate = clone(est)
                    if "random_state" in candidate.get_params(deep=False):
                        candidate.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
                    sample_idx = rng.choice(len(self.X), size=min(sample_n, len(self.X)), replace=False)
                    candidate.fit(self.X[sample_idx])
                    labels = candidate.predict(self.X)
                    labels_by_seed.append(np.asarray(labels))
                stability = [adjusted_rand_score(labels_by_seed[0], other) for other in labels_by_seed[1:]]
                return float(np.mean(stability))

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
