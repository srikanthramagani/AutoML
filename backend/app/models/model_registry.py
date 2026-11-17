"""
ModelRegistry
=============
Responsibility: Central registry mapping model names to estimator factories
per task family (classification, regression).
"""

from typing import Any, Callable, Dict, List, Optional

from sklearn.base import clone


class ModelRegistry:
    """
    Registry of available ML models per task type.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._registry: Dict[str, Dict[str, Callable[[], Any]]] = {
            "classification": {},
            "regression": {},
            "clustering": {},
            "anomaly_detection": {},
            "time_series_forecasting": {},
        }
        self._tags: Dict[str, Dict[str, List[str]]] = {
            "classification": {},
            "regression": {},
            "clustering": {},
            "anomaly_detection": {},
            "time_series_forecasting": {},
        }
        self._register_builtin_models()

    @staticmethod
    def _task_key(task_type: str) -> str:
        if task_type in {"classification", "binary_classification", "multiclass_classification"}:
            return "classification"
        if task_type == "regression":
            return "regression"
        if task_type in {"clustering", "unsupervised"}:
            return "clustering"
        if task_type in {"anomaly_detection", "outlier_detection"}:
            return "anomaly_detection"
        if task_type in {"time_series_forecasting", "time_series", "forecasting"}:
            return "time_series_forecasting"
        raise ValueError(f"Unknown task family: {task_type}")

    def _register_builtin_models(self) -> None:
        from .classification import get_classification_models
        from .regression import get_regression_models
        from .unsupervised import get_clustering_models, get_anomaly_detection_models
        from .time_series import get_time_series_models

        for task_type, factory in (
            ("classification", get_classification_models),
            ("regression", get_regression_models),
            ("clustering", get_clustering_models),
            ("anomaly_detection", get_anomaly_detection_models),
            ("time_series_forecasting", get_time_series_models),
        ):
            for entry in factory(random_state=self.random_state):
                prototype = entry["estimator"]
                self.register(
                    task_type,
                    entry["name"],
                    lambda prototype=prototype: clone(prototype),
                    tags=entry.get("tags", []),
                )

    def register(
        self,
        task_type: str,
        name: str,
        estimator_factory: Callable[[], Any],
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a model factory.
        """
        task_key = self._task_key(task_type)
        self._registry[task_key][name] = estimator_factory
        self._tags[task_key][name] = list(tags or [])

    def get(self, task_type: str, name: str) -> Any:
        """
        Return a fresh unfitted estimator.
        """
        task_key = self._task_key(task_type)
        if name not in self._registry.get(task_key, {}):
            raise KeyError(f"Model '{name}' not found for task '{task_key}'")
        return self._registry[task_key][name]()

    def list_models(self, task_type: str) -> List[str]:
        """Return a list of registered model names for a task type."""
        task_key = self._task_key(task_type)
        return list(self._registry.get(task_key, {}).keys())

    def list_entries(self, task_type: str) -> List[Dict[str, Any]]:
        task_key = self._task_key(task_type)
        return [
            {
                "name": name,
                "estimator": self.get(task_key, name),
                "tags": list(self._tags[task_key].get(name, [])),
            }
            for name in self.list_models(task_key)
        ]
