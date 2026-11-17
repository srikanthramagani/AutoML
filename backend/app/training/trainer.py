"""
Trainer
=======
Responsibility: Fit a full pipeline (preprocessing + estimator) on data partitions.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline


@dataclass
class TrainingResult:
    """
    Result of a single training execution.
    """
    model_name: str
    pipeline: Pipeline
    fit_time_seconds: float
    hyperparams: Dict[str, Any]


class Trainer:
    """
    Fit preprocessing + estimator pipelines.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit(
        self,
        preprocessing_pipeline: Pipeline,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "model",
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """
        Assemble the full pipeline and fit on training partitions.
        """
        hyperparams = hyperparams or {}
        est = clone(estimator)
        if hyperparams:
            est.set_params(**hyperparams)

        full_pipeline = Pipeline(
            steps=list(preprocessing_pipeline.steps) + [("estimator", est)]
        )

        start = time.time()
        full_pipeline.fit(X, y)
        fit_time = time.time() - start

        return TrainingResult(
            model_name=model_name,
            pipeline=full_pipeline,
            fit_time_seconds=round(fit_time, 4),
            hyperparams=hyperparams,
        )
