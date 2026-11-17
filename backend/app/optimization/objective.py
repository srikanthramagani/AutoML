"""
OptunaObjective
===============
Responsibility: Define the objective function for hyperparameter optimization.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from .search_space import SearchSpace, HAS_OPTUNA


class OptunaObjective:
    """
    Optuna objective function for candidate models.
    """

    def __init__(
        self,
        preprocessing_pipeline: Pipeline,
        estimator: Any,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str,
        cv_folds: int = 5,
        task_type: str = "classification",
        search_space: Optional[SearchSpace] = None,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.preprocessing_pipeline = preprocessing_pipeline
        self.estimator = estimator
        self.model_name = model_name
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.task_type = task_type
        self.search_space = search_space or SearchSpace()
        self.random_state = random_state
        self.n_jobs = n_jobs

    def __call__(self, trial: Any) -> float:
        params = self.search_space.sample(self.model_name, trial)
        est = clone(self.estimator)
        if params:
            est.set_params(**params)

        pipeline = Pipeline(
            steps=list(self.preprocessing_pipeline.steps) + [("estimator", est)]
        )

        if "classification" in self.task_type:
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
        elif self.task_type in ("time_series_forecasting", "time_series", "forecasting"):
            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            cv = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )

        sklearn_scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        }.get(self.scoring.lower(), self.scoring)

        try:
            scores = cross_val_score(
                pipeline,
                self.X,
                self.y,
                cv=cv,
                scoring=sklearn_scoring,
                error_score=np.nan,
                n_jobs=self.n_jobs,
            )
            valid_scores = scores[~np.isnan(scores)]
            if len(valid_scores) == 0:
                return -1e9
            return float(np.mean(valid_scores))
        except Exception:
            return -1e9
