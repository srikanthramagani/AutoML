"""
OptunaObjective
===============
Responsibility: Define the objective function for hyperparameter optimization.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import get_scorer
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
        study_name: Optional[str] = None,
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
        self.study_name = study_name or model_name
        self.direction = "minimize" if scoring.lower().replace("neg_", "") in {"rmse", "root_mean_squared_error", "mse", "mean_squared_error", "mae", "mean_absolute_error"} else "maximize"

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
            "root_mean_squared_error": "neg_root_mean_squared_error",
            "mse": "neg_mean_squared_error",
            "mean_squared_error": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "mean_absolute_error": "neg_mean_absolute_error",
        }.get(self.scoring.lower(), self.scoring)
        if self.scoring.lower().startswith("neg_"):
            sklearn_scoring = self.scoring.lower()

        scorer = get_scorer(sklearn_scoring)
        values = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(self.X, self.y)):
            fold_pipeline = clone(pipeline)
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]
            fold_pipeline.fit(X_train, y_train)
            score = float(scorer(fold_pipeline, X_valid, y_valid))
            if self.direction == "minimize" and sklearn_scoring.startswith("neg_"):
                score = -score
            values.append(score)
            trial.report(score, step=fold)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if not values:
            raise ValueError("No cross-validation folds were evaluated.")
        return float(np.mean(values))
