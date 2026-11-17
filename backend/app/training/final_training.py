"""
FinalTrainer
============
Responsibility: Retrain the winning complete pipeline on the entire training set.
"""

import time
from typing import Any, Optional
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline


class FinalTrainer:
    """
    Retrain the winning pipeline configuration on the full training partition.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.fit_time_seconds_ = 0.0

    def fit(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
    ) -> Pipeline:
        """
        Fit the winning pipeline on the full training set.
        """
        fitted_pipeline = clone(pipeline)
        start = time.time()
        if y_train is not None:
            fitted_pipeline.fit(X_train, y_train)
        else:
            fitted_pipeline.fit(X_train)
        fit_time = time.time() - start
        self.fit_time_seconds_ = round(fit_time, 4)
        print(f"[FinalTrainer] Successfully trained full pipeline in {fit_time:.2f}s on {len(X_train):,} samples.")
        return fitted_pipeline
