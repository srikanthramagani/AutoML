"""
CrossValidator
==============
Responsibility: Execute cross-validation on the full pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


@dataclass
class CVResult:
    """
    Cross-validation result for a single model.
    """
    model_name: str
    scores: List[float]
    mean_score: float
    std_score: float
    scoring: str
    n_folds: int
    is_valid: bool = True
    error: Optional[str] = None


class CrossValidator:
    """
    K-fold cross-validator for complete sklearn Pipelines.
    """

    def __init__(
        self,
        n_folds: int = 5,
        scoring: str = "roc_auc",
        task_type: str = "classification",
        random_state: int = 42,
    ) -> None:
        self.n_folds = n_folds
        self.scoring = scoring
        self.task_type = task_type
        self.random_state = random_state

    def validate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "model",
    ) -> CVResult:
        """
        Run k-fold cross-validation on the complete pipeline.
        """
        if "classification" in self.task_type:
            cv = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )
        else:
            cv = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )

        try:
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=self.scoring,
                error_score=np.nan,
            )
            valid_scores = scores[~np.isnan(scores)]
            if len(valid_scores) == 0:
                raise ValueError("All cross-validation folds failed.")
            error = None
            is_valid = True
        except Exception as exc:
            valid_scores = np.array([], dtype=float)
            error = str(exc)
            is_valid = False

        return CVResult(
            model_name=model_name,
            scores=[float(s) for s in valid_scores],
            mean_score=float(np.mean(valid_scores)) if len(valid_scores) else float("nan"),
            std_score=float(np.std(valid_scores)) if len(valid_scores) else float("nan"),
            scoring=self.scoring,
            n_folds=self.n_folds,
            is_valid=is_valid,
            error=error,
        )
