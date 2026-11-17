"""
HyperparameterOptimizer
=======================
Responsibility: Orchestrate Optuna hyperparameter optimization studies for model candidates.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False


@dataclass
class ModelOptimizationResult:
    """
    Result for a single model's HPO study.
    """
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    is_valid: bool = True
    error: Optional[str] = None


class HyperparameterOptimizer:
    """
    Run Optuna HPO for model candidates with graceful fallback when Optuna is not installed.
    """

    def __init__(
        self,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state

    def optimize_model(
        self,
        objective: Any,
        model_name: str,
    ) -> ModelOptimizationResult:
        """
        Run Optuna study for one model candidate, or fallback to default params if Optuna is absent.
        """
        if not HAS_OPTUNA:
            return ModelOptimizationResult(
                model_name=model_name,
                best_params={},
                best_score=0.0,
                n_trials=0,
            )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        trials_to_run = max(1, self.n_trials)
        study.optimize(
            objective,
            n_trials=trials_to_run,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        best_params = study.best_params if len(study.trials) > 0 else {}
        best_score = float(study.best_value) if len(study.trials) > 0 else 0.0
        is_valid = np.isfinite(best_score) and best_score > -1e8

        return ModelOptimizationResult(
            model_name=model_name,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials),
            is_valid=is_valid,
            error=None if is_valid else "All optimization trials failed.",
        )
