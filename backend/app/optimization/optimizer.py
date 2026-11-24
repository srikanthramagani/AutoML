"""
HyperparameterOptimizer
=======================
Responsibility: Orchestrate Optuna hyperparameter optimization studies for model candidates.
"""

from dataclasses import dataclass
import time
from pathlib import Path
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
    elapsed_seconds: float = 0.0
    study_name: Optional[str] = None
    direction: str = "maximize"
    optimization_history: Optional[List[Dict[str, Any]]] = None
    parameter_importance: Optional[Dict[str, float]] = None


class HyperparameterOptimizer:
    """
    Run Optuna HPO for model candidates with graceful fallback when Optuna is not installed.
    """

    def __init__(
        self,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        random_state: int = 42,
        storage_url: Optional[str] = None,
        pruner: bool = True,
    ) -> None:
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.storage_url = storage_url
        self.pruner = pruner

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
                best_score=float("nan"),
                n_trials=0,
                is_valid=False,
                error="Optuna is not installed.",
            )

        start = time.perf_counter()
        direction = getattr(objective, "direction", "maximize")
        study_name = getattr(objective, "study_name", model_name)
        storage = self.storage_url
        if storage and storage.startswith("sqlite:///"):
            db_path = Path(storage[len("sqlite:///"):])
            db_path.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=bool(storage),
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5) if self.pruner else optuna.pruners.NopPruner(),
        )

        trials_to_run = max(1, self.n_trials)
        before = len(study.trials)
        study.optimize(
            objective,
            n_trials=trials_to_run,
            timeout=self.timeout,
            show_progress_bar=False,
            catch=(Exception,),
        )

        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
        best_params = dict(study.best_params) if completed else {}
        best_score = float(study.best_value) if completed else float("nan")
        is_valid = bool(completed) and np.isfinite(best_score)

        try:
            importance = {key: float(value) for key, value in optuna.importance.get_param_importances(study).items()} if completed else {}
        except Exception:
            importance = {}
        return ModelOptimizationResult(
            model_name=model_name,
            best_params=best_params,
            best_score=best_score,
            n_trials=len(study.trials) - before,
            is_valid=is_valid,
            error=None if is_valid else "All optimization trials failed.",
            elapsed_seconds=round(time.perf_counter() - start, 4),
            study_name=study_name,
            direction=direction,
            optimization_history=[{"number": t.number, "value": t.value, "state": t.state.name, "params": t.params} for t in study.trials],
            parameter_importance=importance,
        )
