"""Budget-matched Optuna, grid, and randomized search comparisons."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, cross_val_score
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline


_ERROR_METRICS = {"rmse", "mse", "mae", "root_mean_squared_error", "mean_squared_error", "mean_absolute_error"}


def _scorer_name(metric: str) -> str:
    return {
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "root_mean_squared_error": "neg_root_mean_squared_error",
        "mean_squared_error": "neg_mean_squared_error",
        "mean_absolute_error": "neg_mean_absolute_error",
    }.get(metric.lower(), metric)


def _actual_score(score: float, metric: str) -> float:
    metric_name = metric.lower()[4:] if metric.lower().startswith("neg_") else metric.lower()
    return -float(score) if metric_name in _ERROR_METRICS else float(score)


def _suggest(trial: optuna.Trial, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    values = {}
    for name, spec in space.items():
        if "grid" in spec:
            values[name] = trial.suggest_categorical(name, list(spec["grid"]))
            continue
        kind = spec["type"]
        if kind == "int":
            values[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
        elif kind == "float":
            values[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False), step=spec.get("step"))
        elif kind == "categorical":
            values[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported search parameter type: {kind}")
    return values


def _random_distributions(space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    result = {}
    for name, spec in space.items():
        if "grid" in spec:
            result[name] = list(spec["grid"])
            continue
        kind = spec["type"]
        if kind == "categorical":
            result[name] = list(spec["choices"])
        elif kind == "int":
            if spec.get("log"):
                result[name] = np.unique(np.geomspace(spec["low"], spec["high"], 100).astype(int)).tolist()
            else:
                result[name] = randint(spec["low"], spec["high"] + 1)
        elif kind == "float":
            result[name] = loguniform(spec["low"], spec["high"]) if spec.get("log") else uniform(spec["low"], spec["high"] - spec["low"])
    return result


def _grid_values(space: Dict[str, Dict[str, Any]], budget: int) -> List[Dict[str, List[Any]]]:
    grid = {}
    for name, spec in space.items():
        if "grid" in spec:
            grid[name] = list(spec["grid"])
        elif spec["type"] == "categorical":
            grid[name] = list(spec["choices"])
        elif spec["type"] == "int":
            grid[name] = np.unique(np.linspace(spec["low"], spec["high"], min(3, budget)).round().astype(int)).tolist()
        else:
            grid[name] = np.linspace(spec["low"], spec["high"], min(3, budget)).tolist()
    # Grid search can otherwise exceed the shared trial budget. Keep a deterministic
    # prefix of the Cartesian grid and report the actual number evaluated.
    configs = list(ParameterGrid(grid))[:budget]
    return [{key: [value] for key, value in config.items()} for config in configs]


def compare_optimizers(
    pipeline: Pipeline,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    space: Dict[str, Dict[str, Any]],
    metric: str,
    cv: Any,
    budget: int = 12,
    random_state: int = 42,
    timeout: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Compare optimizers on fixed train/test data, pipeline, CV, and parameter bounds."""
    if budget < 1:
        raise ValueError("budget must be positive")
    if any("grid" not in spec or not spec["grid"] for spec in space.values()):
        raise ValueError("Controlled comparisons require a non-empty finite 'grid' for every parameter so all optimizers search identical configurations.")
    scorer = _scorer_name(metric)
    metric_name = metric.lower()[4:] if metric.lower().startswith("neg_") else metric.lower()
    direction = "minimize" if metric_name in _ERROR_METRICS else "maximize"
    total_configurations = int(np.prod([len(spec["grid"]) for spec in space.values()]))
    effective_budget = min(budget, total_configurations)
    results: List[Dict[str, Any]] = []

    def record(name: str, best_params: Dict[str, Any], best_score: float, evaluations: int, elapsed: float):
        fitted = clone(pipeline).set_params(**best_params)
        fitted.fit(X_train, y_train)
        test = get_scorer(scorer)(fitted, X_test, y_test)
        results.append({
            "optimizer": name,
            "validation_score": _actual_score(best_score, metric),
            "test_score": _actual_score(float(test), metric),
            "best_params": best_params,
            "configurations_evaluated": int(evaluations),
            "optimization_seconds": float(elapsed),
            "direction": direction,
            "metric": metric,
        })

    start = time.perf_counter()
    grid = GridSearchCV(clone(pipeline), _grid_values(space, effective_budget), scoring=scorer, cv=cv, n_jobs=1, error_score=np.nan)
    grid.fit(X_train, y_train)
    record("GridSearchCV", grid.best_params_, float(grid.best_score_), len(grid.cv_results_["params"]), time.perf_counter() - start)

    start = time.perf_counter()
    random = RandomizedSearchCV(clone(pipeline), _random_distributions(space), n_iter=effective_budget, scoring=scorer, cv=cv, random_state=random_state, n_jobs=1, error_score=np.nan)
    random.fit(X_train, y_train)
    record("RandomizedSearchCV", random.best_params_, float(random.best_score_), len(random.cv_results_["params"]), time.perf_counter() - start)

    start = time.perf_counter()
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))

    def objective(trial: optuna.Trial) -> float:
        candidate = clone(pipeline).set_params(**_suggest(trial, space))
        scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring=scorer, n_jobs=1, error_score="raise")
        score = float(np.mean(scores))
        return _actual_score(score, metric)

    study.optimize(objective, n_trials=effective_budget, timeout=timeout, show_progress_bar=False, catch=(Exception,))
    elapsed = time.perf_counter() - start
    if not study.best_trials:
        raise RuntimeError("Optuna completed no valid benchmark trials.")
    record("Optuna TPE", study.best_params, float(study.best_value), len(study.trials), elapsed)
    return results
