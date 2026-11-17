"""
Optimization package.
Provides Optuna-based hyperparameter optimization:
  - SearchSpace: defines HPO search spaces per model.
  - Optimizer: orchestrates Optuna studies.
  - Objective: the Optuna objective function wrapping CV scoring.
"""
from .search_space import SearchSpace
from .optimizer import HyperparameterOptimizer
from .objective import OptunaObjective

__all__ = ["SearchSpace", "HyperparameterOptimizer", "OptunaObjective"]
