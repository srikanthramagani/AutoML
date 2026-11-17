"""
Settings & Global Configuration
================================
Central configuration module for the Adaptive AutoML Platform.

All experiment-wide hyperparameters, random seeds, file paths, and
behavioral flags are defined here so every module can import a single
source of truth instead of hard-coding values.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class Settings:
    """
    Master configuration dataclass.

    Attributes
    ----------
    random_seed : int
        Global random seed for reproducibility across numpy, sklearn, optuna.
    test_size : float
        Fraction of data held out as the final validation set (0 < test_size < 1).
    cv_folds : int
        Number of folds used in cross-validation during model evaluation.
    n_trials : int
        Number of Optuna trials per model during hyperparameter optimization.
    optimization_timeout : Optional[int]
        Maximum seconds per Optuna study; None means no timeout.
    primary_metric : str
        The sklearn-compatible scoring string used to rank models
        (e.g. 'f1_weighted', 'roc_auc', 'r2', 'neg_mean_squared_error').
    n_jobs : int
        Parallelism level for sklearn estimators (-1 = all cores).
    verbosity : int
        Logging verbosity (0 = silent, 1 = info, 2 = debug).
    export_dir : str
        Directory where trained pipeline .pkl files are saved.
    datasets_raw_dir : str
        Directory containing raw CSV files.
    datasets_processed_dir : str
        Directory for preprocessed/cached data.
    experiments_dir : str
        Directory for experiment logs and results.
    """

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Data splitting ---
    test_size: float = 0.2

    # --- Cross-validation ---
    cv_folds: int = 5

    # --- Optuna HPO ---
    n_trials: int = 50
    optimization_timeout: Optional[int] = None  # seconds

    # --- Evaluation & Task ---
    task: str = "auto"  # 'auto', 'classification', 'regression', 'clustering', 'anomaly_detection'
    primary_metric: str = "auto"  # 'auto' → resolved by TaskDetector

    # --- Compute & Speed ---
    n_jobs: int = -1
    verbosity: int = 1
    fast_mode: bool = False
    cache_preprocessed: bool = True

    # --- Paths (relative to project root) ---
    export_dir: str = os.path.join("models")
    datasets_raw_dir: str = os.path.join("datasets", "raw")
    datasets_processed_dir: str = os.path.join("datasets", "processed")
    experiments_dir: str = os.path.join("experiments")
    sensitive_columns: List[str] = field(default_factory=list)
    keep_identifier_columns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        if self.optimization_timeout is not None and self.optimization_timeout <= 0:
            raise ValueError("optimization_timeout must be positive when provided.")
        if not isinstance(self.primary_metric, str) or not self.primary_metric.strip():
            raise ValueError("primary_metric must be a non-empty string.")


# Module-level singleton — import this in other modules
default_settings = Settings()


class BackendSettings(BaseSettings):
    """Environment-backed web and storage configuration."""

    database_url: str = "sqlite:///./adaptive_automl.db"
    upload_directory: str = "backend/uploads"
    model_directory: str = "backend/trained_models"
    experiment_directory: str = "backend/experiments"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


backend_settings = BackendSettings()
