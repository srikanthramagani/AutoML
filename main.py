"""
main.py
=======
Adaptive AutoML Platform ? Phase 1 CLI Entry Point.
"""

import argparse
import sys
from backend.app.automl.engine import AutoMLEngine
from backend.app.config.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="adaptive-automl",
        description="Adaptive AutoML Platform - Phase 1 CLI",
    )
    parser.add_argument(
        "--csv", required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--target", default=None, help="Target column name (default: last column)."
    )
    parser.add_argument(
        "--output", default="trained_pipeline",
        help="Base filename for the exported .pkl (default: trained_pipeline).",
    )
    parser.add_argument(
        "--trials", type=int, default=15,
        help="Number of Optuna HPO trials per model (default: 15).",
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--metric", default="auto",
        help="Evaluation metric ('auto', 'roc_auc', 'f1_weighted', 'r2', 'silhouette_score', etc.).",
    )
    parser.add_argument(
        "--task", default="auto",
        choices=["auto", "classification", "regression", "clustering", "anomaly_detection", "time_series", "time_series_forecasting"],
        help="Task type ('auto', 'classification', 'regression', 'clustering', 'anomaly_detection', 'time_series').",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: optimizes search budget and pre-caches transformations for rapid turnaround.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = Settings(
        random_seed=args.seed,
        n_trials=args.trials,
        cv_folds=args.folds,
        primary_metric=args.metric,
        task=args.task,
        fast_mode=args.fast,
    )

    engine = AutoMLEngine(settings=settings)
    engine.fit(
        csv_path=args.csv,
        target_column=args.target,
        export_filename=args.output,
        task=args.task if args.task != "auto" else None,
    )


if __name__ == "__main__":
    main()
