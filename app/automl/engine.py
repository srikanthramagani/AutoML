"""
AutoMLEngine
============
Responsibility: Top-level orchestrator for the Adaptive AutoML Platform.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..config.settings import Settings, default_settings
from ..data.loader import DataLoader
from ..data.profiler import DataProfiler, DataProfile
from backend.app.data.modalities import detect_modalities
from ..data.task_detector import TaskDetector, TaskInfo
from ..evaluation.cross_validation import CrossValidator, CVResult
from ..evaluation.metrics import MetricsCalculator, MetricsResult
from ..evaluation.model_ranker import ModelRanker, RankedModel
from ..export.model_exporter import ModelExporter
from ..models.classification import get_classification_models
from ..models.regression import get_regression_models
from ..optimization.objective import OptunaObjective
from ..optimization.optimizer import HyperparameterOptimizer, ModelOptimizationResult
from ..optimization.search_space import SearchSpace
from ..preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from ..training.final_training import FinalTrainer
from ..training.trainer import Trainer


class AutoMLEngine:
    """
    Top-level AutoML orchestrator.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or default_settings

        # State attributes
        self.profile_: Optional[DataProfile] = None
        self.task_info_: Optional[TaskInfo] = None
        self.optimization_results_: Dict[str, ModelOptimizationResult] = {}
        self.cv_results_: List[CVResult] = []
        self.ranked_models_: List[RankedModel] = []
        self.best_model_name_: Optional[str] = None
        self.best_params_: Dict[str, Any] = {}
        self.final_pipeline_: Optional[Pipeline] = None
        self.holdout_metrics_: Optional[MetricsResult] = None
        self.export_path_: Optional[str] = None
        self.y_holdout_: Optional[pd.Series] = None
        self.y_pred_holdout_: Optional[np.ndarray] = None
        self.y_proba_holdout_: Optional[np.ndarray] = None

    def fit(
        self,
        csv_path: str,
        target_column: Optional[str] = None,
        export_filename: str = "trained_pipeline",
    ) -> "AutoMLEngine":
        """
        Run the complete 11-step Phase 1 AutoML workflow.
        """
        cfg = self.settings

        print("=" * 65)
        print("  Adaptive AutoML Platform - Commencing Phase 1 Run")
        print("=" * 65)

        # ?? Step 1: Load Dataset ??????????????????????????????????????????
        print("[AutoML] Step 1/11 - Loading dataset...")
        loader = DataLoader(target_column=target_column)
        df, load_result = loader.load(csv_path)
        target_col = load_result.target_column
        modality_report = detect_modalities(df, target=target_col)
        if modality_report.text:
            raise ValueError(
                "Natural-language columns were detected and cannot enter the tabular-only pipeline: "
                f"{modality_report.text}. Use the multimodal workflow or explicitly reassign the columns."
            )
        X, y = loader.split_features_target(df, target_col)
        print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns. Target: '{target_col}'")

        # ?? Step 2: Profile Dataset ???????????????????????????????????????
        print("[AutoML] Step 2/11 - Profiling dataset...")
        profiler = DataProfiler()
        self.profile_ = profiler.profile(df)
        print(profiler.summary(self.profile_))

        # ?? Step 3: Detect Task Type ??????????????????????????????????????
        print("[AutoML] Step 3/11 - Detecting task type & evaluation metric...")
        detector = TaskDetector()
        override_m = None if cfg.primary_metric == "auto" else cfg.primary_metric
        self.task_info_ = detector.detect(y, self.profile_, override_metric=override_m)
        print(detector.describe(self.task_info_))

        # ?? Step 4: Train / Holdout Split ?????????????????????????????????
        print(f"[AutoML] Step 4/11 - Creating holdout split ({int(cfg.test_size*100)}%)...")
        stratify = y if self.task_info_.task_type != "regression" else None
        try:
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y,
                test_size=cfg.test_size,
                random_state=cfg.random_seed,
                stratify=stratify,
            )
        except ValueError:
            # Fallback if class has too few instances for stratified split
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y,
                test_size=cfg.test_size,
                random_state=cfg.random_seed,
            )
        if self.task_info_.task_type != "regression":
            min_class_count = int(y_train.value_counts().min())
            if min_class_count < cfg.cv_folds:
                raise ValueError(
                    f"Each training class needs at least {cfg.cv_folds} samples "
                    f"for {cfg.cv_folds}-fold cross-validation; found {min_class_count}."
                )
        print(f"  Training samples: {len(X_train):,} | Holdout validation samples: {len(X_holdout):,}")

        # ?? Step 5: Build Preprocessing Pipeline ??????????????????????????
        print("[AutoML] Step 5/11 - Building preprocessing pipeline candidate...")
        pp_builder = PreprocessingPipelineBuilder()
        preprocessing_pipeline = pp_builder.build(self.profile_, self.task_info_)
        print(f"  Pipeline steps: {[name for name, _ in preprocessing_pipeline.steps]}")

        # ?? Step 6: Candidate Model Pool ??????????????????????????????????
        print("[AutoML] Step 6/11 - Loading candidate model pool...")
        task_family = "classification" if "classification" in self.task_info_.task_type else "regression"
        if task_family == "classification":
            model_pool = get_classification_models(random_state=cfg.random_seed)
        else:
            model_pool = get_regression_models(random_state=cfg.random_seed)
        print(f"  {len(model_pool)} candidate models available for {task_family}.")

        # ?? Step 7: Hyperparameter Optimization ???????????????????????????
        print(f"[AutoML] Step 7/11 - Running Optuna HPO ({cfg.n_trials} trials/model)...")
        search_space = SearchSpace()
        optimizer = HyperparameterOptimizer(
            n_trials=cfg.n_trials,
            timeout=cfg.optimization_timeout,
            random_state=cfg.random_seed,
        )

        self.optimization_results_ = {}
        for entry in model_pool:
            m_name = entry["name"]
            est = entry["estimator"]
            obj = OptunaObjective(
                preprocessing_pipeline=preprocessing_pipeline,
                estimator=est,
                model_name=m_name,
                X=X_train,
                y=y_train,
                scoring=self.task_info_.primary_metric,
                cv_folds=cfg.cv_folds,
                task_type=task_family,
                search_space=search_space,
                random_state=cfg.random_seed,
            )
            opt_res = optimizer.optimize_model(obj, m_name)
            self.optimization_results_[m_name] = opt_res
            if opt_res.best_params:
                print(f"    - {m_name:<28}: Best HPO Score = {opt_res.best_score:.4f} ({len(opt_res.best_params)} params tuned)")
            else:
                print(f"    - {m_name:<28}: Best HPO Score = {opt_res.best_score:.4f} (default params)")

        model_pool = [
            entry for entry in model_pool
            if self.optimization_results_[entry["name"]].is_valid
        ]
        if not model_pool:
            raise RuntimeError("No candidate model completed hyperparameter optimization successfully.")

        # ?? Step 8: Cross-Validation ??????????????????????????????????????
        print(f"[AutoML] Step 8/11 - Performing {cfg.cv_folds}-fold cross validation on tuned candidates...")
        validator = CrossValidator(
            n_folds=cfg.cv_folds,
            scoring=self.task_info_.primary_metric,
            task_type=task_family,
            random_state=cfg.random_seed,
        )
        trainer = Trainer(random_state=cfg.random_seed)
        self.cv_results_ = []

        for entry in model_pool:
            m_name = entry["name"]
            est = entry["estimator"]
            best_p = self.optimization_results_[m_name].best_params
            training_result = trainer.fit(
                preprocessing_pipeline,
                est,
                X_train,
                y_train,
                model_name=m_name,
                hyperparams=best_p,
            )
            cv_res = validator.validate(
                training_result.pipeline,
                X_train,
                y_train,
                model_name=m_name,
            )
            self.cv_results_.append(cv_res)

        # ?? Step 9: Model Ranking ?????????????????????????????????????????
        print("[AutoML] Step 9/11 - Ranking models & generating leaderboard...")
        ranker = ModelRanker(higher_is_better=True)
        self.ranked_models_ = ranker.rank(self.cv_results_)
        print(ranker.leaderboard_str(self.ranked_models_))

        winner = self.ranked_models_[0]
        self.best_model_name_ = winner.model_name
        self.best_params_ = self.optimization_results_[self.best_model_name_].best_params
        print(f"  Winner: {self.best_model_name_} (CV Mean = {winner.mean_score:.4f})")

        # ?? Step 10: Final Retraining ?????????????????????????????????????
        print(f"[AutoML] Step 10/11 - Retraining winning pipeline ({self.best_model_name_}) on entire training set...")
        winner_entry = next(m for m in model_pool if m["name"] == self.best_model_name_)
        winning_estimator = clone(winner_entry["estimator"])
        if self.best_params_:
            winning_estimator.set_params(**self.best_params_)

        full_unfitted_pipeline = Pipeline(
            steps=list(preprocessing_pipeline.steps) + [("estimator", winning_estimator)]
        )
        final_trainer = FinalTrainer(random_state=cfg.random_seed)
        self.final_pipeline_ = final_trainer.fit(full_unfitted_pipeline, X_train, y_train)

        # ?? Step 11: Holdout Validation & Export ??????????????????????????
        print("[AutoML] Step 11/11 - Evaluating holdout test partition & exporting pipeline...")
        y_pred = self.final_pipeline_.predict(X_holdout)
        self.y_holdout_ = y_holdout.copy()
        self.y_pred_holdout_ = np.asarray(y_pred)

        y_proba = None
        if task_family == "classification" and hasattr(self.final_pipeline_, "predict_proba"):
            try:
                y_proba = self.final_pipeline_.predict_proba(X_holdout)
            except Exception:
                y_proba = None
        self.y_proba_holdout_ = np.asarray(y_proba) if y_proba is not None else None

        calc = MetricsCalculator(task_type=self.task_info_.task_type)
        self.holdout_metrics_ = calc.compute(
            model_name=self.best_model_name_,
            y_true=y_holdout,
            y_pred=y_pred,
            y_proba=y_proba,
            n_features=X_holdout.shape[1],
        )

        print("  Holdout Test Set Performance:")
        for met_name, met_val in self.holdout_metrics_.metrics.items():
            print(f"    - {met_name:<20}: {met_val:.4f}")

        exporter = ModelExporter(export_dir=cfg.export_dir)
        self.export_path_ = exporter.export(
            pipeline=self.final_pipeline_,
            model_name=self.best_model_name_,
            metadata={
                "task_type": self.task_info_.task_type,
                "primary_metric": self.task_info_.primary_metric,
                "best_cv_score": winner.mean_score,
                "best_hyperparameters": self.best_params_,
                "holdout_metrics": self.holdout_metrics_.metrics,
            },
            filename=export_filename,
        )

        print("=" * 65)
        print(f"  AutoML Run Complete! Best Model: {self.best_model_name_}")
        print(f"  Exported Pipeline Artifact: {self.export_path_}")
        print("=" * 65)

        return self
