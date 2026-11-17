"""
AutoMLEngine
============
Responsibility: Top-level orchestrator for the Adaptive AutoML Platform supporting
Supervised (Classification & Regression) and Unsupervised (Clustering & Anomaly Detection).
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
from ..data.task_detector import TaskDetector, TaskInfo
from ..evaluation.cross_validation import CrossValidator, CVResult
from ..evaluation.fairness import FairnessAuditResult, FairnessAuditor
from ..evaluation.generalization import GeneralizationAuditor, GeneralizationDiagnostic
from ..evaluation.metrics import MetricsCalculator, MetricsResult
from ..evaluation.model_ranker import ModelRanker, RankedModel
from ..export.model_exporter import ModelExporter
from ..models.model_registry import ModelRegistry
from ..optimization.objective import OptunaObjective
from ..optimization.unsupervised_objective import UnsupervisedOptunaObjective
from ..optimization.optimizer import HyperparameterOptimizer, ModelOptimizationResult
from ..optimization.search_space import SearchSpace
from ..preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from ..training.final_training import FinalTrainer
from ..training.trainer import Trainer


class AutoMLEngine:
    """
    Top-level AutoML orchestrator supporting classification, regression, and unsupervised learning.
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
        self.final_training_time_seconds_: Optional[float] = None
        self.model_records_: List[Dict[str, Any]] = []
        self.fairness_audit_: Optional[FairnessAuditResult] = None
        self.feature_columns_: List[str] = []
        self.prediction_series_: Optional[List[Dict[str, Any]]] = None
        self.generalization_audit_: Optional[GeneralizationDiagnostic] = None
        self.candidate_diagnostics_: Dict[str, Any] = {}

    def fit(
        self,
        csv_path: str,
        target_column: Optional[str] = None,
        export_filename: str = "trained_pipeline",
        keep_identifier_columns: Optional[List[str]] = None,
        task: Optional[str] = None,
    ) -> "AutoMLEngine":
        """
        Run the complete Phase 1 AutoML workflow for Supervised or Unsupervised tasks.
        """
        cfg = self.settings
        task_mode = task or cfg.task

        print("=" * 65)
        print("  Adaptive AutoML Platform - Commencing AutoML Run")
        print("=" * 65)

        # ── Step 1: Load Dataset ──────────────────────────────────────────
        print("[AutoML] Step 1/11 - Loading dataset...")
        is_unsupervised_request = task_mode in ("clustering", "anomaly_detection", "unsupervised") or (
            target_column is None and task_mode not in ("classification", "regression")
        )

        loader = DataLoader(target_column=target_column)
        df, load_result = loader.load(csv_path)

        # Determine target
        if is_unsupervised_request and (target_column is None or target_column == ""):
            target_col = None
        else:
            target_col = load_result.target_column

        profiler = DataProfiler()
        self.profile_ = profiler.profile(df, target_column=target_col)
        keep_columns = set(keep_identifier_columns or [])
        excluded_identifiers = [
            column for column in self.profile_.identifier_columns
            if column not in keep_columns and column != target_col
        ]
        self.profile_.identifier_columns = excluded_identifiers

        if target_col is not None and target_col in df.columns:
            X, y = loader.split_features_target(df, target_col, excluded_columns=excluded_identifiers)
        else:
            cols_to_drop = [c for c in excluded_identifiers if c in df.columns]
            X = df.drop(columns=cols_to_drop).copy()
            y = None

        self.feature_columns_ = list(X.columns)
        sensitive_columns = list(cfg.sensitive_columns)
        missing_sensitive = [column for column in sensitive_columns if column not in X.columns]
        if missing_sensitive:
            raise ValueError(f"Sensitive columns not found in features: {missing_sensitive}")
        sensitive_values = X[sensitive_columns].copy() if sensitive_columns else None
        if sensitive_columns:
            X = X.drop(columns=sensitive_columns)

        print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns. Target: '{target_col or '[None - Unsupervised]'}'")

        # ── Step 2: Profile Dataset ───────────────────────────────────────
        print("[AutoML] Step 2/11 - Profiling dataset...")
        print(profiler.summary(self.profile_))
        if excluded_identifiers:
            print(f"  Excluded identifier columns: {', '.join(excluded_identifiers)}")

        # ── Step 3: Detect Task Type ──────────────────────────────────────
        print("[AutoML] Step 3/11 - Detecting task type & evaluation metric...")
        detector = TaskDetector()
        override_m = None if cfg.primary_metric == "auto" else cfg.primary_metric
        self.task_info_ = detector.detect(
            y=y,
            profile=self.profile_,
            override_metric=override_m,
            task_override=task_mode if task_mode != "auto" else None,
        )
        print(detector.describe(self.task_info_))

        is_unsupervised = self.task_info_.task_type in ("clustering", "anomaly_detection")

        # ── Step 4: Train / Holdout Split ─────────────────────────────────
        print(f"[AutoML] Step 4/11 - Creating holdout split ({int(cfg.test_size*100)}%)...")
        is_time_series = self.task_info_.task_type in ("time_series_forecasting", "time_series", "forecasting")

        if is_time_series:
            # Chronological temporal split (train on earlier data, validate on future)
            split_idx = int(len(X) * (1.0 - cfg.test_size))
            X_train, X_holdout = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
            y_train, y_holdout = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
            sensitive_train = sensitive_holdout = None
            effective_cv_folds = min(cfg.cv_folds, 5)
        elif not is_unsupervised:
            stratify = y if self.task_info_.task_type != "regression" else None
            try:
                split_values = [X, y] + ([sensitive_values] if sensitive_values is not None else [])
                split_result = train_test_split(
                    *split_values,
                    test_size=cfg.test_size,
                    random_state=cfg.random_seed,
                    stratify=stratify,
                )
                X_train, X_holdout, y_train, y_holdout = split_result[:4]
                sensitive_train = sensitive_holdout = None
                if sensitive_values is not None:
                    sensitive_train, sensitive_holdout = split_result[4:]
            except ValueError:
                # Fallback if class has too few instances for stratified split
                X_train, X_holdout, y_train, y_holdout = train_test_split(
                    X, y,
                    test_size=cfg.test_size,
                    random_state=cfg.random_seed,
                )
                sensitive_train = sensitive_holdout = None

            if self.task_info_.task_type != "regression":
                min_class_count = int(y_train.value_counts().min())
                effective_cv_folds = min(cfg.cv_folds, min_class_count)
                if effective_cv_folds < 2:
                    raise ValueError(
                        f"The training split has only {min_class_count} sample per class. "
                        "At least 2 samples per class are required for cross-validation; "
                        "use a larger dataset or a larger holdout split."
                    )
                if effective_cv_folds != cfg.cv_folds:
                    print(f"  Adjusting cross-validation folds from {cfg.cv_folds} to {effective_cv_folds} for the smallest class.")
            else:
                effective_cv_folds = min(cfg.cv_folds, len(y_train))
                if effective_cv_folds < 2:
                    raise ValueError("At least 2 training rows are required for cross-validation.")
        else:
            # Unsupervised split (no y target)
            X_train, X_holdout = train_test_split(
                X, test_size=cfg.test_size, random_state=cfg.random_seed
            )
            y_train = y_holdout = None
            sensitive_train = sensitive_holdout = None
            effective_cv_folds = cfg.cv_folds

        print(f"  Training samples: {len(X_train):,} | Holdout validation samples: {len(X_holdout):,}")

        # ── Step 5: Build Preprocessing Pipeline ──────────────────────────
        print("[AutoML] Step 5/11 - Building preprocessing pipeline candidate...")
        pp_builder = PreprocessingPipelineBuilder()
        preprocessing_pipeline = pp_builder.build(self.profile_, self.task_info_)
        print(f"  Pipeline steps: {[name for name, _ in preprocessing_pipeline.steps]}")

        # Speed Optimization: Pre-transform X_train for HPO caching
        if cfg.cache_preprocessed:
            print("[AutoML] Precomputing preprocessed features for high-speed HPO...")
            preprocessor_fitted = clone(preprocessing_pipeline).fit(X_train)
            X_train_transformed = preprocessor_fitted.transform(X_train)
        else:
            preprocessor_fitted = None
            X_train_transformed = None

        # ── Step 6: Candidate Model Pool ──────────────────────────────────
        print("[AutoML] Step 6/11 - Loading candidate model pool...")
        if is_unsupervised:
            task_family = self.task_info_.task_type
        elif is_time_series:
            task_family = "time_series_forecasting"
        else:
            task_family = "classification" if "classification" in self.task_info_.task_type else "regression"

        model_pool = ModelRegistry(random_state=cfg.random_seed).list_entries(task_family)
        print(f"  {len(model_pool)} candidate models available for {task_family}.")

        # ── Step 7: Hyperparameter Optimization ───────────────────────────
        trials_count = max(3, cfg.n_trials // 2) if cfg.fast_mode else cfg.n_trials
        print(f"[AutoML] Step 7/11 - Running Optuna HPO ({trials_count} trials/model)...")
        search_space = SearchSpace()
        optimizer = HyperparameterOptimizer(
            n_trials=trials_count,
            timeout=cfg.optimization_timeout,
            random_state=cfg.random_seed,
        )

        self.optimization_results_ = {}
        for entry in model_pool:
            m_name = entry["name"]
            est = entry["estimator"]

            if is_unsupervised:
                # Fast unsupervised objective on pre-transformed data
                feat_matrix = X_train_transformed if X_train_transformed is not None else X_train
                obj = UnsupervisedOptunaObjective(
                    estimator=est,
                    model_name=m_name,
                    X=feat_matrix,
                    task_type=task_family,
                    scoring=self.task_info_.primary_metric,
                    search_space=search_space,
                    random_state=cfg.random_seed,
                )
            else:
                obj = OptunaObjective(
                    preprocessing_pipeline=preprocessing_pipeline,
                    estimator=est,
                    model_name=m_name,
                    X=X_train,
                    y=y_train,
                    scoring=self.task_info_.primary_metric,
                    cv_folds=effective_cv_folds,
                    task_type=task_family,
                    search_space=search_space,
                    random_state=cfg.random_seed,
                    n_jobs=cfg.n_jobs,
                )

            opt_res = optimizer.optimize_model(obj, m_name)
            self.optimization_results_[m_name] = opt_res
            score_disp = f"{opt_res.best_score:.4f}" if np.isfinite(opt_res.best_score) else "n/a"
            if opt_res.best_params:
                print(f"    - {m_name:<28}: Best HPO Score = {score_disp} ({len(opt_res.best_params)} params tuned)")
            else:
                print(f"    - {m_name:<28}: Best HPO Score = {score_disp} (default params)")

        model_pool = [
            entry for entry in model_pool
            if self.optimization_results_[entry["name"]].is_valid
        ]
        if not model_pool:
            raise RuntimeError("No candidate model completed hyperparameter optimization successfully.")

        # ── Step 8: Cross-Validation & Validation Evaluation ──────────────
        self.cv_results_ = []
        auditor = GeneralizationAuditor()
        self.candidate_diagnostics_ = {}

        if not is_unsupervised:
            print(f"[AutoML] Step 8/11 - Performing {effective_cv_folds}-fold cross validation on tuned candidates...")
            validator = CrossValidator(
                n_folds=effective_cv_folds,
                scoring=self.task_info_.primary_metric,
                task_type=task_family,
                random_state=cfg.random_seed,
                n_jobs=cfg.n_jobs,
            )
            trainer = Trainer(random_state=cfg.random_seed)

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
                cv_res.training_time_seconds = training_result.fit_time_seconds
                self.cv_results_.append(cv_res)

                # Compute training partition performance & generalization diagnostics
                try:
                    if "classification" in task_family:
                        from sklearn.metrics import accuracy_score
                        train_s = float(accuracy_score(y_train, training_result.pipeline.predict(X_train)))
                        metric_n = "accuracy"
                        higher_better = True
                    else:
                        from sklearn.metrics import r2_score, mean_squared_error
                        preds_tr = training_result.pipeline.predict(X_train)
                        if self.task_info_.primary_metric in ("rmse", "mse", "mae"):
                            train_s = float(np.sqrt(mean_squared_error(y_train, preds_tr))) if self.task_info_.primary_metric == "rmse" else float(mean_squared_error(y_train, preds_tr))
                            metric_n = self.task_info_.primary_metric
                            higher_better = False
                        else:
                            train_s = float(r2_score(y_train, preds_tr))
                            metric_n = "r2"
                            higher_better = True

                    diag = auditor.evaluate_fit(
                        model_name=m_name,
                        train_score=train_s,
                        validation_score=cv_res.mean_score,
                        metric_name=metric_n,
                        higher_is_better=higher_better,
                    )
                    self.candidate_diagnostics_[m_name] = diag
                except Exception:
                    self.candidate_diagnostics_[m_name] = None
        else:
            print("[AutoML] Step 8/11 - Computing validation scores on tuned unsupervised candidates...")
            trainer = Trainer(random_state=cfg.random_seed)
            for entry in model_pool:
                m_name = entry["name"]
                est = entry["estimator"]
                best_p = self.optimization_results_[m_name].best_params
                best_s = self.optimization_results_[m_name].best_score
                cv_res = CVResult(
                    model_name=m_name,
                    scores=[best_s],
                    mean_score=best_s,
                    std_score=0.0,
                    scoring=self.task_info_.primary_metric,
                    n_folds=1,
                    is_valid=True,
                    training_time_seconds=self.optimization_results_[m_name].elapsed_seconds,
                )
                self.cv_results_.append(cv_res)
                diag = auditor.evaluate_fit(
                    model_name=m_name,
                    train_score=best_s,
                    validation_score=best_s,
                    metric_name=self.task_info_.primary_metric,
                    higher_is_better=True,
                )
                self.candidate_diagnostics_[m_name] = diag

        self.model_records_ = [
            {
                "model_name": result.model_name,
                "optimization_score": self.optimization_results_[result.model_name].best_score,
                "optimization_trials": self.optimization_results_[result.model_name].n_trials,
                "optimization_time_seconds": self.optimization_results_[result.model_name].elapsed_seconds,
                "best_params": self.optimization_results_[result.model_name].best_params,
                "cv_mean_score": result.mean_score,
                "cv_std_score": result.std_score,
                "cv_folds": result.n_folds,
                "train_score": self.candidate_diagnostics_.get(result.model_name).train_score if self.candidate_diagnostics_.get(result.model_name) else None,
                "generalization_gap": self.candidate_diagnostics_.get(result.model_name).generalization_gap if self.candidate_diagnostics_.get(result.model_name) else None,
                "fit_status": self.candidate_diagnostics_.get(result.model_name).status if self.candidate_diagnostics_.get(result.model_name) else "well_fitted",
                "fit_risk_level": self.candidate_diagnostics_.get(result.model_name).risk_level if self.candidate_diagnostics_.get(result.model_name) else "low",
                "training_time_seconds": result.training_time_seconds,
                "is_valid": result.is_valid,
                "error": result.error,
            }
            for result in self.cv_results_
        ]

        # ── Step 9: Model Ranking ─────────────────────────────────────────
        print("[AutoML] Step 9/11 - Ranking models & generating leaderboard...")
        ranker = ModelRanker(higher_is_better=True)
        self.ranked_models_ = ranker.rank(self.cv_results_, diagnostics=self.candidate_diagnostics_)
        print(ranker.leaderboard_str(self.ranked_models_))

        winner = self.ranked_models_[0]
        self.best_model_name_ = winner.model_name
        self.best_params_ = self.optimization_results_[self.best_model_name_].best_params
        print(f"  Winner: {self.best_model_name_} (Score = {winner.mean_score:.4f})")

        # ── Step 10: Final Retraining ─────────────────────────────────────
        print(f"[AutoML] Step 10/11 - Retraining winning pipeline ({self.best_model_name_}) on entire training set...")
        winner_entry = next(m for m in model_pool if m["name"] == self.best_model_name_)
        winning_estimator = clone(winner_entry["estimator"])
        if self.best_params_:
            winning_estimator.set_params(**self.best_params_)

        full_unfitted_pipeline = Pipeline(
            steps=list(preprocessing_pipeline.steps) + [("estimator", winning_estimator)]
        )
        final_trainer = FinalTrainer(random_state=cfg.random_seed)
        if not is_unsupervised:
            self.final_pipeline_ = final_trainer.fit(full_unfitted_pipeline, X_train, y_train=y_train)
        else:
            self.final_pipeline_ = final_trainer.fit(full_unfitted_pipeline, X_train, y_train=None)
        self.final_training_time_seconds_ = final_trainer.fit_time_seconds_

        # ── Step 11: Holdout Validation & Export ──────────────────────────
        print("[AutoML] Step 11/11 - Evaluating holdout test partition & exporting pipeline...")
        calc = MetricsCalculator(task_type=self.task_info_.task_type)

        if not is_unsupervised:
            y_pred = self.final_pipeline_.predict(X_holdout)
            y_proba = None
            if task_family == "classification" and hasattr(self.final_pipeline_, "predict_proba"):
                try:
                    y_proba = self.final_pipeline_.predict_proba(X_holdout)
                except Exception:
                    y_proba = None

            # Generate prediction series for visualization
            try:
                y_true_list = [float(v) for v in np.asarray(y_holdout)]
                y_pred_list = [float(v) for v in np.asarray(y_pred)]
                # If there's an index or date column
                index_labels = [str(idx) for idx in X_holdout.index]
                self.prediction_series_ = [
                    {
                        "index": index_labels[i],
                        "actual": round(y_true_list[i], 4),
                        "predicted": round(y_pred_list[i], 4),
                        "residual": round(y_true_list[i] - y_pred_list[i], 4),
                    }
                    for i in range(min(120, len(y_true_list)))  # sample first 120 points for chart responsiveness
                ]
            except Exception:
                self.prediction_series_ = []

            self.holdout_metrics_ = calc.compute(
                model_name=self.best_model_name_,
                y_true=y_holdout,
                y_pred=y_pred,
                y_proba=y_proba,
                n_features=X_holdout.shape[1],
            )
            if sensitive_holdout is not None:
                self.fairness_audit_ = FairnessAuditor().audit(
                    sensitive_holdout,
                    y_holdout,
                    y_pred,
                    sensitive_columns,
                )
            else:
                self.fairness_audit_ = FairnessAuditResult(
                    status="not_run",
                    sensitive_columns=[],
                    groups={},
                    note="Provide Settings.sensitive_columns to run group fairness diagnostics.",
                )
        else:
            # Unsupervised holdout metrics
            fitted_est = self.final_pipeline_.named_steps["estimator"]
            preprocessed_holdout = self.final_pipeline_.named_steps["feature_scaler"].transform(
                self.final_pipeline_.named_steps["categorical_encoder"].transform(
                    self.final_pipeline_.named_steps["missing_value_handler"].transform(
                        self.final_pipeline_.named_steps["outlier_handler"].transform(X_holdout)
                        if "outlier_handler" in self.final_pipeline_.named_steps else X_holdout
                    )
                )
            )
            if hasattr(self.final_pipeline_, "predict"):
                try:
                    holdout_preds = self.final_pipeline_.predict(X_holdout)
                except Exception:
                    holdout_preds = getattr(fitted_est, "labels_", None)
                    if holdout_preds is not None and len(holdout_preds) != len(X_holdout):
                        # Transductive clustering (e.g. DBSCAN) without separate predict on holdout:
                        # fit_predict on preprocessed holdout to evaluate holdout cluster structure
                        holdout_preds = clone(fitted_est).fit_predict(preprocessed_holdout)
            elif hasattr(fitted_est, "labels_"):
                # Fit clone on holdout for cluster quality measurement
                holdout_preds = clone(fitted_est).fit_predict(preprocessed_holdout)
            else:
                holdout_preds = np.zeros(len(X_holdout), dtype=int)

            scores = None
            if hasattr(fitted_est, "decision_function"):
                try:
                    scores = fitted_est.decision_function(preprocessed_holdout)
                except Exception:
                    scores = None

            self.holdout_metrics_ = calc.compute(
                model_name=self.best_model_name_,
                y_true=preprocessed_holdout,
                y_pred=holdout_preds,
                y_proba=scores,
            )
            self.fairness_audit_ = FairnessAuditResult(
                status="not_run",
                sensitive_columns=[],
                groups={},
                note="Fairness audit not applicable to unsupervised learning without ground truth labels.",
            )

        # Generalization & Fit Diagnostics for winning model
        winning_cv_score = next(r.mean_score for r in self.cv_results_ if r.model_name == self.best_model_name_)
        self.generalization_audit_ = auditor.audit_pipeline(
            pipeline=self.final_pipeline_,
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            task_type=self.task_info_.task_type,
            primary_metric=self.task_info_.primary_metric,
            cv_score=winning_cv_score,
            model_name=self.best_model_name_,
        )

        print("  Holdout Performance:")
        for met_name, met_val in self.holdout_metrics_.metrics.items():
            print(f"    - {met_name:<26}: {met_val:.4f}")

        print("  Generalization & Fit Audit:")
        print(f"    - Fit Status                : {self.generalization_audit_.status.replace('_', ' ').title()} (Risk: {self.generalization_audit_.risk_level.title()})")
        print(f"    - Training Score            : {self.generalization_audit_.train_score:.4f}")
        print(f"    - CV Score                  : {self.generalization_audit_.validation_score:.4f}")
        if self.generalization_audit_.holdout_score is not None:
            print(f"    - Holdout Score             : {self.generalization_audit_.holdout_score:.4f}")
        print(f"    - Generalization Gap        : {self.generalization_audit_.generalization_gap:.4f} ({self.generalization_audit_.gap_percentage:.1f}%)")
        print(f"    - Diagnosis                 : {self.generalization_audit_.explanation}")

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
                "generalization_audit": self.generalization_audit_.__dict__ if self.generalization_audit_ else None,
                "optimization_trials": trials_count,
                "final_training_time_seconds": self.final_training_time_seconds_,
                "model_records": self.model_records_,
                "fairness_audit": self.fairness_audit_.__dict__,
            },
            filename=export_filename,
        )

        print("=" * 65)
        print(f"  AutoML Run Complete! Best Model: {self.best_model_name_}")
        print(f"  Exported Pipeline Artifact: {self.export_path_}")
        print("=" * 65)

        return self
