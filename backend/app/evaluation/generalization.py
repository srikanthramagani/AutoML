"""
GeneralizationAuditor
=====================
Responsibility: Detect overfitting, underfitting, and evaluate model generalization
by comparing training partition, cross-validation, and holdout performance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, silhouette_score


@dataclass
class GeneralizationDiagnostic:
    """
    Diagnostic assessment of a model's bias-variance tradeoff.
    """
    model_name: str
    status: str  # 'well_fitted', 'overfitting', 'underfitting'
    train_score: float
    validation_score: float
    holdout_score: Optional[float] = None
    generalization_gap: float = 0.0
    gap_percentage: float = 0.0
    risk_level: str = "low"  # 'low', 'moderate', 'high'
    metric_name: str = "score"
    explanation: str = ""
    recommendations: List[str] = field(default_factory=list)


class GeneralizationAuditor:
    """
    Audit training vs validation vs holdout divergence to identify
    overfitting (high variance) and underfitting (high bias).
    """

    def __init__(
        self,
        overfit_gap_threshold: float = 0.12,
        underfit_score_threshold: float = 0.60,
    ) -> None:
        self.overfit_gap_threshold = overfit_gap_threshold
        self.underfit_score_threshold = underfit_score_threshold

    def evaluate_fit(
        self,
        model_name: str,
        train_score: float,
        validation_score: float,
        holdout_score: Optional[float] = None,
        metric_name: str = "accuracy",
        higher_is_better: bool = True,
    ) -> GeneralizationDiagnostic:
        """
        Evaluate fit status based on training and validation performance.
        """
        # Calibrate gap so positive gap means train is performing better than validation
        if higher_is_better:
            gap = float(train_score - validation_score)
            base_score = max(abs(train_score), 1e-4)
            gap_pct = round((gap / base_score) * 100, 2)
            eval_score = validation_score
            train_eval = train_score
        else:
            # For error metrics like RMSE/MAE, lower is better. Overfit means train error << val error.
            gap = float(validation_score - train_score)
            base_score = max(abs(train_score), 1e-4)
            gap_pct = round((gap / base_score) * 100, 2)
            eval_score = -validation_score
            train_eval = -train_score

        recommendations = []

        # Determine status
        if gap > self.overfit_gap_threshold or gap_pct > 20.0:
            status = "overfitting"
            risk_level = "high" if (gap > self.overfit_gap_threshold * 1.8 or gap_pct > 35.0) else "moderate"
            explanation = (
                f"Model exhibits high variance (overfitting). Training score ({train_score:.4f}) is noticeably higher "
                f"than cross-validation score ({validation_score:.4f}), indicating memorization of training samples."
            )
            recommendations = [
                "Increase estimator regularization (e.g., higher L1/L2 penalties, higher min_samples_leaf).",
                "Constrain model capacity (e.g., lower max_depth or reduce number of boosting estimators).",
                "Acquire more training observations or apply feature selection to eliminate noisy predictors.",
            ]
        elif higher_is_better and (train_eval < self.underfit_score_threshold and eval_score < self.underfit_score_threshold):
            status = "underfitting"
            risk_level = "high" if eval_score < 0.40 else "moderate"
            explanation = (
                f"Model exhibits high bias (underfitting). Both training score ({train_score:.4f}) and validation "
                f"score ({validation_score:.4f}) fall below expected predictive thresholds."
            )
            recommendations = [
                "Increase model capacity (e.g., deeper trees, non-linear kernel, gradient boosting ensemble).",
                "Engineer richer interaction or polynomial domain features.",
                "Relax regularization constraints to allow the model to capture non-linear relationships.",
            ]
        else:
            status = "well_fitted"
            risk_level = "low"
            explanation = (
                f"Optimal generalization fit. Training score ({train_score:.4f}) and validation score "
                f"({validation_score:.4f}) align closely (generalization gap = {gap:.4f})."
            )
            recommendations = [
                "Model shows well-calibrated bias-variance tradeoff; safe for production deployment.",
            ]

        return GeneralizationDiagnostic(
            model_name=model_name,
            status=status,
            train_score=round(float(train_score), 4),
            validation_score=round(float(validation_score), 4),
            holdout_score=round(float(holdout_score), 4) if holdout_score is not None else None,
            generalization_gap=round(gap, 4),
            gap_percentage=gap_pct,
            risk_level=risk_level,
            metric_name=metric_name,
            explanation=explanation,
            recommendations=recommendations,
        )

    def audit_pipeline(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series],
        X_holdout: pd.DataFrame,
        y_holdout: Optional[pd.Series],
        task_type: str,
        primary_metric: str,
        cv_score: float,
        model_name: str = "winner",
    ) -> GeneralizationDiagnostic:
        """
        Audit a fully fitted pipeline across train and holdout datasets.
        """
        is_classification = "classification" in task_type
        is_unsupervised = task_type in ("clustering", "anomaly_detection")

        if is_unsupervised:
            # Unsupervised silhouette check
            try:
                train_labels = pipeline.predict(X_train) if hasattr(pipeline, "predict") else pipeline.fit_predict(X_train)
                holdout_labels = pipeline.predict(X_holdout) if hasattr(pipeline, "predict") else pipeline.fit_predict(X_holdout)
                
                # Preprocess for silhouette calculation
                prep = Pipeline(steps=pipeline.steps[:-1])
                X_tr_t = prep.transform(X_train)
                X_ho_t = prep.transform(X_holdout)
                
                train_s = float(silhouette_score(X_tr_t, train_labels)) if len(set(train_labels)) > 1 else 0.0
                holdout_s = float(silhouette_score(X_ho_t, holdout_labels)) if len(set(holdout_labels)) > 1 else 0.0
            except Exception:
                train_s = cv_score
                holdout_s = cv_score
            higher_is_better = True
        elif is_classification:
            train_preds = pipeline.predict(X_train)
            holdout_preds = pipeline.predict(X_holdout)
            train_s = float(accuracy_score(y_train, train_preds))
            holdout_s = float(accuracy_score(y_holdout, holdout_preds))
            higher_is_better = True
        else:
            # Regression or Time Series
            train_preds = pipeline.predict(X_train)
            holdout_preds = pipeline.predict(X_holdout)
            if primary_metric in ("rmse", "mse", "mae"):
                train_s = float(np.sqrt(mean_squared_error(y_train, train_preds))) if primary_metric == "rmse" else float(mean_squared_error(y_train, train_preds))
                holdout_s = float(np.sqrt(mean_squared_error(y_holdout, holdout_preds))) if primary_metric == "rmse" else float(mean_squared_error(y_holdout, holdout_preds))
                higher_is_better = False
            else:
                train_s = float(r2_score(y_train, train_preds))
                holdout_s = float(r2_score(y_holdout, holdout_preds))
                higher_is_better = True

        return self.evaluate_fit(
            model_name=model_name,
            train_score=train_s,
            validation_score=cv_score,
            holdout_score=holdout_s,
            metric_name="accuracy" if is_classification else primary_metric,
            higher_is_better=higher_is_better,
        )
