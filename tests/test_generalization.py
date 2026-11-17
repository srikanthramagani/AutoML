"""
Tests for GeneralizationAuditor, Overfit/Underfit Detection, and Expanded Metrics
"""

import numpy as np
import pandas as pd
from backend.app.evaluation.generalization import GeneralizationAuditor
from backend.app.evaluation.model_ranker import ModelRanker, RankedModel
from backend.app.evaluation.cross_validation import CVResult
from backend.app.evaluation.metrics import MetricsCalculator


def test_generalization_auditor_overfit():
    auditor = GeneralizationAuditor()
    diag = auditor.evaluate_fit(
        model_name="deep_tree",
        train_score=0.99,
        validation_score=0.68,
        metric_name="accuracy",
        higher_is_better=True,
    )
    assert diag.status == "overfitting"
    assert diag.risk_level == "high"
    assert diag.generalization_gap > 0.12
    assert len(diag.recommendations) >= 2


def test_generalization_auditor_underfit():
    auditor = GeneralizationAuditor()
    diag = auditor.evaluate_fit(
        model_name="linear_simple",
        train_score=0.45,
        validation_score=0.42,
        metric_name="accuracy",
        higher_is_better=True,
    )
    assert diag.status == "underfitting"
    assert "bias" in diag.explanation.lower()
    assert len(diag.recommendations) >= 2


def test_generalization_auditor_well_fitted():
    auditor = GeneralizationAuditor()
    diag = auditor.evaluate_fit(
        model_name="tuned_ridge",
        train_score=0.88,
        validation_score=0.85,
        metric_name="accuracy",
        higher_is_better=True,
    )
    assert diag.status == "well_fitted"
    assert diag.risk_level == "low"
    assert diag.generalization_gap <= 0.08


def test_model_ranker_penalizes_overfitting():
    auditor = GeneralizationAuditor()
    # Candidate A: CV = 0.81, Train = 0.83 (well fitted, gap = 0.02)
    diag_a = auditor.evaluate_fit("model_a", train_score=0.83, validation_score=0.81, higher_is_better=True)
    # Candidate B: CV = 0.82 (higher nominal CV), Train = 0.99 (massive overfit, gap = 0.17)
    diag_b = auditor.evaluate_fit("model_b", train_score=0.99, validation_score=0.82, higher_is_better=True)

    r_a = CVResult("model_a", [0.81, 0.81], 0.81, 0.01, "accuracy", 2)
    r_b = CVResult("model_b", [0.82, 0.82], 0.82, 0.01, "accuracy", 2)

    ranker = ModelRanker(higher_is_better=True, penalize_overfit=True, overfit_penalty_weight=0.30)
    ranked = ranker.rank([r_a, r_b], diagnostics={"model_a": diag_a, "model_b": diag_b})

    # Model A should rank #1 because Model B received an overfit penalty!
    assert ranked[0].model_name == "model_a"
    assert ranked[0].fit_status == "well_fitted"
    assert ranked[1].model_name == "model_b"
    assert ranked[1].fit_status == "overfitting"


def test_expanded_metrics_classification():
    calc = MetricsCalculator(task_type="binary_classification")
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])

    res = calc.compute("clf_model", y_true, y_pred, y_proba=y_proba)
    m = res.metrics
    assert "accuracy" in m
    assert "balanced_accuracy" in m
    assert "precision_macro" in m
    assert "recall_macro" in m
    assert "f1_macro" in m
    assert "f1_weighted" in m
    assert "roc_auc" in m
    assert "log_loss" in m
    assert "mcc" in m


def test_expanded_metrics_regression():
    calc = MetricsCalculator(task_type="regression")
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.5, 30.5, 38.0])

    res = calc.compute("reg_model", y_true, y_pred, n_features=2)
    m = res.metrics
    assert "r2" in m
    assert "adjusted_r2" in m
    assert "explained_variance" in m
    assert "max_error" in m
    assert "median_absolute_error" in m
    assert "mae" in m
    assert "mse" in m
    assert "rmse" in m
    assert "mape" in m
