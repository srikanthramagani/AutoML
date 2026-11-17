"""
Tests for Evaluation, Ranking, and Export
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from backend.app.evaluation.cross_validation import CrossValidator, CVResult
from backend.app.evaluation.metrics import MetricsCalculator
from backend.app.evaluation.fairness import FairnessAuditor
from backend.app.evaluation.model_ranker import ModelRanker
from backend.app.export.model_exporter import ModelExporter
from backend.app.preprocessing.scaling import FeatureScaler


def test_metrics_calculator():
    calc = MetricsCalculator(task_type="binary_classification")
    res = calc.compute("test_m", [0, 1, 0, 1], [0, 1, 1, 1])
    assert "accuracy" in res.metrics
    assert "f1_weighted" in res.metrics
    assert "balanced_accuracy" in res.metrics
    assert "recall_class_0" in res.metrics


def test_fairness_audit_reports_group_gaps():
    sensitive = pd.DataFrame({"region": ["a", "a", "b", "b"]})
    result = FairnessAuditor().audit(sensitive, [0, 1, 0, 1], [0, 1, 1, 1], ["region"])
    assert result.status == "complete"
    assert result.demographic_parity_difference == 0.5


def test_model_ranker():
    ranker = ModelRanker()
    r1 = CVResult("m1", [0.8, 0.8], 0.8, 0.0, "roc_auc", 2)
    r2 = CVResult("m2", [0.9, 0.9], 0.9, 0.0, "roc_auc", 2)
    ranked = ranker.rank([r1, r2])
    assert ranked[0].model_name == "m2"
    assert ranked[1].model_name == "m1"
    table = ranker.leaderboard_str(ranked)
    assert "Rank" in table


def test_model_ranker_ignores_invalid_results():
    invalid = CVResult("broken", [], float("nan"), float("nan"), "roc_auc", 2, False, "fit failed")
    valid = CVResult("working", [0.7, 0.8], 0.75, 0.05, "roc_auc", 2)
    ranked = ModelRanker().rank([invalid, valid])
    assert [model.model_name for model in ranked] == ["working"]


def test_model_exporter(tmp_path):
    exporter = ModelExporter(export_dir=str(tmp_path))
    pipe = Pipeline([("scaler", FeatureScaler()), ("est", LogisticRegression())])
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1])
    pipe.fit(X, y)

    saved_path = exporter.export(pipe, "lr", {"cv_score": 0.95})
    assert (tmp_path / "trained_pipeline.pkl").exists()
    assert (tmp_path / "trained_pipeline_metadata.json").exists()

    loaded_pipe = exporter.load(saved_path)
    preds = loaded_pipe.predict(X)
    assert len(preds) == 4
