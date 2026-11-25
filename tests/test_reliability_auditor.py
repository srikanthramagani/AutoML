import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.evaluation.reliability_auditor import (
    EvaluationReliabilityAuditor,
    run_corrected_evaluation,
)


def _audit(frame, **overrides):
    arguments = {
        "dataframe": frame,
        "target_column": "target",
        "task_type": "binary_classification",
        "model_name": "test-model",
        "metrics": {"accuracy": 0.8, "f1_weighted": 0.8},
        "validation_strategy": "stratified holdout + stratified cross-validation",
        "cv_folds": 5,
        "random_seed": 42,
        "preprocessing_fold_local": True,
    }
    arguments.update(overrides)
    return EvaluationReliabilityAuditor().audit(**arguments)


def test_normal_dataset_has_no_critical_findings():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "feature_a": rng.normal(size=160),
        "feature_b": rng.normal(size=160),
        "target": np.tile([0, 1], 80),
    })

    report = _audit(frame)

    assert report["reliability"]["score"] is not None
    assert not any(item["severity"] == "Critical" for item in report["findings"])
    assert report["dataset"]["fingerprint"]


def test_perfect_score_does_not_by_itself_mean_leakage():
    values = np.arange(120)
    frame = pd.DataFrame({"separating_feature": values, "target": (values >= 60).astype(int)})

    report = _audit(frame, metrics={"accuracy": 1.0, "f1_weighted": 1.0})

    assert "high-classification-score" in {item["id"] for item in report["findings"]}
    assert "target-proxy-feature" not in {item["id"] for item in report["findings"]}
    assert report["reliability"]["status"] not in {"SUSPICIOUS", "INVALID EVALUATION"}


def test_direct_target_copy_is_reported_as_suspected_feature():
    frame = pd.DataFrame({
        "signal": np.arange(100),
        "target": np.tile([0, 1], 50),
    })
    frame["target_copy"] = frame["target"]

    report = _audit(frame, feature_columns=["signal", "target_copy"])

    finding = next(item for item in report["findings"] if item["id"] == "target-proxy-feature")
    assert "target_copy" in finding["evidence"]
    assert finding["verified"] is False


def test_duplicate_records_crossing_the_split_are_reported():
    base = pd.DataFrame({
        "feature": np.arange(30),
        "target": np.tile([0, 1], 15),
    })
    frame = pd.concat([base] * 4, ignore_index=True)

    report = _audit(frame, metrics={"accuracy": 0.8})

    assert report["dataset"]["duplicate_record_count"] > 0
    assert report["summary"]["duplicate_train_test_overlap"] > 0
    assert "duplicate-rows" in {item["id"] for item in report["findings"]}


def test_time_column_with_random_validation_recommends_chronological_check():
    frame = pd.DataFrame({
        "event_date": pd.date_range("2020-01-01", periods=100),
        "signal": np.arange(100),
        "target": np.tile([0, 1], 50),
    })

    report = _audit(frame, validation_strategy="random holdout + stratified CV")

    assert "temporal-validation-review" in {item["id"] for item in report["findings"]}


def test_corrected_evaluation_retrains_pipeline_on_new_split():
    values = np.arange(100, dtype=float)
    frame = pd.DataFrame({"signal": values, "target": (values > 49).astype(int)})
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])
    pipeline.fit(frame[["signal"]], frame["target"])

    result = run_corrected_evaluation(
        dataframe=frame,
        target_column="target",
        fitted_pipeline=pipeline,
        task_type="binary_classification",
        feature_columns=["signal"],
    )

    assert result["status"] == "complete"
    assert result["test_size"] == 20
    assert "accuracy" in result["metrics"]


def test_too_small_dataset_reports_corrected_evaluation_limitation():
    frame = pd.DataFrame({"signal": np.arange(8), "target": np.tile([0, 1], 4)})
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])

    result = run_corrected_evaluation(frame, "target", pipeline, "binary_classification")

    assert result["status"] == "not_possible"
    assert "10 rows" in result["reason"]
