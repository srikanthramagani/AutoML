"""
Tests for DataProfiler and TaskDetector
"""

import numpy as np
import pandas as pd
import pytest
from backend.app.data.profiler import DataProfiler
from backend.app.data.task_detector import TaskDetectionError, TaskDetector
from backend.app.config.settings import Settings


@pytest.fixture
def mixed_df():
    return pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat": ["a", "b", "a", None, "b"],
        "target_class": [0, 1, 0, 1, 0],
    })


def test_profiler(mixed_df):
    profiler = DataProfiler()
    profile = profiler.profile(mixed_df)
    assert profile.n_rows == 5
    assert profile.n_cols == 3
    assert profile.n_numeric >= 1
    assert profile.columns["num"].n_missing == 1
    summary = profiler.summary(profile)
    assert "Dataset Profile" in summary


def test_sequential_id_is_excluded_by_default():
    profile = DataProfiler().profile(pd.DataFrame({"Id": range(1, 11), "value": range(10)}))
    column = profile.columns["Id"]
    assert column.identifier_candidate is True
    assert column.identifier_excluded_by_default is True
    assert "sequential" in column.identifier_reason
    assert profile.identifier_columns == ["Id"]


def test_random_unique_named_id_is_excluded_by_default():
    profile = DataProfiler().profile(pd.DataFrame({"record_id": [91, 4, 77, 13, 52, 8], "value": [1, 2, 3, 4, 5, 6]}))
    column = profile.columns["record_id"]
    assert column.identifier_candidate is True
    assert column.identifier_excluded_by_default is True
    assert "unique" in column.identifier_reason


def test_legitimate_integer_feature_is_not_excluded():
    profile = DataProfiler().profile(pd.DataFrame({"age": [21, 35, 42, 29, 51, 63], "value": [0, 1, 0, 1, 0, 1]}))
    assert profile.columns["age"].identifier_excluded_by_default is False


def test_target_column_is_never_identifier_candidate():
    profile = DataProfiler().profile(pd.DataFrame({"Id": range(1, 7), "target": range(6)}), target_column="Id")
    assert profile.columns["Id"].identifier_candidate is False
    assert "Id" not in profile.identifier_columns


def test_categorical_id_is_excluded_by_default():
    profile = DataProfiler().profile(pd.DataFrame({"customer_id": ["a1", "b2", "c3", "d4", "e5", "f6"], "value": [1, 2, 3, 4, 5, 6]}))
    column = profile.columns["customer_id"]
    assert column.identifier_candidate is True
    assert column.identifier_excluded_by_default is True


def test_task_detector_classification(mixed_df):
    detector = TaskDetector()
    info = detector.detect(mixed_df["target_class"])
    assert info.task_type == "binary_classification"
    assert info.primary_metric == "roc_auc"


def test_task_detector_regression():
    y = pd.Series([10.5, 20.2, 35.1, 40.8, 55.4, 62.1, 70.0, 85.3, 90.1, 105.4], name="price")
    detector = TaskDetector()
    info = detector.detect(y)
    assert info.task_type == "regression"
    assert info.primary_metric == "r2"


def test_task_detector_rejects_single_class_target():
    with pytest.raises(TaskDetectionError, match="at least two distinct values"):
        TaskDetector().detect(pd.Series([1, 1, 1], name="target"))


def test_settings_reject_invalid_values():
    with pytest.raises(ValueError, match="cv_folds"):
        Settings(cv_folds=1)
