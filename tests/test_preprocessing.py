"""
Tests for Preprocessing Pipeline
"""

import numpy as np
import pandas as pd
import pytest
from backend.app.preprocessing.outlier_handler import OutlierHandler
from backend.app.preprocessing.missing_values import MissingValueHandler
from backend.app.preprocessing.encoding import CategoricalEncoder
from backend.app.preprocessing.scaling import FeatureScaler
from backend.app.preprocessing.pipeline_builder import PreprocessingPipelineBuilder


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 1000.0],
        "cat_low": ["red", "blue", "red", None, "blue"],
        "cat_high": [f"c_{i}" for i in range(5)],
    })


def test_outlier_handler(sample_df):
    handler = OutlierHandler(strategy="clip")
    res = handler.fit_transform(sample_df)
    assert res["num"].max() < 1000.0


def test_missing_value_handler(sample_df):
    handler = MissingValueHandler()
    res = handler.fit_transform(sample_df)
    assert not res["num"].isnull().any()
    assert not res["cat_low"].isnull().any()


def test_missing_value_handler_fractional_median_in_nullable_integer_column():
    frame = pd.DataFrame({"num": pd.Series([100, pd.NA, 103], dtype="Int64")})

    result = MissingValueHandler().fit_transform(frame)

    assert result["num"].tolist() == [100.0, 101.5, 103.0]


def test_pipeline_builder(sample_df):
    builder = PreprocessingPipelineBuilder()
    pipeline = builder.build()
    res = pipeline.fit_transform(sample_df)
    assert isinstance(res, pd.DataFrame)
    assert not res.isnull().any().any()
