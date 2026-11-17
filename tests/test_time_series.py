"""
Tests for Time Series Feature Engineering, Models, and Forecasting Pipeline
"""

import numpy as np
import pandas as pd
from backend.app.preprocessing.time_features import TimeSeriesFeatureExtractor
from backend.app.models.time_series import get_time_series_models
from backend.app.models.model_registry import ModelRegistry
from backend.app.evaluation.cross_validation import CrossValidator
from backend.app.automl.engine import AutoMLEngine
from backend.app.config.settings import Settings


def test_time_series_feature_extractor():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "feature_1": np.arange(10),
    })

    extractor = TimeSeriesFeatureExtractor(datetime_column="date")
    transformed = extractor.fit_transform(df)

    assert "date_year" in transformed.columns
    assert "date_month" in transformed.columns
    assert "date_day" in transformed.columns
    assert "date_dayofweek" in transformed.columns
    assert "date_month_sin" in transformed.columns
    assert "date_month_cos" in transformed.columns
    assert "date_dow_sin" in transformed.columns
    assert "date_dow_cos" in transformed.columns
    assert "date_is_weekend" in transformed.columns
    assert "date_quarter" in transformed.columns
    assert "trend_step" in transformed.columns
    assert "date" not in transformed.columns


def test_time_series_models_pool():
    models = get_time_series_models()
    assert len(models) >= 4
    for m in models:
        assert "name" in m
        assert hasattr(m["estimator"], "fit")
        assert hasattr(m["estimator"], "predict")


def test_model_registry_time_series():
    reg = ModelRegistry()
    ts_models = reg.list_models("time_series_forecasting")
    assert len(ts_models) >= 4
    assert "time_series_gradient_boosting" in ts_models


def test_time_series_cross_validator():
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge

    cv_calc = CrossValidator(n_folds=3, task_type="time_series_forecasting", scoring="rmse")
    pipe = Pipeline([("estimator", Ridge())])
    X = pd.DataFrame(np.arange(60).reshape(30, 2), columns=["a", "b"])
    y = pd.Series(np.arange(30, dtype=float))

    res = cv_calc.validate(pipe, X, y, model_name="ridge")
    assert res.is_valid
    assert len(res.scores) == 3

    # Also verify temporal forward split ordering
    ts_split = TimeSeriesSplit(n_splits=3)
    splits = list(ts_split.split(X, y))
    for train_idx, test_idx in splits:
        assert np.max(train_idx) < np.min(test_idx)


def test_time_series_e2e_forecasting():
    settings = Settings(
        n_trials=2,
        cv_folds=3,
        fast_mode=True,
    )
    engine = AutoMLEngine(settings=settings)
    engine.fit(
        csv_path="datasets/raw/daily_temperatures_timeseries.csv",
        target_column="temperature",
        task="time_series_forecasting",
        export_filename="test_time_series_pipeline",
    )

    assert engine.best_model_name_ is not None
    assert engine.final_pipeline_ is not None
    assert engine.export_path_ is not None
    assert engine.prediction_series_ is not None
    assert len(engine.prediction_series_) > 0

    sample_pt = engine.prediction_series_[0]
    assert "actual" in sample_pt
    assert "predicted" in sample_pt
    assert "residual" in sample_pt

    df = pd.read_csv("datasets/raw/daily_temperatures_timeseries.csv").head(5)
    preds = engine.final_pipeline_.predict(df)
    assert len(preds) == 5
