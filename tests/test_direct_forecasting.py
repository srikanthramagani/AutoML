import numpy as np
import pandas as pd

from backend.app.models.forecasting import DirectTimeSeriesForecaster
from backend.app.models.forecasting import _series_from_frame


def test_direct_forecast_uses_only_pre_origin_values_and_emits_metrics():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    values = 10 + np.sin(np.arange(100) / 5) + np.arange(100) * 0.03
    frame = pd.DataFrame({"date": dates, "value": values})
    forecaster = DirectTimeSeriesForecaster(horizon=5, lags=(1, 2, 3, 7), rolling_windows=(3, 7), n_trials=2, cv_folds=3, random_state=12)
    result = forecaster.fit_evaluate(frame, "value", "date")

    changed_future = frame.copy()
    changed_future.loc[80:, "value"] += 5000
    second = DirectTimeSeriesForecaster(horizon=5, lags=(1, 2, 3, 7), rolling_windows=(3, 7), n_trials=2, cv_folds=3, random_state=12).fit_evaluate(changed_future, "value", "date")

    assert len(result["predictions"]) == 5
    assert np.allclose([row["predicted"] for row in result["predictions"]], [row["predicted"] for row in second["predictions"]])
    assert "model_rmse" in result["metrics"] and "naive_mae" in result["metrics"]
    assert result["config"]["forecast_mode"] == "fixed-origin direct multi-step"


def test_missing_timestamp_is_not_scored_as_a_forward_filled_target():
    dates = pd.date_range("2023-01-01", periods=100, freq="D").delete(86)
    frame = pd.DataFrame({"date": dates, "value": np.arange(len(dates), dtype=float)})
    series = _series_from_frame(frame, "value", "date", "D")
    assert pd.isna(series.loc[pd.Timestamp("2023-03-28")])
    result = DirectTimeSeriesForecaster(horizon=10, lags=(1, 2, 3), rolling_windows=(3,), n_trials=2, cv_folds=2).fit_evaluate(frame, "value", "date", "D")
    missing_rows = [row for row in result["predictions"] if row["actual"] is None]
    assert len(missing_rows) == 1
    assert missing_rows[0]["residual"] is None
