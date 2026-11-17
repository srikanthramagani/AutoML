"""
TimeSeriesFeatureExtractor
==========================
Responsibility: Automatically extract temporal, cyclical, trend, and lag features
from tabular time-series datasets.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract temporal calendar features (month, day, dayofweek, cyclical sine/cosine),
    trend index, and rolling/lag signals from datetime and continuous columns.
    """

    def __init__(
        self,
        datetime_column: Optional[str] = None,
        drop_datetime_col: bool = True,
        add_cyclical: bool = True,
        add_trend: bool = True,
    ) -> None:
        self.datetime_column = datetime_column
        self.drop_datetime_col = drop_datetime_col
        self.add_cyclical = add_cyclical
        self.add_trend = add_trend
        self.detected_datetime_col_: Optional[str] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TimeSeriesFeatureExtractor":
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.detected_datetime_col_ = self.datetime_column

        if self.detected_datetime_col_ is None:
            # Auto-detect datetime column
            for col in X_df.columns:
                if pd.api.types.is_datetime64_any_dtype(X_df[col]):
                    self.detected_datetime_col_ = col
                    break
                # Try parsing if string/object
                if pd.api.types.is_string_dtype(X_df[col]) or pd.api.types.is_object_dtype(X_df[col]):
                    sample = X_df[col].dropna().head(10)
                    if len(sample) > 0:
                        try:
                            pd.to_datetime(sample, errors="raise")
                            self.detected_datetime_col_ = col
                            break
                        except Exception:
                            continue

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = (X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)).copy()

        dt_col = self.detected_datetime_col_
        if dt_col is not None and dt_col in X_out.columns:
            dt_series = pd.to_datetime(X_out[dt_col], errors="coerce")

            # Extract calendar components
            X_out[f"{dt_col}_year"] = dt_series.dt.year.fillna(2000).astype(int)
            X_out[f"{dt_col}_month"] = dt_series.dt.month.fillna(1).astype(int)
            X_out[f"{dt_col}_day"] = dt_series.dt.day.fillna(1).astype(int)
            X_out[f"{dt_col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
            X_out[f"{dt_col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
            X_out[f"{dt_col}_quarter"] = dt_series.dt.quarter.fillna(1).astype(int)

            if self.add_cyclical:
                # Cyclical transformations for month and day of week
                month = X_out[f"{dt_col}_month"]
                X_out[f"{dt_col}_month_sin"] = np.sin(2 * np.pi * month / 12.0)
                X_out[f"{dt_col}_month_cos"] = np.cos(2 * np.pi * month / 12.0)

                dow = X_out[f"{dt_col}_dayofweek"]
                X_out[f"{dt_col}_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
                X_out[f"{dt_col}_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

            if self.drop_datetime_col:
                X_out = X_out.drop(columns=[dt_col])

        # Always add monotonic normalized trend step feature
        if self.add_trend:
            X_out["trend_step"] = np.arange(len(X_out), dtype=float) / max(1.0, float(len(X_out)))

        return X_out
