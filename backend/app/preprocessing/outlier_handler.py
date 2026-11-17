"""
OutlierHandler
==============
Responsibility: Detect and clip outliers in numeric features using Tukey's IQR fences.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Detect and clip outliers in numeric features.
    """

    def __init__(
        self,
        strategy: str = "clip",
        iqr_multiplier: float = 1.5,
    ) -> None:
        self.strategy = strategy
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_: Dict[str, float] = {}
        self.upper_bounds_: Dict[str, float] = {}
        self.numeric_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierHandler":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.numeric_cols_ = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
        ]

        if self.strategy == "clip":
            for col in self.numeric_cols_:
                series = X[col].dropna()
                if len(series) > 0:
                    q25 = series.quantile(0.25)
                    q75 = series.quantile(0.75)
                    iqr = q75 - q25
                    lower = q25 - self.iqr_multiplier * iqr
                    upper = q75 + self.iqr_multiplier * iqr
                    self.lower_bounds_[col] = float(lower)
                    self.upper_bounds_[col] = float(upper)

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        if self.strategy == "clip":
            for col in self.numeric_cols_:
                if col in X_out.columns and col in self.lower_bounds_:
                    lower = self.lower_bounds_[col]
                    upper = self.upper_bounds_[col]
                    X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        return X_out
