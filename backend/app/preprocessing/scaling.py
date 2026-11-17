"""
FeatureScaler
=============
Responsibility: Scale numeric features to a standardized range or distribution.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scale numeric features while preserving categorical columns.
    """

    _SCALER_MAP = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    def __init__(
        self,
        strategy: str = "standard",
        numeric_cols: Optional[List[str]] = None,
    ) -> None:
        self.strategy = strategy
        self.numeric_cols = numeric_cols
        self.scaler_ = None
        self.fitted_numeric_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureScaler":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.strategy == "none":
            return self

        if self.numeric_cols is not None:
            self.fitted_numeric_cols_ = [c for c in self.numeric_cols if c in X.columns]
        else:
            self.fitted_numeric_cols_ = [
                c for c in X.columns
                if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
            ]

        scaler_cls = self._SCALER_MAP.get(self.strategy, StandardScaler)
        self.scaler_ = scaler_cls()

        if self.fitted_numeric_cols_:
            self.scaler_.fit(X[self.fitted_numeric_cols_])

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.strategy == "none" or self.scaler_ is None or not self.fitted_numeric_cols_:
            return X.copy()

        X_out = X.copy()
        existing = [c for c in self.fitted_numeric_cols_ if c in X_out.columns]
        if existing:
            scaled_vals = self.scaler_.transform(X_out[existing])
            X_out[existing] = scaled_vals

        return X_out
