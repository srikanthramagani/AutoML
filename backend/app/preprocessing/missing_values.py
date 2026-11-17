"""
MissingValueHandler
===================
Responsibility: Impute missing values for numeric and categorical features.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values in a DataFrame while preserving schema and columns.
    """

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        add_indicator: bool = False,
        fill_value: Optional[Any] = None,
    ) -> None:
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.add_indicator = add_indicator
        self.fill_value = fill_value

        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.numeric_impute_values_: Dict[str, float] = {}
        self.categorical_impute_values_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "MissingValueHandler":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.numeric_cols_ = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
        ]
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        # Learn numeric imputation values
        self.numeric_impute_values_ = {}
        for col in self.numeric_cols_:
            series = X[col].dropna()
            if len(series) == 0:
                val = 0.0
            elif self.numeric_strategy == "mean":
                val = float(series.mean())
            elif self.numeric_strategy == "constant" and self.fill_value is not None:
                val = float(self.fill_value)
            else:  # default: median
                val = float(series.median())
            self.numeric_impute_values_[col] = val

        # Learn categorical imputation values
        self.categorical_impute_values_ = {}
        for col in self.categorical_cols_:
            series = X[col].dropna()
            if len(series) == 0:
                val = "missing"
            elif self.categorical_strategy == "constant" and self.fill_value is not None:
                val = str(self.fill_value)
            else:  # default: most_frequent
                mode_vals = series.mode()
                val = mode_vals.iloc[0] if len(mode_vals) > 0 else "missing"
            self.categorical_impute_values_[col] = val

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        # Add missingness indicators if configured
        if self.add_indicator:
            for col in self.numeric_cols_ + self.categorical_cols_:
                if col in X_out.columns:
                    X_out[f"{col}_isnan"] = X_out[col].isnull().astype(int)

        # Impute numeric
        for col in self.numeric_cols_:
            if col in X_out.columns:
                fill_val = self.numeric_impute_values_.get(col, 0.0)
                X_out[col] = X_out[col].fillna(fill_val)

        # Impute categorical
        for col in self.categorical_cols_:
            if col in X_out.columns:
                fill_val = self.categorical_impute_values_.get(col, "missing")
                X_out[col] = X_out[col].fillna(fill_val).astype(str)

        return X_out
