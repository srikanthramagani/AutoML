"""
CategoricalEncoder
==================
Responsibility: Encode categorical columns into numeric representations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical columns with automatic cardinality branching.
    """

    def __init__(
        self,
        strategy: str = "auto",
        max_onehot_cardinality: int = 15,
        column_overrides: Optional[Dict[str, str]] = None,
        handle_unknown: str = "ignore",
    ) -> None:
        self.strategy = strategy
        self.max_onehot_cardinality = max_onehot_cardinality
        self.column_overrides = column_overrides
        self.handle_unknown = handle_unknown

        self.onehot_cols_: List[str] = []
        self.ordinal_cols_: List[str] = []
        self.passthrough_cols_: List[str] = []
        self.onehot_encoder_: Optional[OneHotEncoder] = None
        self.ordinal_encoder_: Optional[OrdinalEncoder] = None
        self.onehot_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalEncoder":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.onehot_cols_ = []
        self.ordinal_cols_ = []
        self.passthrough_cols_ = []

        for col in X.columns:
            is_cat = (
                not pd.api.types.is_numeric_dtype(X[col])
                or pd.api.types.is_bool_dtype(X[col])
                or isinstance(X[col].dtype, pd.CategoricalDtype)
            )

            if not is_cat:
                self.passthrough_cols_.append(col)
                continue

            overrides = self.column_overrides or {}
            strat = overrides.get(col, self.strategy)
            n_unique = X[col].nunique(dropna=True)

            if strat == "ordinal":
                self.ordinal_cols_.append(col)
            elif strat == "onehot":
                self.onehot_cols_.append(col)
            else:  # auto
                if n_unique <= self.max_onehot_cardinality:
                    self.onehot_cols_.append(col)
                else:
                    self.ordinal_cols_.append(col)

        # Fit OneHotEncoder
        if self.onehot_cols_:
            try:
                self.onehot_encoder_ = OneHotEncoder(
                    handle_unknown=self.handle_unknown,
                    sparse_output=False,
                )
            except TypeError:
                self.onehot_encoder_ = OneHotEncoder(
                    handle_unknown=self.handle_unknown,
                    sparse=False,
                )
            data_oh = X[self.onehot_cols_].astype(str)
            self.onehot_encoder_.fit(data_oh)

            # Extract feature names across all sklearn versions
            if hasattr(self.onehot_encoder_, "get_feature_names_out"):
                try:
                    self.onehot_feature_names_ = list(
                        self.onehot_encoder_.get_feature_names_out(self.onehot_cols_)
                    )
                except Exception:
                    self.onehot_feature_names_ = self._generate_feature_names()
            elif hasattr(self.onehot_encoder_, "get_feature_names"):
                try:
                    self.onehot_feature_names_ = list(
                        self.onehot_encoder_.get_feature_names(self.onehot_cols_)
                    )
                except Exception:
                    self.onehot_feature_names_ = self._generate_feature_names()
            else:
                self.onehot_feature_names_ = self._generate_feature_names()
        else:
            self.onehot_encoder_ = None
            self.onehot_feature_names_ = []

        # Fit OrdinalEncoder
        if self.ordinal_cols_:
            try:
                self.ordinal_encoder_ = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
            except TypeError:
                self.ordinal_encoder_ = OrdinalEncoder()
            data_ord = X[self.ordinal_cols_].astype(str)
            self.ordinal_encoder_.fit(data_ord)
        else:
            self.ordinal_encoder_ = None

        return self

    def _generate_feature_names(self) -> List[str]:
        names = []
        for col, cats in zip(self.onehot_cols_, self.onehot_encoder_.categories_):
            for cat in cats:
                names.append(f"{col}_{cat}")
        return names

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        parts = []

        # Passthrough numeric columns
        if self.passthrough_cols_:
            existing = [c for c in self.passthrough_cols_ if c in X.columns]
            if existing:
                parts.append(X[existing].reset_index(drop=True))

        # Transformed OneHot columns
        if self.onehot_cols_ and self.onehot_encoder_ is not None:
            existing_oh = [c for c in self.onehot_cols_ if c in X.columns]
            if len(existing_oh) == len(self.onehot_cols_):
                data_oh = X[self.onehot_cols_].astype(str)
                oh_arr = self.onehot_encoder_.transform(data_oh)
                if hasattr(oh_arr, "toarray"):
                    oh_arr = oh_arr.toarray()
                oh_df = pd.DataFrame(oh_arr, columns=self.onehot_feature_names_)
                parts.append(oh_df.reset_index(drop=True))

        # Transformed Ordinal columns
        if self.ordinal_cols_ and self.ordinal_encoder_ is not None:
            existing_ord = [c for c in self.ordinal_cols_ if c in X.columns]
            if len(existing_ord) == len(self.ordinal_cols_):
                data_ord = X[self.ordinal_cols_].astype(str)
                ord_arr = self.ordinal_encoder_.transform(data_ord)
                if hasattr(ord_arr, "toarray"):
                    ord_arr = ord_arr.toarray()
                ord_df = pd.DataFrame(ord_arr, columns=self.ordinal_cols_)
                parts.append(ord_df.reset_index(drop=True))

        if not parts:
            return pd.DataFrame(index=X.index)

        out_df = pd.concat(parts, axis=1)
        out_df.index = X.index
        return out_df
