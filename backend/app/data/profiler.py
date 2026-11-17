"""
DataProfiler
============
Responsibility: Compute a comprehensive statistical profile of a raw DataFrame.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """
    Statistical profile for a single column.
    """
    name: str
    dtype: str
    inferred_type: str  # 'numeric', 'categorical', 'boolean', 'datetime', 'text', 'id'
    n_missing: int
    pct_missing: float
    n_unique: int
    pct_unique: float
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    top_values: List[Any] = field(default_factory=list)
    identifier_candidate: bool = False
    identifier_excluded_by_default: bool = False
    identifier_reason: Optional[str] = None


@dataclass
class DataProfile:
    """
    Full dataset profile produced by DataProfiler.
    """
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    n_boolean: int
    n_datetime: int
    total_missing_pct: float
    duplicate_row_count: int
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    target_column: Optional[str] = None
    identifier_columns: List[str] = field(default_factory=list)
    identifier_reasons: Dict[str, str] = field(default_factory=dict)


class DataProfiler:
    """
    Analyze a DataFrame and return a DataProfile.
    """

    def __init__(
        self,
        cardinality_threshold: int = 10,
        text_length_threshold: int = 50,
        id_pct_unique_threshold: float = 0.95,
    ) -> None:
        self.cardinality_threshold = cardinality_threshold
        self.text_length_threshold = text_length_threshold
        self.id_pct_unique_threshold = id_pct_unique_threshold

    def profile(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DataProfile:
        """
        Generate a comprehensive DataProfile for the given DataFrame.
        """
        column_profiles: Dict[str, ColumnProfile] = {}
        n_numeric = 0
        n_categorical = 0
        n_boolean = 0
        n_datetime = 0
        identifier_columns = []
        identifier_reasons = {}

        for col in df.columns:
            col_prof = self._profile_column(df, col, is_target=col == target_column)
            column_profiles[col] = col_prof
            if col_prof.identifier_excluded_by_default:
                identifier_columns.append(col)
                identifier_reasons[col] = col_prof.identifier_reason or "Likely identifier/index column."
            if col_prof.inferred_type == "numeric":
                n_numeric += 1
            elif col_prof.inferred_type in ("categorical", "text", "id"):
                n_categorical += 1
            elif col_prof.inferred_type == "boolean":
                n_boolean += 1
            elif col_prof.inferred_type == "datetime":
                n_datetime += 1

        total_cells = df.size
        total_missing = int(df.isnull().sum().sum())
        total_missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

        return DataProfile(
            n_rows=len(df),
            n_cols=len(df.columns),
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_boolean=n_boolean,
            n_datetime=n_datetime,
            total_missing_pct=round(float(total_missing_pct), 2),
            duplicate_row_count=int(df.duplicated().sum()),
            columns=column_profiles,
            target_column=target_column,
            identifier_columns=identifier_columns,
            identifier_reasons=identifier_reasons,
        )

    def _profile_column(self, df: pd.DataFrame, col: str, is_target: bool = False) -> ColumnProfile:
        series = df[col]
        n_missing = int(series.isnull().sum())
        n_unique = int(series.nunique(dropna=True))
        n_total = len(series)
        pct_missing = round(n_missing / n_total * 100, 2) if n_total else 0.0
        pct_unique = round(n_unique / n_total * 100, 2) if n_total else 0.0
        dtype_str = str(series.dtype)

        inferred_type = self._infer_type(series, n_unique, pct_unique)
        identifier_candidate, identifier_excluded, identifier_reason = self._detect_identifier(
            col, series, n_unique, pct_unique, is_target
        )

        mean_val = std_val = min_val = q25_val = med_val = q75_val = max_val = skew_val = None
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                mean_val = float(clean.mean())
                std_val = float(clean.std()) if len(clean) > 1 else 0.0
                min_val = float(clean.min())
                q25_val = float(clean.quantile(0.25))
                med_val = float(clean.median())
                q75_val = float(clean.quantile(0.75))
                max_val = float(clean.max())
                skew_val = float(clean.skew()) if len(clean) > 2 else 0.0

        top_val_series = series.value_counts(dropna=True).head(5)
        top_values = top_val_series.index.tolist()

        return ColumnProfile(
            name=col,
            dtype=dtype_str,
            inferred_type=inferred_type,
            n_missing=n_missing,
            pct_missing=pct_missing,
            n_unique=n_unique,
            pct_unique=pct_unique,
            mean=mean_val,
            std=std_val,
            min=min_val,
            q25=q25_val,
            median=med_val,
            q75=q75_val,
            max=max_val,
            skewness=skew_val,
            top_values=top_values,
            identifier_candidate=identifier_candidate,
            identifier_excluded_by_default=identifier_excluded,
            identifier_reason=identifier_reason,
        )

    def _detect_identifier(
        self,
        name: str,
        series: pd.Series,
        n_unique: int,
        pct_unique: float,
        is_target: bool,
    ) -> tuple[bool, bool, Optional[str]]:
        if is_target:
            return False, False, None

        normalized_name = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
        name_match = bool(re.search(
            r"(^id$|(^|_)(id|index|idx|row_number|row_num|record_id|record_number|customer_id|sample_id)($|_))",
            normalized_name,
        ))
        clean = series.dropna()
        high_uniqueness = bool(len(clean) >= 5 and n_unique / len(clean) >= self.id_pct_unique_threshold)
        sequential = False
        monotonic = False
        if pd.api.types.is_integer_dtype(series) and len(clean) >= 5:
            values = clean.to_numpy()
            sequential = bool(np.all(np.diff(values) == 1) or np.all(np.diff(values) == -1))
            monotonic = bool(clean.is_monotonic_increasing or clean.is_monotonic_decreasing)

        reasons = []
        if name_match:
            reasons.append(f"name matches identifier pattern '{name}'")
        if high_uniqueness:
            reasons.append(f"{pct_unique:.1f}% unique values")
        if sequential:
            reasons.append("sequential integer values")
        elif monotonic:
            reasons.append("monotonic values")

        candidate = bool(reasons)
        confirmed = bool(name_match or (high_uniqueness and not pd.api.types.is_numeric_dtype(series)))
        return candidate, confirmed, "; ".join(reasons) if reasons else None

    def _infer_type(self, series: pd.Series, n_unique: int, pct_unique: float) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            clean_vals = series.dropna().unique()
            if n_unique <= 2 and set(clean_vals).issubset({0, 1, 0.0, 1.0}):
                return "boolean"
            # Floating point with continuous spread -> numeric
            if pd.api.types.is_float_dtype(series):
                is_all_int = np.all(np.mod(series.dropna().values, 1) == 0)
                if not is_all_int:
                    return "numeric"
            # Integer column with very few unique values in large dataset -> categorical
            if n_unique <= self.cardinality_threshold and (len(series) > 50 and pct_unique < 5.0):
                return "categorical"
            return "numeric"

        # String / object column
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return "categorical"

        # Check for boolean string values
        lower_vals = set(clean.str.lower().unique())
        if lower_vals.issubset({"true", "false", "yes", "no", "t", "f", "0", "1"}):
            return "boolean"

        avg_len = clean.str.len().mean()
        if avg_len > self.text_length_threshold:
            return "text"

        if pct_unique >= self.id_pct_unique_threshold * 100 and len(clean) > 50:
            return "id"

        return "categorical"

    def summary(self, profile: DataProfile) -> str:
        lines = [
            "=" * 50,
            " Dataset Profile Summary",
            "=" * 50,
            f"  Total Rows         : {profile.n_rows:,}",
            f"  Total Columns      : {profile.n_cols}",
            f"  Numeric Columns    : {profile.n_numeric}",
            f"  Categorical Columns: {profile.n_categorical}",
            f"  Boolean Columns    : {profile.n_boolean}",
            f"  Missing Cells      : {profile.total_missing_pct:.2f}%",
            f"  Duplicate Rows     : {profile.duplicate_row_count:,}",
            "=" * 50,
        ]
        return "\n".join(lines)
