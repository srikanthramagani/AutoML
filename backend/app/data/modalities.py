"""Heuristic, overrideable modality detection for mixed CSV data."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import re
from typing import List, Optional
import pandas as pd


@dataclass
class ModalityReport:
    numerical: List[str]
    categorical: List[str]
    boolean: List[str]
    datetime: List[str]
    text: List[str]
    identifiers: List[str]
    high_cardinality: List[str]
    rows: int

    def to_dict(self):
        return asdict(self)


def detect_modalities(frame: pd.DataFrame, target: Optional[str] = None, overrides: Optional[dict] = None) -> ModalityReport:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("A non-empty dataframe is required.")
    overrides = overrides or {}
    result = {key: [] for key in ("numerical", "categorical", "boolean", "datetime", "text", "identifiers", "high_cardinality")}
    for col in frame.columns:
        if col == target:
            continue
        s = frame[col].dropna()
        n = len(s)
        unique = int(s.nunique())
        dtype = frame[col].dtype
        name = str(col)
        if name in overrides:
            kind = overrides[name]
            if kind in result:
                result[kind].append(name)
            elif kind not in ("target", "exclude"):
                raise ValueError(f"Unsupported modality override for {name}: {kind}")
            continue
        if n and unique / n > .98 and unique > 10 and re.search(r"(^id$|_id$|uuid|identifier|^index$)", name, re.I):
            result["identifiers"].append(name); continue
        if pd.api.types.is_bool_dtype(dtype) or (unique <= 2 and n and set(map(str.lower, map(str, s.unique()))) <= {"true", "false", "yes", "no", "0", "1"}):
            result["boolean"].append(name); continue
        if pd.api.types.is_datetime64_any_dtype(dtype):
            result["datetime"].append(name); continue
        if pd.api.types.is_numeric_dtype(dtype):
            result["numerical"].append(name)
            if unique > max(50, int(.5 * n)): result["high_cardinality"].append(name)
            continue
        date_name = re.search(r"date|time|timestamp|created|updated", name, re.I)
        parsed = pd.to_datetime(s.head(100), errors="coerce") if n and date_name else pd.Series([], dtype="datetime64[ns]")
        if n and len(parsed) and parsed.notna().mean() > .9:
            result["datetime"].append(name); continue
        avg_len = s.astype(str).str.len().mean() if n else 0
        avg_words = s.astype(str).str.split().str.len().mean() if n else 0
        unique_ratio = unique / max(n, 1)
        text_name = bool(re.search(r"text|review|comment|description|narrative|message|abstract|content|title|body",name,re.I))
        # Multiword labels (e.g. "all or nothing") are categorical, not prose.
        # Require both language-like length and meaningful value diversity unless
        # values are long free-form documents.
        natural = ((text_name and avg_len >= 20)
                   or (avg_words >= 3 and avg_len >= 25 and unique >= 20 and unique_ratio >= .2)
                   or (avg_len >= 55 and unique >= 10))
        if natural:
            result["text"].append(name)
        else:
            result["categorical"].append(name)
            if unique > 50: result["high_cardinality"].append(name)
    return ModalityReport(**result, rows=len(frame))
