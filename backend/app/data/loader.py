"""
DataLoader
==========
Responsibility: Load a CSV file from disk into a validated pandas DataFrame.
"""

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd


class DataLoadError(Exception):
    """Raised when a CSV cannot be loaded or fails basic validation."""


@dataclass
class LoadResult:
    """
    Summary of a successful CSV load.
    """
    file_path: str
    n_rows: int
    n_cols: int
    columns: List[str]
    target_column: Optional[str] = None
    encoding: str = "utf-8"
    delimiter: Optional[str] = None


class DataLoader:
    """
    Load and validate CSV datasets.
    """

    def __init__(
        self,
        target_column: Optional[str] = None,
        encoding: str = "utf-8",
        sep: Optional[str] = None,
    ) -> None:
        self.target_column = target_column
        self.encoding = encoding
        self.sep = sep

    def load(self, file_path: Union[str, Path]) -> Tuple[pd.DataFrame, LoadResult]:
        """
        Load a CSV file and return (DataFrame, LoadResult).
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise DataLoadError(f"Could not read CSV '{path}': {exc}") from exc

        encodings = self._encoding_candidates()
        errors = []
        df = None
        detected_encoding = None
        detected_delimiter = None
        for encoding in encodings:
            try:
                decoded_text = raw_bytes.decode(encoding)
                if "\x00" in decoded_text:
                    raise UnicodeError("decoded CSV contains NUL characters")
                candidate_df, delimiter = self._parse_csv(path, encoding, decoded_text)
                df = candidate_df
                detected_encoding = (
                    "utf-8-sig"
                    if encoding == "utf-8" and raw_bytes.startswith(b"\xef\xbb\xbf")
                    else encoding
                )
                detected_delimiter = delimiter
                break
            except (UnicodeError, LookupError, OSError, ValueError, pd.errors.ParserError) as exc:
                errors.append(f"{encoding}: {exc}")

        if df is None or detected_encoding is None:
            attempted = ", ".join(encodings)
            details = "; ".join(errors)
            raise DataLoadError(
                f"Could not parse CSV '{path}' using supported encodings "
                f"({attempted}). Details: {details}"
            )

        if df.empty or len(df) == 0:
            raise DataLoadError(f"Loaded DataFrame from '{path}' is empty (0 rows).")

        # Strip whitespace from column names for cleaner handling
        df.columns = [str(c).strip() for c in df.columns]

        target = self.target_column or df.columns[-1]
        if target not in df.columns:
            raise DataLoadError(
                f"Target column '{target}' not found in dataset. Available columns: {list(df.columns)}"
            )

        result = LoadResult(
            file_path=str(path),
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=list(df.columns),
            target_column=target,
            encoding=detected_encoding,
            delimiter=detected_delimiter,
        )
        return df, result

    def _encoding_candidates(self) -> List[str]:
        """Return strict decoding candidates, preserving custom encoding support."""
        candidates = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252"]
        if self.encoding not in candidates:
            candidates.insert(0, self.encoding)
        return candidates

    def _parse_csv(
        self, path: Path, encoding: str, decoded_text: str
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Parse decoded CSV text while retaining the previous pandas behavior."""
        delimiter = self.sep
        if delimiter is None:
            try:
                dialect = csv.Sniffer().sniff(decoded_text[:8192], delimiters=",;")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = None

        if delimiter is not None:
            return pd.read_csv(path, encoding=encoding, sep=delimiter), delimiter

        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python"), None
        except Exception:
            return pd.read_csv(path, encoding=encoding), None

    def split_features_target(
        self,
        df: pd.DataFrame,
        target_column: str,
        excluded_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split a DataFrame into feature matrix X and target vector y.
        Drops rows where target y is missing.
        """
        if target_column not in df.columns:
            raise DataLoadError(f"Target column '{target_column}' does not exist in DataFrame.")

        # Drop rows where target is missing
        valid_idx = df[target_column].dropna().index
        df_valid = df.loc[valid_idx]

        columns_to_drop = [target_column] + [
            column for column in (excluded_columns or [])
            if column in df_valid.columns and column != target_column
        ]
        X = df_valid.drop(columns=columns_to_drop).copy()
        y = df_valid[target_column].copy()
        return X, y
