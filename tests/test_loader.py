"""
Tests for DataLoader
"""

import pandas as pd
import pytest
from backend.app.data.loader import DataLoader, DataLoadError


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0],
        "feature_b": ["x", "y", "z"],
        "target": [0, 1, 0],
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_success(sample_csv):
    loader = DataLoader(target_column="target")
    df, result = loader.load(sample_csv)
    assert len(df) == 3
    assert result.n_rows == 3
    assert result.n_cols == 3
    assert result.target_column == "target"


def test_load_file_not_found():
    loader = DataLoader()
    with pytest.raises(DataLoadError):
        loader.load("non_existent_file_12345.csv")


def test_load_infers_last_column(sample_csv):
    loader = DataLoader()
    df, result = loader.load(sample_csv)
    assert result.target_column == "target"


def test_split_features_target(sample_csv):
    loader = DataLoader()
    df, result = loader.load(sample_csv)
    X, y = loader.split_features_target(df, "target")
    assert "target" not in X.columns
    assert len(X) == 3
    assert len(y) == 3


@pytest.mark.parametrize(
    ("encoding", "expected_encoding", "delimiter"),
    [
        ("utf-8", "utf-8", ","),
        ("utf-8-sig", "utf-8-sig", ","),
        ("utf-16", "utf-16", ";"),
        ("cp1252", "cp1252", ","),
    ],
)
def test_loads_common_csv_encodings(tmp_path, encoding, expected_encoding, delimiter):
    path = tmp_path / f"sample_{encoding}.csv"
    content = f"name{delimiter}city{delimiter}target\nRenée{delimiter}München{delimiter}1\n"
    path.write_bytes(content.encode(encoding))

    dataframe, result = DataLoader(target_column="target").load(path)

    assert result.encoding == expected_encoding
    assert result.delimiter == delimiter
    assert dataframe.loc[0, "name"] == "Renée"
    assert dataframe.loc[0, "city"] == "München"


def test_load_reports_clear_error_for_unsupported_bytes(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"\x00\x00\x00\x00")

    with pytest.raises(DataLoadError, match="Could not parse CSV.*supported encodings"):
        DataLoader().load(path)
