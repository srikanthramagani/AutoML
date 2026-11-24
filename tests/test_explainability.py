import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.explainability import ExplainabilityService


def test_explainability_returns_holdout_permutation_and_local_ablation():
    X = pd.DataFrame({"signal": np.arange(30, dtype=float), "noise": np.tile([0.0, 1.0], 15)})
    y = 2.0 * X["signal"]
    pipeline = make_pipeline(StandardScaler(), LinearRegression()).fit(X.iloc[:20], y.iloc[:20])
    result = ExplainabilityService(max_rows=8).explain(
        pipeline, X.iloc[:20], X.iloc[20:], y.iloc[20:], "regression", "r2"
    )
    assert result["status"] == "complete"
    assert result["holdout_rows_used"] == 8
    assert result["global_feature_importance"]
    assert result["local_explanations"]
    if find_spec("shap"):
        assert result["shap"]["status"] == "complete"
        assert result["shap"]["global_importance"]
    else:
        assert result["shap"]["status"] == "unavailable"
    assert result["lime"]["status"] == ("complete" if find_spec("lime") else "unavailable")


def test_local_ablation_accepts_fractional_median_for_nullable_integer_feature():
    X = pd.DataFrame({"signal": pd.Series([100, 102, 103, 105, 106, 108], dtype="Int64")})
    y = X["signal"].astype(float) * 2
    pipeline = make_pipeline(StandardScaler(), LinearRegression()).fit(X.iloc[:4], y.iloc[:4])

    result = ExplainabilityService()._ablation_explanations(
        pipeline, X.iloc[:4], X.iloc[4:], "regression"
    )

    assert result
    assert result[0]["features"][0]["replacement"] == "102.5"


def test_shap_and_lime_explanation_paths_support_binary_classification():
    rng = np.random.default_rng(21)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = (X["a"] - 0.4 * X["b"] > 0).astype(int)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression()).fit(X.iloc[:55], y.iloc[:55])
    result = ExplainabilityService(max_rows=10).explain(
        pipeline, X.iloc[:55], X.iloc[55:], y.iloc[55:], "binary_classification", "accuracy"
    )
    if find_spec("shap"):
        assert result["shap"]["status"] == "complete"
        assert result["shap"]["local"]
    else:
        assert result["shap"]["status"] == "unavailable"
    if find_spec("lime"):
        assert result["lime"]["status"] == "complete"
        assert result["lime"]["feature_weights"]
    else:
        assert result["lime"]["status"] == "unavailable"
from importlib.util import find_spec

