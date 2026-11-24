import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.benchmarking import compare_optimizers


def test_budget_matched_optimizer_comparison_returns_independent_test_scores():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - X["b"] + rng.normal(scale=0.1, size=60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
    results = compare_optimizers(
        Pipeline([("scale", StandardScaler()), ("estimator", Ridge())]),
        X_train, y_train, X_test, y_test,
        {"estimator__alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True, "grid": [0.01, 0.1, 1.0, 10.0]}},
        "r2", KFold(3, shuffle=True, random_state=5), budget=4, random_state=6,
    )
    assert {result["optimizer"] for result in results} == {"GridSearchCV", "RandomizedSearchCV", "Optuna TPE"}
    assert all(0 <= result["configurations_evaluated"] <= 4 for result in results)
    assert all(np.isfinite(result["test_score"]) for result in results)
