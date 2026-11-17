"""
Tests for Optuna Search Space and Optimizer
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from backend.app.optimization.search_space import SearchSpace
from backend.app.optimization.objective import OptunaObjective
from backend.app.optimization.optimizer import HyperparameterOptimizer
from backend.app.preprocessing.pipeline_builder import PreprocessingPipelineBuilder


def test_search_space():
    space = SearchSpace()
    assert "random_forest" in space._spaces
    assert "logistic_regression" in space._spaces


def test_optimizer():
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    X = df[["f1"]]
    y = df["target"]

    preproc = PreprocessingPipelineBuilder().build()
    est = LogisticRegression()
    obj = OptunaObjective(
        preprocessing_pipeline=preproc,
        estimator=est,
        model_name="logistic_regression",
        X=X,
        y=y,
        scoring="accuracy",
        cv_folds=2,
        task_type="classification",
    )
    optimizer = HyperparameterOptimizer(n_trials=2, random_state=42)
    res = optimizer.optimize_model(obj, "logistic_regression")
    assert res.n_trials == 2
    assert res.best_score > 0
