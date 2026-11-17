"""
Tests for Unsupervised Learning (Clustering and Anomaly Detection)
"""

import numpy as np
import pandas as pd
from backend.app.models.unsupervised import (
    get_clustering_models,
    get_anomaly_detection_models,
)
from backend.app.models.model_registry import ModelRegistry
from backend.app.evaluation.unsupervised_metrics import UnsupervisedMetricsCalculator
from backend.app.optimization.unsupervised_objective import UnsupervisedOptunaObjective
from backend.app.optimization.optimizer import HyperparameterOptimizer
from backend.app.automl.engine import AutoMLEngine
from backend.app.config.settings import Settings


def test_clustering_models_pool():
    models = get_clustering_models()
    assert len(models) >= 4
    for m in models:
        assert "name" in m
        assert hasattr(m["estimator"], "fit")


def test_anomaly_detection_models_pool():
    models = get_anomaly_detection_models()
    assert len(models) >= 3
    for m in models:
        assert "name" in m
        assert hasattr(m["estimator"], "fit")


def test_model_registry_unsupervised():
    reg = ModelRegistry()
    assert len(reg.list_models("clustering")) >= 4
    assert len(reg.list_models("anomaly_detection")) >= 3
    kmeans = reg.get("clustering", "kmeans")
    assert kmeans.__class__.__name__ == "KMeans"


def test_unsupervised_metrics():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    labels = np.random.choice([0, 1, 2], size=100)
    calc = UnsupervisedMetricsCalculator(task_type="clustering")
    res = calc.compute(model_name="kmeans", X=X, labels=labels)
    assert res.task_type == "clustering"
    assert "silhouette_score" in res.metrics
    assert "calinski_harabasz_score" in res.metrics


def test_unsupervised_e2e_clustering():
    settings = Settings(
        n_trials=3,
        cv_folds=2,
        fast_mode=True,
    )
    engine = AutoMLEngine(settings=settings)
    engine.fit(
        csv_path="datasets/raw/customer_segmentation_unsupervised.csv",
        target_column=None,
        task="clustering",
        export_filename="test_unsupervised_pipeline",
    )
    assert engine.best_model_name_ is not None
    assert engine.final_pipeline_ is not None
    assert engine.export_path_ is not None

    # Verify inference / label generation on raw dataframe
    df = pd.read_csv("datasets/raw/customer_segmentation_unsupervised.csv").head(5)
    if hasattr(engine.final_pipeline_, "predict"):
        try:
            preds = engine.final_pipeline_.predict(df)
            assert len(preds) == 5
        except Exception:
            # Transductive models (e.g. DBSCAN) without out-of-sample predict method
            preds = engine.final_pipeline_.fit_predict(df)
            assert len(preds) == 5
    else:
        preds = engine.final_pipeline_.fit_predict(df)
        assert len(preds) == 5
