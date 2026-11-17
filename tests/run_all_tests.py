import os
import sys
import unittest

test_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(test_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import pandas as pd
import numpy as np

from backend.app.data.loader import DataLoader, DataLoadError
from backend.app.data.profiler import DataProfiler
from backend.app.data.task_detector import TaskDetector
from backend.app.preprocessing.outlier_handler import OutlierHandler
from backend.app.preprocessing.missing_values import MissingValueHandler
from backend.app.preprocessing.encoding import CategoricalEncoder
from backend.app.preprocessing.scaling import FeatureScaler
from backend.app.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from backend.app.models.classification import get_classification_models
from backend.app.models.regression import get_regression_models
from backend.app.models.model_registry import ModelRegistry
from backend.app.evaluation.cross_validation import CrossValidator, CVResult
from backend.app.evaluation.metrics import MetricsCalculator
from backend.app.evaluation.model_ranker import ModelRanker
from backend.app.training.trainer import Trainer
from backend.app.export.model_exporter import ModelExporter


class TestAutoMLPhase1(unittest.TestCase):

    def setUp(self):
        self.class_csv = os.path.join(project_dir, "datasets", "raw", "sample_classification.csv")
        self.reg_csv = os.path.join(project_dir, "datasets", "raw", "sample_regression.csv")

    def test_01_loader_classification(self):
        loader = DataLoader(target_column="churn")
        df, result = loader.load(self.class_csv)
        self.assertEqual(len(df), 250)
        self.assertEqual(result.target_column, "churn")
        X, y = loader.split_features_target(df, "churn")
        self.assertNotIn("churn", X.columns)
        self.assertEqual(len(y), 250)

    def test_02_loader_regression(self):
        loader = DataLoader(target_column="price")
        df, result = loader.load(self.reg_csv)
        self.assertEqual(len(df), 250)
        self.assertEqual(result.target_column, "price")
        X, y = loader.split_features_target(df, "price")
        self.assertNotIn("price", X.columns)
        self.assertEqual(len(y), 250)

    def test_03_profiler(self):
        loader = DataLoader(target_column="churn")
        df, _ = loader.load(self.class_csv)
        profiler = DataProfiler()
        profile = profiler.profile(df)
        self.assertEqual(profile.n_rows, 250)
        self.assertGreaterEqual(profile.n_numeric, 2)
        summary = profiler.summary(profile)
        self.assertIn("Dataset Profile", summary)

    def test_04_task_detector(self):
        detector = TaskDetector()
        y_class = pd.Series([0, 1, 0, 1, 1, 0, 1], name="target")
        info_c = detector.detect(y_class)
        self.assertEqual(info_c.task_type, "binary_classification")
        self.assertEqual(info_c.primary_metric, "roc_auc")

        y_reg = pd.Series([10.5, 20.3, 30.1, 40.7, 50.2, 60.9], name="target")
        info_r = detector.detect(y_reg)
        self.assertEqual(info_r.task_type, "regression")
        self.assertEqual(info_r.primary_metric, "r2")

    def test_05_preprocessing_pipeline(self):
        loader = DataLoader(target_column="churn")
        df, _ = loader.load(self.class_csv)
        X, y = loader.split_features_target(df, "churn")

        builder = PreprocessingPipelineBuilder()
        pipeline = builder.build()
        X_trans = pipeline.fit_transform(X)

        self.assertIsInstance(X_trans, pd.DataFrame)
        self.assertFalse(X_trans.isnull().any().any())
        self.assertTrue(all(pd.api.types.is_numeric_dtype(X_trans[c]) for c in X_trans.columns))

    def test_06_candidate_models(self):
        cls_models = get_classification_models()
        self.assertEqual(len(cls_models), 9)
        reg_models = get_regression_models()
        self.assertEqual(len(reg_models), 10)

        registry = ModelRegistry()
        for m in cls_models:
            registry.register("classification", m["name"], lambda m=m: m["estimator"])
            est = registry.get("classification", m["name"])
            self.assertTrue(hasattr(est, "fit"))

    def test_07_trainer_and_cv(self):
        loader = DataLoader(target_column="churn")
        df, _ = loader.load(self.class_csv)
        X, y = loader.split_features_target(df, "churn")

        builder = PreprocessingPipelineBuilder()
        pipeline = builder.build()
        cls_models = get_classification_models()

        trainer = Trainer(random_state=42)
        t_res = trainer.fit(pipeline, cls_models[0]["estimator"], X, y, model_name=cls_models[0]["name"])
        self.assertTrue(hasattr(t_res.pipeline, "predict"))

        validator = CrossValidator(n_folds=3, scoring="accuracy", task_type="classification")
        cv_res = validator.validate(t_res.pipeline, X, y, model_name=cls_models[0]["name"])
        self.assertGreater(cv_res.mean_score, 0.0)

    def test_08_model_ranker(self):
        ranker = ModelRanker(higher_is_better=True)
        r1 = CVResult("model_a", [0.75, 0.75], 0.75, 0.01, "roc_auc", 2)
        r2 = CVResult("model_b", [0.88, 0.88], 0.88, 0.02, "roc_auc", 2)
        ranked = ranker.rank([r1, r2])
        self.assertEqual(ranked[0].model_name, "model_b")
        self.assertEqual(ranked[1].model_name, "model_a")
        table = ranker.leaderboard_str(ranked)
        self.assertIn("Rank", table)

    def test_09_metrics_calculator(self):
        calc_c = MetricsCalculator(task_type="binary_classification")
        m_res_c = calc_c.compute("test_c", [0, 1, 0, 1], [0, 1, 1, 1])
        self.assertIn("accuracy", m_res_c.metrics)
        self.assertIn("f1_weighted", m_res_c.metrics)

        calc_r = MetricsCalculator(task_type="regression")
        m_res_r = calc_r.compute("test_r", [10.0, 20.0, 30.0], [11.0, 19.5, 30.5])
        self.assertIn("r2", m_res_r.metrics)
        self.assertIn("rmse", m_res_r.metrics)

    def test_10_model_exporter_and_inference(self):
        loader = DataLoader(target_column="churn")
        df, _ = loader.load(self.class_csv)
        X, y = loader.split_features_target(df, "churn")

        builder = PreprocessingPipelineBuilder()
        pipeline = builder.build()
        cls_models = get_classification_models()
        trainer = Trainer(random_state=42)
        t_res = trainer.fit(pipeline, cls_models[0]["estimator"], X, y, model_name="logistic_regression")

        exporter = ModelExporter(export_dir=os.path.join(project_dir, "models"))
        pkl_path = exporter.export(t_res.pipeline, "logistic_regression", filename="unit_test_pipeline")
        self.assertTrue(os.path.exists(pkl_path))

        loaded_pipeline = exporter.load(pkl_path)
        predictions = loaded_pipeline.predict(X)
        self.assertEqual(len(predictions), len(X))


if __name__ == "__main__":
    unittest.main(verbosity=2)
