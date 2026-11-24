import joblib
import numpy as np
import pandas as pd

from backend.app.data.modalities import detect_modalities
from backend.app.automl.multimodal import train_multimodal


def _fixture():
    rng = np.random.default_rng(8)
    labels = np.array(["positive", "negative"] * 24)
    return pd.DataFrame({
        "amount": rng.normal(size=len(labels)),
        "review": ["excellent quick delivery" if y == "positive" else "damaged late shipment" for y in labels],
        "customer_id": [f"customer-{i}" for i in range(len(labels))],
        "target": labels,
    })


def test_modality_detection_separates_text_and_identifier():
    report = detect_modalities(_fixture(), target="target")
    assert report.text == ["review"]
    assert report.numerical == ["amount"]
    assert report.identifiers == ["customer_id"]


def test_natural_language_detector_does_not_promote_multword_categories():
    frame=pd.DataFrame({'funding_method':['all or nothing','flexible']*20,'description':['A detailed campaign description with several words and varied meaning.']*40,'target':['a','b']*20})
    report=detect_modalities(frame,target='target')
    assert 'funding_method' in report.categorical
    assert 'description' in report.text


def test_early_fusion_export_predicts_unseen_records(tmp_path):
    frame = _fixture()
    report = detect_modalities(frame, target="target")
    path = tmp_path / "pipeline.joblib"
    result = train_multimodal(frame, "target", report, trials=1, folds=2, export_path=path)
    pipeline = joblib.load(path)
    predictions = pipeline.predict(frame.drop(columns="target").head(4))
    assert len(predictions) == 4
    assert result["n_train"] + result["n_test"] == len(frame)
    assert 0 <= result["metrics"]["f1_macro"] <= 1
    assert result["cv_score"] >= 0
