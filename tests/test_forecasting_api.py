import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.api.state import DATASETS
from backend.app.main import app


def test_forecast_endpoint_exports_and_predicts_future(tmp_path, monkeypatch):
    from backend.app.config.settings import backend_settings
    monkeypatch.setattr(backend_settings, "model_directory", str(tmp_path / "models"))
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=90, freq="D"),
        "value": 4 + np.sin(np.arange(90) / 4) + np.arange(90) * 0.02,
    })
    path = tmp_path / "daily.csv"
    frame.to_csv(path, index=False)
    DATASETS["forecast-test"] = {"path": str(path)}
    try:
        response = TestClient(app).post("/api/forecast/run", json={
            "dataset_id": "forecast-test",
            "target_column": "value",
            "datetime_column": "date",
            "horizon": 3,
            "n_trials": 2,
            "cv_folds": 2,
        })
    finally:
        DATASETS.pop("forecast-test", None)
    assert response.status_code == 200
    result = response.json()
    assert len(result["predictions"]) == 3
    assert "model_rmse" in result["metrics"]
    assert "naive_rmse" in result["metrics"]
    model_id = result["forecast_id"]
    client = TestClient(app)
    download = client.get(f"/api/forecasts/{model_id}/download")
    assert download.status_code == 200
    prediction = client.post(f"/api/forecasts/{model_id}/predict", data={"horizon": "2"})
    assert prediction.status_code == 200
    assert len(prediction.json()["predictions"]) == 2
    recent = frame.tail(3).copy()
    recent["date"] = pd.date_range("2024-03-31", periods=3, freq="D")
    recent["value"] += 3
    response = client.post(
        f"/api/forecasts/{model_id}/predict",
        files={"history_file": ("recent.csv", recent.to_csv(index=False), "text/csv")},
        data={"horizon": "2"},
    )
    assert response.status_code == 200
    assert response.json()["predictions"][0]["index"].startswith("2024-04-03")
