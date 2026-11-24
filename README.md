# Adaptive AutoML Platform

An evolving tabular AutoML application. The active engine and FastAPI application live under `backend/app/`; `frontend/` contains the React/Vite client. The root `app/` directory is a legacy duplicate.

## Current capabilities

The application supports CSV profiling, classification and regression with fold-local sklearn preprocessing, per-model Optuna TPE optimization with SQLite persistence, holdout diagnostics, clustering and anomaly analysis, and joblib pipeline export. Fixed-origin forecasts can be trained at `POST /api/forecast/run`, downloaded at `/api/forecasts/{forecast_id}/download`, and predicted from saved history or appended observed history at `/api/forecasts/{forecast_id}/predict`. Explainability includes holdout permutation importance, local ablation, SHAP, and LIME. Controlled optimizer benchmarking and reproducible multi-dataset experiment artifacts are included. This remains a tabular project and does not implement image-text multimodal fusion.

See [the implementation audit](docs/implementation-audit.md), [implementation report](docs/implementation-report.md), and [architecture/methodology](docs/architecture.md).
For a current feature-by-feature description and setup/API usage instructions, see [implemented features and how to use the platform](docs/implemented-features-and-usage.md).

## Install and run

```powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API. Run the browser client in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Optuna studies default to `experiments/optuna_studies.db`. You can set `Settings.optuna_storage_url` to another Optuna-compatible storage URL. Uploaded files and model artifacts use the paths configured in `backend/app/config/settings.py`.

Install optional SHAP/LIME support with `python -m pip install -r backend\requirements-explainability.txt`. Run the reproducible experiments with `.venv\Scripts\python.exe experiments\run_multi_dataset_experiments.py`; the runner writes CSV and PNG artifacts under `experiments/results/` using sklearn datasets and documented synthetic data.

## Tests

```powershell
python -m pytest -q
```

Benchmark data sources, seeds, metrics, and limitations are recorded in `experiments/results/multi_dataset_results.csv` and the technical report.
