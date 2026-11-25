# Adaptive AutoML Platform

An evolving tabular AutoML application. The active engine and FastAPI application live under `backend/app/`; `frontend/` contains the React/Vite client. The root `app/` directory is a legacy duplicate.

## Current capabilities

The application supports CSV profiling, classification and regression with fold-local sklearn preprocessing, per-model Optuna TPE optimization with SQLite persistence, holdout diagnostics, clustering and anomaly analysis, and joblib pipeline export. Fixed-origin forecasts can be trained at `POST /api/forecast/run`, downloaded at `/api/forecasts/{forecast_id}/download`, and predicted from saved history or appended observed history at `/api/forecasts/{forecast_id}/predict`. Explainability includes holdout permutation importance, local ablation, SHAP, and LIME. Controlled optimizer benchmarking and reproducible multi-dataset experiment artifacts are included. This remains a tabular project and does not implement image-text multimodal fusion.

## Evaluation reliability auditing

Every completed run through `POST /api/training/start` now receives an evidence-based evaluation audit, stored with that experiment's SQLite-backed job record. The auditor reports exact duplicate rows and split overlap, repeated feature vectors with conflicting labels, near-deterministic feature/target relationships, temporal/group split review warnings, small-test-set risk, unusually high scores, and train/validation gaps. These are screening checks, not proof of leakage; features are never removed automatically.

The reliability score is separate from predictive performance. Its configurable default weights are leakage (25), split integrity (20), validation (20), preprocessing (15), dataset adequacy (10), reproducibility (5), and metric validity (5). A passed component earns its full weight, a warning half, and a failed or unchecked component zero; the score is withheld until at least four weighted checks execute. The calculation and per-component evidence are included in every report.

Open the **Evaluation Reliability** view after a completed training run to inspect findings, the score formula, dataset integrity, and JSON/CSV exports. A corrected evaluation is user-initiated: it clones the selected pipeline, optionally excludes features selected by the user, retrains on a new split, and calculates metrics on an independent holdout. Corrected evaluation currently supports classification and regression, requires the original dataset and saved pipeline to remain available, and uses a stratified holdout for classification or a random holdout for regression. Group-aware and chronological corrected retraining, repeated-seed stability analysis, and forecasting-job audit integration remain future work; corresponding temporal/group checks only issue review recommendations.

Audit API:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/audit/{audit_id}` | Retrieve the evidence report associated with a completed training job ID. |
| `GET` | `/api/audit/{audit_id}/report?format=json` | Export the full audit JSON. |
| `GET` | `/api/audit/{audit_id}/report?format=csv` | Export the findings table as CSV. |
| `POST` | `/api/audit/{audit_id}/corrected-evaluation` | Retrain/evaluate on a new split. Optional JSON body: `{"excluded_features":["feature_name"]}`. |

Reports use one of `TRUSTED WITHIN CHECKED CONDITIONS`, `TRUSTWORTHY WITH WARNINGS`, `SUSPICIOUS`, `INVALID EVALUATION`, or `INCONCLUSIVE`; the wording intentionally avoids claiming absolute trustworthiness. The checks are heuristic and do not establish feature provenance, deployment-time availability, causal validity, or whether the split matches a particular production population.

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

Install optional SHAP/LIME support with `python -m pip install -r backend\requirements-explainability.txt`. Install recurrent forecasting support with `python -m pip install -r backend\requirements-forecasting.txt`. The interactive application uses uploaded or repository datasets under `datasets/`; synthetic data is restricted to explicitly labeled benchmark experiment scripts and is not used as a substitute for user CSV training.

## Tests

```powershell
python -m pytest -q
```

Benchmark data sources, seeds, metrics, and limitations are recorded in `experiments/results/multi_dataset_results.csv` and the technical report.
