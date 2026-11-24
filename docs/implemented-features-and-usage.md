# Implemented features and how to use the platform

This guide describes the active code in `backend/app/` and `frontend/`. The top-level `app/` package is a legacy copy; it is not the backend entry point.

## What is implemented

### CSV intake and profiling

- Uploads CSV files, detects encoding and delimiter, and reports dataset dimensions, data types, missing values, duplicates, value summaries, likely identifiers, and detected modalities.
- CSV is the supported upload format. The UI states a 50 MB limit.
- The uploaded dataset is kept on disk, while its dataset ID and metadata are held in backend memory.

### Tabular AutoML

- Supervised classification and regression, plus clustering and anomaly detection through the general AutoML engine.
- Automatic task and primary metric selection, with task/metric overrides where supported.
- Candidate model comparison, Optuna TPE hyperparameter search, cross-validation, and a held-out evaluation set.
- Preprocessing for numeric and categorical features, including imputation, encoding, and scaling as appropriate. Supervised preprocessing is evaluated in sklearn pipelines during cross-validation.
- Results include the selected model and parameters, candidate leaderboard, holdout metrics, timing, and available diagnostics. Classification includes confusion-matrix visualizations; clustering/anomaly workflows include projection and score visualizations where available.
- Identifier-like columns are excluded by default; the training API accepts an explicit list of identifier columns to retain.
- The fitted preprocessing and estimator pipeline is exported as a joblib artifact with metadata.

### Tabular plus text classification

- A separate classification path combines detected text columns with numeric/categorical data. It supports TF-IDF text features and optional Sentence Transformer embeddings, and early or late feature fusion.
- Optuna searches model, text representation, fusion choice, and selected model settings using stratified cross-validation and macro F1. The chosen pipeline is evaluated on a holdout set.
- It returns classification metrics, feature importance, modality ablation diagnostics, and optional record-level explanations. A prediction CSV can be uploaded after training.
- Sentence Transformer support requires the optional `sentence-transformers` dependency and may download its model on first use. TF-IDF does not require it.
- This is tabular-plus-text classification. Image input and image-text fusion are not implemented.

### Time-series forecasting

- A dedicated fixed-origin, direct multi-step forecasting endpoint supports a user-selected target, optional datetime column/frequency, and horizons from 1 to 90 steps.
- It parses and sorts timestamps, handles duplicate times, builds lag/rolling/calendar features, uses expanding chronological validation with a horizon gap, and tunes Ridge per horizon with Optuna.
- Holdout results report MAE, RMSE, MAPE, and sMAPE alongside a last-value naive baseline. The exported forecaster can predict using its stored history or a CSV of appended observed history.
- The dedicated endpoint currently uses Ridge, not the general AutoML estimator pool. Supply a frequency when it cannot be inferred from the timestamps.

### Explainability, benchmarking, and experiments

- The general supervised workflow provides holdout permutation importance and local single-feature ablation. SHAP global/local values and LIME local weights are available when the optional packages work in the environment; results show an unavailable/fallback status otherwise.
- A controlled benchmark compares Optuna TPE, GridSearchCV, and RandomizedSearchCV on the same finite configuration budget, split, pipeline, metric, and CV setup.
- A reproducible multi-dataset script writes result tables and plots under `experiments/results/`. The checked-in runs are small workflow experiments, not evidence that one optimizer is generally better.
- Fairness and generalization diagnostics are included in the general training result where applicable. Fairness output is diagnostic and should not be read as a fairness guarantee.

## Run it

From the repository root (`adaptive-automl`):

```powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. To run the browser UI in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

For general AutoML explainability extras, install `backend\requirements-explainability.txt`. To enable the sentence-transformer text representation, install `backend\requirements-sentence-transformers.txt`.

## Use the browser workflow

1. Open the frontend and upload a CSV. Review the profile, detected target, and detected column types. Select the target column if it was not inferred correctly.
2. Choose a task: Auto, classification, regression, clustering, or forecasting. Set the trial and CV-fold counts. For forecasting, choose the target, optionally enter the datetime column, and set the horizon.
3. For ordinary tabular tasks, start training and wait for the job to complete. Review the selected model, leaderboard, metrics, diagnostics, and available explanations.
4. Download the model pipeline from the results/export view. For classification, use the unseen-CSV prediction control to create and download a predictions CSV.
5. For forecasting, review forecast-versus-actual and naive-baseline plots, download the forecaster, or provide an updated history CSV and request future predictions.
6. To run text classification, enable the tabular + text option and ensure the CSV has at least one text feature and a valid classification target. Select representation/fusion settings if shown; then review holdout metrics and optionally upload unseen rows for predictions.

The job and dataset IDs used by these workflows are currently held in memory. Restarting the backend clears their lookup state, so use/download artifacts during the same backend session. The default output folders are configured in `backend/app/config/settings.py` (`backend/uploads`, `backend/trained_models`, and `backend/experiments`).

## API workflow

All application endpoints are under `/api`; the full schemas are available at `/docs`.

| Purpose | Endpoint | Use |
| --- | --- | --- |
| Upload and profile CSV | `POST /api/upload` | Multipart form field `file`; returns dataset ID and profile. |
| Get profile | `GET /api/datasets/{dataset_id}/profile` | Retrieve the stored profile. |
| Start general AutoML | `POST /api/training/start` | JSON: `dataset_id`, optional `target_column`, `task`, `metric`, `cv_folds`, `optimization_trials`, `keep_identifier_columns`; returns job ID. |
| Poll general training | `GET /api/training/{job_id}` | Poll until status is `complete` or `failed`. |
| Model info and charts | `GET /api/models/{model_id}`, `GET /api/models/{model_id}/leaderboard-chart` | Retrieve result metadata and chart. |
| Download general pipeline | `GET /api/models/{model_id}/download` | Download exported joblib model. |
| Detect modalities | `GET /api/datasets/{dataset_id}/modalities` | Inspect recognized text, numeric, categorical, date, and identifier columns. |
| Train tabular + text classifier | `POST /api/multimodal/train` | JSON: `dataset_id`, `target_column`; optional `representation` (`tfidf`, `sentence_transformer`, `auto`), `fusion` (`early`, `late`, `auto`), `trials`, `cv_folds`, `overrides`, `seed`. |
| Predict from records / CSV | `POST /api/models/{model_id}/predict`, `POST /api/models/{model_id}/predict-upload` | Send records as JSON or a multipart CSV. CSV prediction returns a download URL. |
| Forecast and export | `POST /api/forecast/run` | JSON: `dataset_id`, `target_column`; optional `datetime_column`, `frequency`, `horizon`, `n_trials`, `cv_folds`, `random_seed`. Returns holdout metrics and forecast ID. |
| Download/predict forecast | `GET /api/forecasts/{forecast_id}/download`, `POST /api/forecasts/{forecast_id}/predict` | Predict endpoint accepts optional multipart `history_file` and `horizon`. |

For training API responses, the regular model ID is the returned job ID. The text-classification endpoint returns its completed result directly and uses its returned `job_id` for model operations.

## Implementation boundaries

- This is an interactive development platform, not a hosted production prediction service. Authentication, multi-user isolation, persistent job registry, and deployment/serving infrastructure are not provided.
- Job metadata and dataset/model lookup IDs are in memory. Some model/study files persist on disk, but that alone does not restore their API lookup after a restart.
- The root `app/` duplicate and some repository docs may describe older states. This guide follows `backend/app/`, the current FastAPI entry point, and the current frontend.
- Experiment artifacts and reported scores are bounded by their documented datasets/seeds/configurations; they do not establish general model or optimizer superiority.
