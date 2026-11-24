# Technical implementation and experiment report

Date: 2026-11-18

## Implemented in this continuation

- Added `ExplainabilityService`: holdout permutation importance and local feature ablation are always available; SHAP provides global importance, dependence points, and local contributions when installed; LIME provides local weights for numeric input when installed. Optional integrations return explicit unavailable/fallback states. UI charts use returned backend results.
- Added controlled finite-grid optimizer comparison across Optuna TPE, `GridSearchCV`, and `RandomizedSearchCV`. They share a train/test split, pipeline, CV splitter, metric, same finite set of parameter configurations, and the same maximum configuration budget. Optuna remains the production optimizer.
- Added `DirectTimeSeriesForecaster` and `POST /api/forecast/run`: datetime parsing/sorting, duplicate timestamp aggregation, optional frequency reindex, causal forward-fill for feature history (missing target values are excluded from fitting/scoring), lag/rolling/calendar features, direct fixed-origin horizons, per-horizon Optuna Ridge tuning with expanding `TimeSeriesSplit` and a horizon gap, a last-value naive baseline, and MAE/RMSE/MAPE/sMAPE. Forecast models are refit on all observed history after the holdout evaluation, joblib-exported with metadata, downloadable, and callable for future predictions using stored history or appended observed-history CSV input. Future targets are never used as features. Added an actual-vs-predicted-vs-naive frontend plot and unseen-forecast prediction flow.
- Added a runnable multi-dataset suite and generated CSVs/plots under `experiments/results/` for binary, multiclass, regression, clustering, anomaly detection with ground-truth labels, and time-series forecasting. Dataset sources and synthetic-data recipes are in the CSV.
- Added holdout confusion-matrix, feature-importance, local explanation, PCA cluster/anomaly projection, and anomaly score histogram visualizations. Existing continuous prediction plots use backend holdout predictions.
- Fixed mixed-type anomaly ground-truth normalization and two failing tests discovered when a supported interpreter became available.

## Reproducible experiments

Run from the repository root:

```powershell
.venv\Scripts\python.exe experiments\run_multi_dataset_experiments.py
```

The runner used seeds 17 and 42. The optimizer comparisons use six configurations per search method and three-fold CV. Results are preliminary: they use two seeds, one model family per supervised dataset, and a small finite search space. The CSV is the source of record; no optimizer is declared superior.

Dataset sizes and sources: Breast Cancer (569 × 30, `sklearn.datasets.load_breast_cancer`), Iris (150 × 4, `sklearn.datasets.load_iris`), Diabetes (442 × 10, `sklearn.datasets.load_diabetes`), blobs (360 × 4, `make_blobs`), anomaly data (530 × 3, 500 Gaussian inliers plus 30 uniform injected outliers), and time series (260 rows, generated AR(1) + trend + weekly seasonality + Gaussian noise). The generated data definitions and per-run settings are retained in the experiment runner and CSV.

| Task / dataset | Optimizer or model | Mean test score | SD across seeds |
|---|---|---:|---:|
| Binary / Breast Cancer | GridSearchCV | 0.9860 accuracy | 0.0000 |
| Binary / Breast Cancer | RandomizedSearchCV | 0.9860 accuracy | 0.0000 |
| Binary / Breast Cancer | Optuna TPE | 0.9615 accuracy | 0.0247 |
| Multiclass / Iris | GridSearchCV | 0.9211 accuracy | 0.0000 |
| Multiclass / Iris | RandomizedSearchCV | 0.9211 accuracy | 0.0000 |
| Multiclass / Iris | Optuna TPE | 0.9474 accuracy | 0.0000 |
| Regression / Diabetes | GridSearchCV | 0.4856 R² | 0.0010 |
| Regression / Diabetes | RandomizedSearchCV | 0.4856 R² | 0.0010 |
| Regression / Diabetes | Optuna TPE | 0.4877 R² | 0.0040 |
| Clustering / synthetic blobs | KMeans and agglomerative | 0.8097 silhouette mean | 0.0395 |
| Labeled anomaly / synthetic outliers | Isolation Forest F1 | 0.7393 | 0.1455 |
| Time series / synthetic daily series | Direct Ridge RMSE | 0.6217 | 0.3284 |

For the two time-series seeds, the model RMSE values were 0.8539 and 0.3895; naive last-value RMSE values were 1.5223 and 1.5821. This small synthetic experiment is a workflow check, not general evidence of forecasting performance. Clustering values combine train/holdout intrinsic silhouette diagnostics; the agglomerative holdout result is transductive.

Artifacts:

- `experiments/results/multi_dataset_results.csv`
- `experiments/results/forecast_predictions.csv`
- `experiments/results/clustering_projection.csv`
- `experiments/results/anomaly_scores.csv`
- `experiments/results/optimizer_comparison.png`
- `experiments/results/forecast_actual_vs_predicted.png`
- `experiments/results/clustering_projection.png`
- `experiments/results/anomaly_score_distribution.png`

## Validation

- Supported environment found at `.venv`: Python 3.12.10, Optuna 5.0.0, scikit-learn 1.9.1, pytest 9.1.1. Installed SHAP 0.54.1 and LIME 0.2.0.1 from the optional requirements file.
- SHAP and LIME paths both passed on real regression and binary classification pipelines; forecast joblib export, download, saved-history prediction, and appended-history prediction have API test coverage.
- Final full test suite: **61 passed** in 16.70 seconds. Six non-failing warnings remain: one Starlette/AnyIO deprecation, four calls to deprecated `datetime.utcnow()` during model exports, and one Optuna parameter-importance warning on a very small study.
- `npm run build` passed with the forecasting download/prediction controls and existing backend-result visualizations.
- Frontend: `npm run build` passed after the new result views.
- Multi-dataset script executed successfully and wrote 26 result rows and all listed charts/CSVs.
- Final full-suite status is recorded after the last validation run below.

## Remaining work and limits

- The existing generic `AutoMLEngine` time-series model pool remains separate from the new direct-horizon forecasting workflow. The forecast endpoint is Ridge-only, and irregular timestamp frequency should be supplied by the user when it cannot be inferred.
- Forecast artifacts are joblib-persisted, but their API identifiers currently use the existing in-memory `JOBS` map; after a backend restart the artifact file remains on disk while ID-based download/prediction lookup requires a persistent registry.
- Explainability includes holdout permutation and single-feature ablation, which are perturbation diagnostics. SHAP dependence values are returned and SHAP global/local results and LIME local weights are displayed when available; dedicated dependence scatter plots remain incomplete.
- Benchmarking currently covers one estimator family per supervised dataset and two seeds; there are no statistical significance tests or ablations. Do not infer that Optuna is generally better or worse from these runs.
- Multi-dataset experiments use built-in sklearn data and documented synthetic clustering/anomaly/time-series data. Broader external public benchmarks and additional model families remain future work.
- Fairness remains a diagnostic only and was not evaluated in this experiment suite. No fairness claim is made.
- The root `app/` remains a legacy duplicate; changes target the active `backend/app/` package.

## Research integrity

This is still a tabular AutoML project. The experiments are reproducible and real but small, and are not sufficient to support a general research superiority claim. No text-image multimodal fusion claim is made.
