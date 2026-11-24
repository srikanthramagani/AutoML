# Implementation audit (before changes)

## Scope and architecture

The repository contains two copies of the Python application: root `app/` and `backend/app/`. The FastAPI application and checked-in backend tests import `backend.app`; the root copy is not referenced by those entry points. `docs/architecture.md` currently calls the root copy the working engine, which is inconsistent with the code. The React/Vite frontend calls the FastAPI endpoints and the backend owns training.

## Implemented

- CSV loading, profiling, task detection, tabular preprocessing, model registry, Optuna TPE search, supervised cross-validation, holdout metrics, model ranking, joblib export, fairness/generalization diagnostics, FastAPI upload/training/status/download endpoints, and a React dashboard.
- Clustering and anomaly detector registries; clustering metrics and time-series calendar features / chronological holdout / `TimeSeriesSplit` are present.
- Pipeline based supervised preprocessing means imputation/encoding/scaling are generally fit inside supervised CV folds.

## Partial or missing

- Optuna creates an in-memory maximize-only study per model. There is no persistence/resume, pruning, study history/importance API, experiment metadata, or isolation between repeated named runs. Failed objectives often return sentinel scores instead of failing trials.
- Supervised metrics include core classification and regression values, but PR-AUC, confusion matrix, threshold analysis and residual diagnostics are absent. Error objectives use sklearn's negated scorer convention (maximization), and the leaderboard always maximizes.
- Clustering uses silhouette, while anomaly detection currently optimizes mean decision/score value or a preferred anomaly fraction. No true-label metrics or honest unlabeled diagnostics are returned. Unsupervised preprocessing is cached by fitting on all of X_train before objective evaluation, creating fold leakage.
- Time-series uses calendar/trend features, but no lag/rolling features or naive baseline. Timestamp handling and frequency/horizon controls are limited. The pipeline must remain chronological and avoid deriving target lags from future rows.
- Explainability, controlled optimizer baselines, multi-dataset experiment artifacts/reporting, and frontend panels for optimization history, importance, clustering, anomaly diagnostics and explanations are not implemented.
- Export saves a full preprocessing pipeline and JSON metadata, but metadata does not fully describe feature schema, versions, seed or reproducibility configuration.
- Existing tests include end-to-end references to `datasets/raw/...` files that are not present in this checkout (the available data is under `datasets/public/`). There is no verified experiment results set.

## Correctness risks to address first

1. Invalid/sentinel optimization objectives can count as successful studies; no completed-trial check guards access to `best_params`.
2. Unsupervised preprocessing is fit once before optimization CV, contaminating validation partitions.
3. Anomaly objective rewards estimator-specific score conventions and an arbitrary contamination ratio, so model scores are incomparable and scientifically invalid.
4. Ranking assumes larger is always better, including explicit error metrics.
5. In unsupervised holdout evaluation, the engine manually reconstructs preprocessing by hard-coded step names and can mis-handle scores (for example, LOF decision scores are inlier-oriented).
6. Backend and docs disagree on the active source tree; several requested features are research extensions, not currently claimed implementation.

## Change and validation record

This audit was written before implementation changes. See the repository change record and final response for the exact files changed, checks run, and remaining scope.
