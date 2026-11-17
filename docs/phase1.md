# Phase 1

## Intended workflow

CSV -> profiling -> task detection -> preprocessing candidates -> model candidates -> Optuna optimization -> cross-validation -> complete-pipeline ranking -> final holdout validation -> joblib export.

## Current scaffold status

The AutoML engine under `backend/app` implements the workflow for local CLI execution and the development FastAPI adapter. Uploads and model artifacts use filesystem storage, while job metadata currently lives in memory until the optional database layer is connected.

Not included: automated feature engineering, SHAP, LIME, neural networks, image processing, text processing, multimodal learning, ONNX, Docker, production deployment, drift detection, and automatic retraining.