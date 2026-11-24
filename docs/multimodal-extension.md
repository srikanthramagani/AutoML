# Multimodal extension status

## Implemented in this checkpoint

- Existing tabular APIs and `AutoMLEngine` remain available and unchanged.
- Upload now includes a modality report. `GET /api/datasets/{dataset_id}/modalities` re-runs detection.
- `POST /api/multimodal/train` performs a stratified untouched holdout split, Optuna search over model family, representation, fusion, and supported parameters using macro F1 CV on training rows, refits the selected complete pipeline, reports holdout metrics, and exports the estimator plus metadata.
- Late fusion weight is chosen by macro F1 over inner out-of-fold training predictions.
- `POST /api/models/{model_id}/predict` accepts JSON records; `/predict-upload` accepts unseen CSV and returns a downloadable prediction CSV.
- `POST /api/models/{model_id}/explain` reports raw-field permutation importance and per-record masking/linear-term explanations. These are predictive attributions, not causal effects.
- Supported candidate families are Logistic Regression, Random Forest, Extra Trees, and calibrated Linear SVM. Sparse TF-IDF stays sparse during early fusion.
- The UI exposes the multimodal run switch, automatic/manual representations and fusion, budget, fold controls, column reassignment, modality report, model download, and unseen CSV predictions.

## Run

From `adaptive-automl`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
cd frontend
npm run dev
```

The standard requirements keep Sentence Transformers optional. Install with `python -m pip install -r backend/requirements-sentence-transformers.txt`; the library selects CUDA when PyTorch reports an available GPU and otherwise uses CPU. The first embedding run downloads the configured model from Hugging Face; TF-IDF is the no-model-download option. This development machine had CPU-only PyTorch, so GPU execution was not verified.

## API sketch

```json
POST /api/multimodal/train
{
  "dataset_id": "uploaded-id",
  "target_column": "label",
  "representation": "tfidf",
  "fusion": "early",
  "trials": 6,
  "cv_folds": 3
}
```

The `overrides` field maps column names to `numerical`, `categorical`, `boolean`, `datetime`, `text`, `identifiers`, or `exclude`. Training output includes CV macro F1, holdout aggregate and per-class metrics, confusion matrix, ROC-AUC when defined, field importance, runtime, model size, selected method, and model path.

## Research and validation limits

The execution and modeling limits are disclosed in `docs/research_report.md`. The experiment budget currently remains resource-conscious; scores are exploratory. Auto-detection is heuristic; multimodal regression is not supported. The CPU-only host does not verify NVIDIA execution. GPU, arbitrary external datasets and clinical validity are not claimed.

## Change log

- Added the modality classifier/report and upload metadata.
- Added a self-contained sparse-aware multimodal classifier, optional sentence encoder, CV/Optuna model selection, holdout reporting, and complete estimator export.
- Added multimodal train/predict endpoints and frontend controls without removing the tabular workflow.
- Added optional sentence-transformers installation instructions.
- Added adaptive representation/fusion/model parameter search, OOF-only late fusion weighting, holdout attribution, and prediction upload/download.
- Added four UCI datasets, leakage-aware experiment definitions, repeated E1–E8 runner, resource metrics, charts, ablations, and report/presentation generators.
