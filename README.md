# Adaptive AutoML Platform

Research-oriented Phase 1 AutoML for tabular CSV datasets.

## Objective

The project is designed to profile a dataset, infer its task, build preprocessing and model candidates, optimize them within a defined budget, rank complete pipelines, validate the winner, and export the complete inference pipeline as `.pkl`.

## Repository layers

- `backend/app/`: current working Python AutoML engine plus the FastAPI integration boundary.
- `frontend/`: React + Vite browser interface skeleton using Axios.
- `backend/`: FastAPI, Pydantic, storage, and database integration skeleton.
- `database/`: PostgreSQL metadata architecture notes.
- `docs/`: architecture and Phase 1 scope.

## Run the existing engine

```powershell
.\.venv\Scripts\Activate.ps1
python main.py --csv datasets\raw\sample_classification.csv --trials 1 --folds 2
python -m pytest -q
```

## Run the scaffolded backend

```powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API contract. Upload and training run through the existing AutoML engine; PostgreSQL metadata persistence is still optional and not yet connected.

## Run the scaffolded frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal. The interface is a Phase 1 workflow shell; it does not claim that upload or training is complete until the backend adapters are implemented.

## API contract

- `POST /api/upload`
- `GET /api/datasets/{dataset_id}/profile`
- `POST /api/training/start`
- `GET /api/training/{job_id}`
- `GET /api/models/{model_id}`
- `GET /api/models/{model_id}/download`

## Future phases

Feature engineering, explainability, neural networks, image/text/multimodal support, ONNX, deployment, drift detection, and automatic retraining are deliberately out of scope for this scaffold.# Adaptive AutoML Platform

> **Phase 1** ? Research-oriented AutoML for CSV/tabular datasets using scikit-learn, Optuna, pandas, and joblib.

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1 Execution Flow](#phase-1-execution-flow)
3. [Architecture & Module Responsibilities](#architecture--module-responsibilities)
4. [Directory Structure](#directory-structure)
5. [Dependencies](#dependencies)
6. [Quick Start](#quick-start)
7. [Configuration](#configuration)
8. [Running Tests](#running-tests)
9. [Architectural Decisions](#architectural-decisions)
10. [Phase 2+ Roadmap (Not Implemented)](#phase-2-roadmap-not-implemented)

---

## Overview

The **Adaptive AutoML Platform** is a research-grade automated machine learning system for tabular (CSV) datasets. Given a CSV file and a target column, it automatically:

- Profiles the dataset
- Detects the ML task type (binary classification, multiclass classification, or regression)
- Builds a tailored preprocessing pipeline
- Evaluates a pool of candidate scikit-learn models
- Runs Optuna-based hyperparameter optimization per model
- Ranks models by cross-validated score
- Retrains the winner on the full training set
- Validates against a held-out test set
- Exports the complete inference pipeline as a single `.pkl` file

---

## Phase 1 Execution Flow

```
CSV File
   ?
   ?
???????????????????????????????
?  1. DataLoader              ?  Load & validate CSV ? raw DataFrame
???????????????????????????????
               ?
               ?
???????????????????????????????
?  2. DataProfiler            ?  Statistical profiling ? DataProfile
???????????????????????????????
               ?
               ?
???????????????????????????????
?  3. TaskDetector            ?  Detect task type + primary metric
???????????????????????????????
               ?
               ?
???????????????????????????????
?  4. Train / Holdout Split   ?  Stratified split (default 80/20)
???????????????????????????????
               ?
               ?
???????????????????????????????
?  5. PreprocessingPipeline   ?  Outlier clip ? Imputation ?
?     Builder                 ?  Encoding ? Scaling
???????????????????????????????
               ?
               ?
???????????????????????????????
?  6. Model Pool              ?  8 classifiers or 9 regressors
???????????????????????????????
               ?
               ?
???????????????????????????????
?  7. HyperparameterOptimizer ?  Optuna TPE ? N trials per model
?     (OptunaObjective)       ?
???????????????????????????????
               ?
               ?
???????????????????????????????
?  8. CrossValidator          ?  Stratified k-fold CV (default 5)
???????????????????????????????
               ?
               ?
???????????????????????????????
?  9. ModelRanker             ?  Sort by mean CV score ? leaderboard
???????????????????????????????
               ?
               ?
???????????????????????????????
? 10. FinalTrainer            ?  Retrain winner on full training set
???????????????????????????????
               ?
               ?
???????????????????????????????
? 11. MetricsCalculator       ?  Holdout validation metrics
?     + ModelExporter         ?  ? trained_pipeline.pkl + metadata.json
???????????????????????????????
```

---

## Architecture & Module Responsibilities

### `app/config/`
| File | Responsibility |
|------|---------------|
| `settings.py` | Central `Settings` dataclass ? random seed, test size, CV folds, HPO trials, metric, paths. |

### `app/data/`
| File | Responsibility |
|------|---------------|
| `loader.py` | `DataLoader` ? CSV I/O, structural validation, feature/target split. |
| `profiler.py` | `DataProfiler` ? per-column statistics (dtype, missingness, cardinality, skewness). Returns `DataProfile`. |
| `task_detector.py` | `TaskDetector` ? heuristic task inference (binary_classification / multiclass_classification / regression) + imbalance flag. |

### `app/preprocessing/`
| File | Responsibility |
|------|---------------|
| `missing_values.py` | `MissingValueHandler` ? median imputation (numeric), most-frequent (categorical), optional missingness indicators. |
| `encoding.py` | `CategoricalEncoder` ? auto-selects OneHot vs Ordinal encoding based on cardinality. |
| `scaling.py` | `FeatureScaler` ? StandardScaler / MinMaxScaler / RobustScaler on numeric columns only. |
| `outlier_handler.py` | `OutlierHandler` ? IQR-based clip (default) or row removal. |
| `pipeline_builder.py` | `PreprocessingPipelineBuilder` ? assembles the above into a single unfitted `sklearn.Pipeline`. |

### `app/models/`
| File | Responsibility |
|------|---------------|
| `model_registry.py` | `ModelRegistry` ? central registry mapping name ? estimator factory, per task type. |
| `classification.py` | Phase 1 classification pool: LogisticRegression, Ridge, KNN, DecisionTree, RandomForest, GradientBoosting, SVC, GaussianNB. |
| `regression.py` | Phase 1 regression pool: LinearRegression, Ridge, Lasso, ElasticNet, KNN, DecisionTree, RandomForest, GradientBoosting, SVR. |

### `app/optimization/`
| File | Responsibility |
|------|---------------|
| `search_space.py` | `SearchSpace` ? Optuna hyperparameter search space definitions for every Phase 1 model. |
| `objective.py` | `OptunaObjective` ? callable that samples params ? fits pipeline ? runs CV ? returns score. |
| `optimizer.py` | `HyperparameterOptimizer` ? runs one Optuna study per model, collects `ModelOptimizationResult`. |

### `app/evaluation/`
| File | Responsibility |
|------|---------------|
| `cross_validation.py` | `CrossValidator` ? StratifiedKFold (classification) / KFold (regression) on full pipelines. Returns `CVResult`. |
| `metrics.py` | `MetricsCalculator` ? holdout metrics: accuracy/F1/ROC-AUC/MCC (classification), R?/MAE/RMSE/MAPE (regression). |
| `model_ranker.py` | `ModelRanker` ? sorts `CVResult` list by mean score, breaks ties by std, prints leaderboard. |

### `app/training/`
| File | Responsibility |
|------|---------------|
| `trainer.py` | `Trainer` ? concatenates preprocessing + estimator into a full `Pipeline`, fits it (used during HPO and CV). |
| `final_training.py` | `FinalTrainer` ? retrains the winning pipeline on the entire training partition before holdout evaluation. |

### `app/export/`
| File | Responsibility |
|------|---------------|
| `model_exporter.py` | `ModelExporter` ? serializes complete pipeline via `joblib.dump`, writes companion `_metadata.json`. |

### `app/automl/`
| File | Responsibility |
|------|---------------|
| `engine.py` | `AutoMLEngine` ? top-level orchestrator; `.fit(csv_path, target_column)` runs all 11 steps. |

---

## Directory Structure

```
adaptive-automl/
?
??? app/
?   ??? __init__.py
?   ??? config/
?   ?   ??? __init__.py
?   ?   ??? settings.py
?   ??? data/
?   ?   ??? __init__.py
?   ?   ??? loader.py
?   ?   ??? profiler.py
?   ?   ??? task_detector.py
?   ??? preprocessing/
?   ?   ??? __init__.py
?   ?   ??? missing_values.py
?   ?   ??? encoding.py
?   ?   ??? scaling.py
?   ?   ??? outlier_handler.py
?   ?   ??? pipeline_builder.py
?   ??? models/
?   ?   ??? __init__.py
?   ?   ??? model_registry.py
?   ?   ??? classification.py
?   ?   ??? regression.py
?   ??? optimization/
?   ?   ??? __init__.py
?   ?   ??? search_space.py
?   ?   ??? optimizer.py
?   ?   ??? objective.py
?   ??? evaluation/
?   ?   ??? __init__.py
?   ?   ??? cross_validation.py
?   ?   ??? metrics.py
?   ?   ??? model_ranker.py
?   ??? training/
?   ?   ??? __init__.py
?   ?   ??? trainer.py
?   ?   ??? final_training.py
?   ??? export/
?   ?   ??? __init__.py
?   ?   ??? model_exporter.py
?   ??? automl/
?       ??? __init__.py
?       ??? engine.py
?
??? tests/
?   ??? test_loader.py
?   ??? test_profiler.py
?   ??? test_preprocessing.py
?   ??? test_models.py
?   ??? test_optimizer.py
?   ??? test_evaluation.py
?
??? datasets/
?   ??? raw/           <- Drop your CSV files here
?   ??? processed/     <- Cached preprocessed data
?
??? models/            <- Exported .pkl pipelines land here
??? experiments/       <- Experiment logs and result JSONs
??? notebooks/         <- Jupyter notebooks for exploration
?
??? requirements.txt
??? README.md
??? .gitignore
??? main.py
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | >= 1.4.0 | Primary ML framework ? estimators, pipelines, CV |
| `pandas` | >= 2.1.0 | DataFrame I/O and manipulation |
| `numpy` | >= 1.26.0 | Numerical operations |
| `optuna` | >= 3.6.0 | Hyperparameter optimization (TPE sampler) |
| `joblib` | >= 1.3.0 | Pipeline serialization to `.pkl` |
| `pytest` | >= 8.0.0 | Unit testing framework |
| `pytest-cov` | >= 5.0.0 | Coverage reporting |
| `tqdm` | >= 4.66.0 | Progress bars in optimizer loops |
| `rich` | >= 13.7.0 | Formatted console output (leaderboard, summaries) |

---

## Quick Start

```bash
# 1. Enter the project directory
cd adaptive-automl

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run on your CSV
python main.py --csv datasets/raw/your_data.csv --target target_column
```

---

## Architectural Decisions

### 1. Preprocessing + Model as One Pipeline
The complete inference artifact is a single `sklearn.Pipeline` object containing both preprocessing steps and the estimator. This means loading `trained_pipeline.pkl` and calling `.predict()` on a raw DataFrame is sufficient for inference.

### 2. Data-Leak-Free Cross-Validation
Preprocessing statistics (scaler means, imputer values, encoder categories) are learned inside each CV fold on the training split only. The full pipeline is passed to `cross_val_score`, not a pre-fitted transformer.

### 3. One Optuna Study per Model
Each model gets its own independent Optuna study with a TPE sampler. This avoids search space entanglement and allows models to be evaluated in parallel.

### 4. Settings as a Dataclass
A single `Settings` dataclass is the single source of truth for all configuration. No magic numbers are scattered across modules.

### 5. Modular, Independently Testable Components
Each module is a standalone class with no hidden global state. This makes unit testing straightforward and components easy to swap.
