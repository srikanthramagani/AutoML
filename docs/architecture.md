# Architecture

The repository has two intentionally separate layers:

- The existing root `app/` package is the working Phase 1 AutoML engine.
- `backend/` is the FastAPI integration boundary being scaffolded around that engine.
- `frontend/` is the React/Vite browser client.
- `database/` documents metadata persistence; CSV and model binaries remain in filesystem or object storage.

The engine is organized as a coordinator over loading, profiling, task detection, preprocessing, candidate models, optimization, evaluation, training, and export. Later phases can add feature engineering, explainability, neural, text, or image modules without changing the API boundary.