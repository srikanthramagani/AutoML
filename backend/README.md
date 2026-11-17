# Backend

FastAPI boundary and working AutoML package for the Adaptive AutoML Platform. The existing engine modules live under `backend/app`; upload and training use filesystem storage plus development in-memory job metadata.

Install dependencies from the repository root:

```powershell
python -m pip install -r backend\requirements.txt
```

Run from the repository root:

```powershell
python -m uvicorn backend.app.main:app --reload
```

Or, if your terminal is already inside the `backend` folder:

```powershell
python -m uvicorn app.main:app --reload
```
