# run_all.ps1
# Launch both backend (FastAPI) and frontend (Vite React) in separate background processes

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Starting Adaptive AutoML Full Platform     " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

# 1. Start Backend API
Write-Host "[1/2] Starting Backend API on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process -FilePath $venvPython -ArgumentList "-m uvicorn backend.app.main:app --port 8000" -WorkingDirectory $projectRoot

# 2. Start Frontend Dev Server
Write-Host "[2/2] Starting Frontend on http://localhost:5173 ..." -ForegroundColor Green
$frontendDir = Join-Path $projectRoot "frontend"
Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -WorkingDirectory $frontendDir

Write-Host ""
Write-Host "Platform is up and running!" -ForegroundColor Cyan
Write-Host "  -> Frontend UI : http://localhost:5173" -ForegroundColor Yellow
Write-Host "  -> Backend API: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
