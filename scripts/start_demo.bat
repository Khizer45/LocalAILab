@echo off
REM Demo mode WITHOUT real models: starts the Ollama simulator on port 11500
REM and a separate backend with its own data folder (backend\data-demo).
REM Numbers shown in demo mode are NOT real model measurements.
setlocal
cd /d "%~dp0\..\backend"
if not exist .venv\Scripts\python.exe (echo Run scripts\setup_windows.bat first. & exit /b 1)
start "Ollama simulator" .venv\Scripts\python -m devtools.fake_ollama --port 11500 --speed 0.5
set OLLAMA_BASE_URL=http://127.0.0.1:11500
set EMBEDDING_MODEL=sim-embed
set DEFAULT_MODEL=sim-small:1b
set DATA_DIR=%CD%\data-demo
timeout /t 2 >nul
start "" http://127.0.0.1:8000
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
endlocal
