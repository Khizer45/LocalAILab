@echo off
REM Starts the FastAPI backend on http://127.0.0.1:8000
REM If frontend\dist exists (after setup), the backend also serves the web UI.
setlocal
cd /d "%~dp0\..\backend"
if not exist .venv\Scripts\python.exe (
  echo Run scripts\setup_windows.bat first.
  exit /b 1
)
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
endlocal
