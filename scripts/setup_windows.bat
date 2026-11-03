@echo off
REM ============================================================
REM  Local AI Lab - one-time setup for Windows (CMD)
REM  Run from the project folder:  scripts\setup_windows.bat
REM ============================================================
setlocal
cd /d "%~dp0\.."

echo.
echo [1/6] Checking prerequisites...
where python >nul 2>nul
if errorlevel 1 (
  echo   Python was not found. Install Python 3.11, 3.12 or 3.13 from https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH" during installation.
  exit /b 1
)
python --version
where npm >nul 2>nul
if errorlevel 1 (
  echo   Node.js was not found. Install the LTS version from https://nodejs.org/
  exit /b 1
)
node --version
where ollama >nul 2>nul
if errorlevel 1 (
  echo   WARNING: Ollama was not found. Install it from https://ollama.com/download/windows
  echo   The app installs anyway; chat and benchmarks need Ollama to be running.
) else (
  ollama --version
)

echo.
echo [2/6] Creating configuration file...
if not exist .env (
  copy .env.example .env >nul
  echo   Created .env from .env.example
) else (
  echo   .env already exists - left unchanged
)

echo.
echo [3/6] Creating Python virtual environment in backend\.venv ...
if not exist backend\.venv\Scripts\python.exe (
  python -m venv backend\.venv || exit /b 1
)

echo.
echo [4/6] Installing backend packages (a few minutes the first time)...
backend\.venv\Scripts\python -m pip install --upgrade pip
backend\.venv\Scripts\python -m pip install -r backend\requirements-dev.txt || exit /b 1

echo.
echo [5/6] Installing and building the frontend...
pushd frontend
call npm install || (popd & exit /b 1)
call npm run build || (popd & exit /b 1)
popd

echo.
echo [6/6] Checking this computer...
backend\.venv\Scripts\python scripts\check_system.py
backend\.venv\Scripts\python scripts\make_arena_samples.py --joblib >nul

echo.
echo Setup complete.
echo   1. Download models:   scripts\pull_models.bat
echo   2. Start the app:     scripts\start.bat
echo   Optional: scripts\download_backbones.bat  (image feature extractors for the Model Arena)
echo   3. Open:              http://127.0.0.1:8000
endlocal
