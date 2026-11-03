@echo off
REM ============================================================
REM  Local AI Lab - update an existing installation
REM  Run after copying a new version over your project folder.
REM  Your data (backend\data), .env and backend\.venv are kept.
REM ============================================================
setlocal
cd /d "%~dp0\.."
if not exist backend\.venv\Scripts\python.exe (
  echo No virtual environment found - run scripts\setup_windows.bat first.
  exit /b 1
)
echo.
echo [1/3] Installing new / updated Python packages...
backend\.venv\Scripts\python -m pip install -r backend\requirements-dev.txt || exit /b 1
echo.
echo [2/3] Rebuilding the user interface...
pushd frontend
call npm install || (popd & exit /b 1)
call npm run build || (popd & exit /b 1)
popd
echo.
echo [3/3] Generating the Model Arena sample files (samples\arena)...
backend\.venv\Scripts\python scripts\make_arena_samples.py --joblib
echo.
echo Update complete. Optional: scripts\download_backbones.bat (image feature extractors, 63 MB)
echo Start the app with scripts\start.bat
endlocal
