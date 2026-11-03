@echo off
REM Runs linting and the automated test suite (no models needed - uses the simulator).
setlocal
cd /d "%~dp0\..\backend"
.venv\Scripts\python -m ruff check . || exit /b 1
.venv\Scripts\python -m pytest -q || exit /b 1
cd ..\frontend
call npm run build || exit /b 1
echo All checks passed.
endlocal
