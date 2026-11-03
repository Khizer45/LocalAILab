@echo off
rem Local AI Lab command line - e.g.  scripts\lal.bat backup create
rem                                   scripts\lal.bat bench run --suite quick --model qwen3:1.7b --gate-quality 0.7
rem Runs in the current folder, so relative file names (bundle -o run.zip, backup verify FILE) work as expected.
setlocal
set "PYTHONPATH=%~dp0..\backend"
"%~dp0..\backend\.venv\Scripts\python.exe" -m app.cli %*
set "LAL_EXIT=%ERRORLEVEL%"
endlocal & exit /b %LAL_EXIT%
