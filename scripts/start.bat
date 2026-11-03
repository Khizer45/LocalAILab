@echo off
REM Starts the backend (which also serves the built UI) and opens the browser
REM only when the backend is ready to answer.
setlocal
cd /d "%~dp0\.."
where ollama >nul 2>nul && (
  ollama list >nul 2>nul || (echo Starting Ollama... & start "" /min ollama serve & timeout /t 3 >nul)
)
echo Starting Local AI Lab... the browser opens by itself when the backend is ready (10-30 s).
echo Keep this window open. Press Ctrl+C here to stop the app.
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8000/api/health | Out-Null; Start-Process 'http://127.0.0.1:8000'; exit } catch { Start-Sleep -Seconds 1 } }"
call scripts\start_backend.bat
endlocal
