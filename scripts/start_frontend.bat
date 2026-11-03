@echo off
REM Optional: Vite dev server with hot reload on http://127.0.0.1:5173
REM (only needed when you change frontend code; the backend proxy target is 127.0.0.1:8000)
cd /d "%~dp0\..\frontend"
call npm run dev
