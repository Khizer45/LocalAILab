@echo off
REM Downloads the pretrained image feature extractors (MobileNetV2 13 MB, EfficientNet-Lite4 50 MB)
REM for the Model Arena. Offline-Only mode can stay on: this script runs outside the app.
setlocal
cd /d "%~dp0\.."
backend\.venv\Scripts\python scripts\download_backbones.py %*
endlocal
