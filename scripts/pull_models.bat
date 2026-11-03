@echo off
REM Downloads a small chat model, a second model to compare, and the embedding model.
REM Run scripts\check_system.py first if you are unsure what your PC can handle.
REM Licences: llama3.2 = Llama 3.2 Community License (conditions, e.g. "Built with Llama" when you distribute);
REM qwen2.5:1.5b and nomic-embed-text = Apache-2.0. See docs\LEGAL_AND_LICENSING.md and the Setup page.
where ollama >nul 2>nul || (echo Ollama is not installed: https://ollama.com/download/windows & exit /b 1)
echo Downloading llama3.2:1b  (about 1.3 GB)...
ollama pull llama3.2:1b || exit /b 1
echo Downloading qwen2.5:1.5b (about 1 GB) - second model for benchmarking...
ollama pull qwen2.5:1.5b || exit /b 1
echo Downloading nomic-embed-text (about 274 MB) - embeddings for document Q^&A...
ollama pull nomic-embed-text || exit /b 1
echo.
ollama list
echo Done. More models (with licence and memory fit) on the Setup page of the app.
