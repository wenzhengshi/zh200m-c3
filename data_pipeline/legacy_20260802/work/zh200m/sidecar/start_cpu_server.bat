@echo off
rem llama.cpp local inference sidecar (Windows)
set LLAMA_DIR=C:\llamacpp
set MODEL=C:\path\to\model-q5_k_m.gguf
set THREADS=4
set CTX=2048
set PORT=8080
"%LLAMA_DIR%\llama-server.exe" -m "%MODEL%" -t %THREADS% -c %CTX% --host 127.0.0.1 --port %PORT% --no-webui
