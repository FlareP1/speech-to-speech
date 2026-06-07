@echo off

set PYTHON_VER=3.12

REM Check if Python version meets the recommended version
python --version 2>nul | findstr /b /c:"Python %PYTHON_VER%" >nul
if errorlevel 1 (
    echo Warning: Python version %PYTHON_VER% is recommended.
)

uv venv --python %PYTHON_VER%

call .\.venv\Scripts\activate.bat

uv pip install --upgrade pip
uv sync

REM Install PyTorch with CUDA 12.6 (override uv sync's CPU-only default)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --reinstall
