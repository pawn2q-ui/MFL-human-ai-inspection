@echo off
REM ============================================================
REM  MFL thesis - one-time environment setup (everything on D:)
REM  Creates venv in this folder, installs PyTorch(CUDA) + ultralytics,
REM  then checks the GPU and runs the self-test.
REM ============================================================
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PIP_CACHE_DIR=%ROOT%pip_cache"
set "HF_HOME=%ROOT%hf_cache"
set "TORCH_HOME=%ROOT%torch_cache"

echo [1/6] Looking for Python 3.10-3.12 ...
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo.
    echo   Python was not found. Install Python 3.11 from https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add python.exe to PATH" in the installer, then run this file again.
    pause
    exit /b 1
)
%PY% --version

echo [2/6] Creating virtual environment in %ROOT%venv ...
if not exist "%ROOT%venv\Scripts\python.exe" (
    %PY% -m venv "%ROOT%venv" || (echo venv creation failed & pause & exit /b 1)
)
call "%ROOT%venv\Scripts\activate.bat"
python -m pip install --upgrade pip wheel

echo [3/6] Installing analysis packages ...
pip install numpy pandas opencv-python matplotlib scipy scikit-learn pyyaml huggingface_hub hf_transfer || (echo package install failed & pause & exit /b 1)

echo [4/6] Installing PyTorch with CUDA 12.1 (about 2.5 GB, goes to D:) ...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo   cu121 failed, trying CUDA 11.8 build ...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 || (echo torch install failed & pause & exit /b 1)
)

echo [5/6] Installing ultralytics (YOLOv8) ...
pip install ultralytics || (echo ultralytics install failed & pause & exit /b 1)

echo [6/6] Checking GPU and running the self-test ...
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -m mfl_thesis.selftest --work selftest_out
echo.
echo ===== SETUP FINISHED. If you see "CUDA available: True" and the self-test passed, you are ready. =====
pause
endlocal
