@echo off
REM ============================================================
REM  MFL thesis - download PipeMFL-240K test_data (about 110 GB) and extract
REM  Everything is written under this folder (D:). Safe to re-run: resumes.
REM ============================================================
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "HF_HOME=%ROOT%hf_cache"
set "HF_HUB_ENABLE_HF_TRANSFER=1"
set "PIP_CACHE_DIR=%ROOT%pip_cache"
call "%ROOT%venv\Scripts\activate.bat" || (echo Run setup.bat first & pause & exit /b 1)
echo Download + extract to %ROOT%data  (this can take several hours; leave the window open)
python 01_download_and_extract.py --out "%ROOT%data"
echo.
echo ===== DONE. Check %ROOT%data\test_data =====
pause
endlocal
