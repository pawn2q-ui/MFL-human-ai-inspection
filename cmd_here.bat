@echo off
REM Opens a command window in this folder with the virtual environment activated.
REM Use it to run any "python -m mfl_thesis.xxx" command from README.md.
cd /d "%~dp0"
set "HF_HOME=%~dp0hf_cache"
set "PIP_CACHE_DIR=%~dp0pip_cache"
call "%~dp0venv\Scripts\activate.bat"
cmd /k
