@echo off
REM Runs queue.txt sequentially (training + inference), overnight. Re-run to resume.
cd /d "%~dp0"
call "%~dp0venv\Scripts\activate.bat" || (echo Run setup.bat first & pause & exit /b 1)
python -m mfl_thesis.run_queue queue.txt
pause
