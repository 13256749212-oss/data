@echo off
setlocal
cd /d "%~dp0\..\.."
python run_pipeline.py compare-surfaces --stations all --methods both --continue-on-error
if errorlevel 1 pause
endlocal
