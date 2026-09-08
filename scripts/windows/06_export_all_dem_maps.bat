@echo off
setlocal
cd /d "%~dp0\..\.."
python run_pipeline.py export-dem --stations all --continue-on-error
if errorlevel 1 pause
endlocal
