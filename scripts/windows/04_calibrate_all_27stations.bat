@echo off
setlocal
cd /d "%~dp0\..\.."
python run_pipeline.py calibrate --stations all
if errorlevel 1 pause
endlocal
