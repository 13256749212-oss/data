@echo off
setlocal
cd /d "%~dp0\..\.."
python run_pipeline.py prepare-data
if errorlevel 1 pause
endlocal
