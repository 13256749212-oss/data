@echo off
setlocal
cd /d "%~dp0\..\.."
python run_pipeline.py export-dem --stations 3 --force
if errorlevel 1 pause
endlocal
