@echo off
setlocal
cd /d "%~dp0\..\.."
python run_pipeline.py calibrate --stations 3 --quick --force
if errorlevel 1 pause
endlocal
