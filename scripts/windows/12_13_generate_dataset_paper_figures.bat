@echo off
setlocal
cd /d "%~dp0\..\.."
python run_pipeline.py visualize-dataset
if errorlevel 1 exit /b %errorlevel%
echo.
echo Measurement figures and dataset structure figure completed.
pause
