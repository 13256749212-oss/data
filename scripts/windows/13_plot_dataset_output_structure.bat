@echo off
setlocal
cd /d "%~dp0\..\.."
python run_pipeline.py plot-output-structure
if errorlevel 1 exit /b %errorlevel%
echo.
echo Dataset output structure figure completed.
pause
