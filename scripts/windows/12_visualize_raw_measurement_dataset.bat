@echo off
setlocal
cd /d "%~dp0\..\.."
python run_pipeline.py visualize-measurements
if errorlevel 1 exit /b %errorlevel%
echo.
echo Raw measurement dataset figures completed.
pause
