@echo off
setlocal
cd /d "%~dp0\..\.."
call conda activate sionna_env
if errorlevel 1 (
  echo [ERROR] Unable to activate sionna_env.
  pause
  exit /b 1
)
python run_pipeline.py migrate-output-names
set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo [WARNING] Migration returned code %RC%. Check conflicts above.
pause
exit /b %RC%
