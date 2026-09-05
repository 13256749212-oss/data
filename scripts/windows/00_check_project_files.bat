@echo off
setlocal
cd /d "%~dp0\..\.."
python check_project_layout.py
if errorlevel 1 pause
endlocal
