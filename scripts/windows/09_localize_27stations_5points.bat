@echo off
setlocal
set "PROJECT_ROOT=%~dp0\..\.."
cd /d "%PROJECT_ROOT%"

set "MEASUREMENTS=%PROJECT_ROOT%\data\processed\cell_pci_rsrp_long_27stations.csv"

if not exist "%MEASUREMENTS%" (
    echo [ERROR] Missing localization input:
    echo         %MEASUREMENTS%
    echo Please run: python run_pipeline.py prepare-data
    pause
    exit /b 1
)

python run_pipeline.py localize ^
  --measurements "%MEASUREMENTS%" ^
  --points-per-station 5 ^
  --direction-prior-mode fixed ^
  --bootstrap 20

if errorlevel 1 pause
endlocal
