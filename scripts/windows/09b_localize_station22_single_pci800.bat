@echo off
setlocal
cd /d "%~dp0\..\.."

echo ============================================================
echo Station 22 single-PCI localization: PCI 800
 echo ============================================================

python run_pipeline.py localize-all-pci --station-id 22

if errorlevel 1 (
    echo [ERROR] Station 22 localization failed.
    pause
    exit /b 1
)

echo [OK] Station 22 completed.
echo Output: outputs\localization_all_pci_clusters\station_22
pause
endlocal
