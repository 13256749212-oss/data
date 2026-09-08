@echo off
setlocal
cd /d "%~dp0\..\.."

echo ============================================================
echo Station 2 - all PCI points cluster-center localization

echo Uses all valid measured points from PCI 699 / 700 / 701.
echo Output: outputs\localization_all_pci_clusters\station_02

echo ============================================================

python run_pipeline.py localize-all-pci --station-id 2
if errorlevel 1 (
    echo.
    echo [ERROR] All-PCI cluster localization failed.
    pause
    exit /b 1
)

echo.
echo [OK] Finished.
pause
endlocal
