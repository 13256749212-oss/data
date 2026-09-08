@echo off
setlocal
cd /d "%~dp0\..\.."

echo ============================================================
echo All-station all-PCI cluster localization
echo - 3-sector sites: use all three mapped PCIs
 echo - Station 22: use its single omnidirectional PCI 800
 echo - No Windows FOR loop is required
 echo ============================================================

python run_pipeline.py localize-all-pci --station-id all

if errorlevel 1 (
    echo.
    echo [ERROR] Localization finished with an error. Check the console output and:
    echo outputs\localization_all_pci_clusters\all_stations_all_pci_failures.csv
    pause
    exit /b 1
)

echo.
echo [OK] All available physical stations completed.
echo Output root:
echo outputs\localization_all_pci_clusters
pause
endlocal
