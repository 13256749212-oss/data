@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0\..\.."

echo ================================================================
echo Compare saved 4000x3000 joint best-server map with measurements
echo This step does not run Sionna RT.
echo ================================================================

if not exist "outputs\joint_best_server_4000x3000\joint_best_server_27stations_4000x3000.npz" (
  echo [ERROR] Missing saved joint map NPZ.
  echo Run scripts\windows\10_generate_joint_27stations_best_server_4000x3000.bat first.
  pause
  exit /b 2
)
if not exist "data\processed\cell_pci_rsrp_long_27stations.csv" (
  echo [ERROR] Missing data\processed\cell_pci_rsrp_long_27stations.csv
  pause
  exit /b 2
)

python run_pipeline.py compare-joint-map
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [OK] Results: outputs\joint_map_measurement_comparison
) else (
  echo [ERROR] Comparison failed with exit code %RC%.
)
pause
exit /b %RC%
