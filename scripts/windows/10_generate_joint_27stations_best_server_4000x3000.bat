@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0\..\.."

echo ================================================================
echo 27站联合 best-server 无线电地图（4000m x 3000m, 1m网格）
echo 使用调参后的最佳高度、功率、方向角和下倾角
echo 不重新生成27张单站地图
echo ================================================================

if not exist "outputs\parameter_calibration\all_27stations_summary.csv" (
  echo [ERROR] 缺少 outputs\parameter_calibration\all_27stations_summary.csv
  echo 请先运行: python run_pipeline.py calibrate --stations all
  pause
  exit /b 1
)
if not exist "outputs\parameter_calibration\estimated_initial_directions_27stations.csv" (
  echo [ERROR] 缺少 outputs\parameter_calibration\estimated_initial_directions_27stations.csv
  echo 请先运行: python run_pipeline.py calibrate --stations all
  pause
  exit /b 1
)
if not exist "assets\ground.ply" (
  echo [ERROR] 缺少 assets\ground.ply
  pause
  exit /b 1
)
if not exist "assets\ynu_chenggong_campus-001.ply" (
  echo [ERROR] 缺少 assets\ynu_chenggong_campus-001.ply
  pause
  exit /b 1
)

echo.
echo [1/2] 先检查27站最佳参数、地图范围和48个tile...
python run_pipeline.py export-joint-map --dry-run
if errorlevel 1 (
  echo [ERROR] 联合地图输入检查失败。
  pause
  exit /b 1
)

echo.
echo [2/2] 开始正式联合地图。已完成tile会自动复用。
python run_pipeline.py export-joint-map --continue-on-error
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
  echo [OK] 完成。结果目录:
  echo outputs\joint_best_server_4000x3000
) else (
  echo [WARNING] 有tile失败或缺失。修复问题后重新运行本BAT即可续跑。
  echo 不要添加 --force，程序会复用已经完成的tile。
)
pause
exit /b %RC%
