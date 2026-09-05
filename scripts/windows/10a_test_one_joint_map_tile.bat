@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0\..\.."

echo 测试中心附近一个500m x 500m tile，使用低采样quick设置。
echo 此命令不会组装4000m x 3000m完整地图。
python run_pipeline.py export-joint-map --quick --only-tile 3,2 --force
pause
