@echo off
cd /d "%~dp0\..\.."
echo Nearest-neighbor reconstruction from 1%%-10%% of measured points
python run_pipeline.py reconstruct --station-id 3 --pci 558 --percentages 1,2,3,4,5,6,7,8,9,10
pause
