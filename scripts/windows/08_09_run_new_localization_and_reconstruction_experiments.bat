@echo off
cd /d "%~dp0\..\.."
echo [1/2] Nearest-neighbor reconstruction from 1%%-10%% measured points
python run_pipeline.py reconstruct --station-id 3 --pci 558 --percentages 1,2,3,4,5,6,7,8,9,10
if errorlevel 1 goto :error
echo [2/2] Random localization: 10-15 points x 10 trials
python run_pipeline.py localize-sweep --point-counts 10,11,12,13,14,15 --random-trials 10 --direction-prior-mode fixed --bootstrap 0
if errorlevel 1 goto :error
echo Done.
pause
exit /b 0
:error
echo Experiment failed.
pause
exit /b 1
