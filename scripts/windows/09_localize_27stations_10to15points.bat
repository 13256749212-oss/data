@echo off
cd /d "%~dp0\..\.."
echo Random localization: 10-15 points, 10 independent trials per point count
python run_pipeline.py localize-sweep --point-counts 10,11,12,13,14,15 --random-trials 10 --direction-prior-mode fixed --bootstrap 0
pause
