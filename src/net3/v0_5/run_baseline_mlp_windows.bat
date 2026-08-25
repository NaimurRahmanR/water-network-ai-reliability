@echo off
setlocal

set "WIN=%USERPROFILE%\Downloads\AQUA_VERA_windows_frozen_v0_4"
set "GEN=%USERPROFILE%\Downloads\AQUA_VERA_generated_v0_3a"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_baseline_mlp_v0_5"

echo ============================================================
echo AQUA-VERA NON-GRAPH TEMPORAL MLP BASELINE v0.5
echo ============================================================
echo.
echo IMPORTANT:
echo   This script trains on TRAIN, early-stops on VALIDATION,
echo   chooses threshold on CALIBRATION, and DOES NOT LOAD TEST.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% -c "import torch, sklearn, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    %PY% -m pip install torch scikit-learn pandas numpy
    if errorlevel 1 (
        echo ERROR: Could not install dependencies.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_train_baseline_mlp.py" ^
  --windows "%WIN%" ^
  --generated "%GEN%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo BASELINE TRAINING FAILED.
    echo Held-out test remains untouched.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS - BASELINE FROZEN PRE-TEST
echo ============================================================
echo.
echo Send:
echo %OUT%\baseline_summary.txt
echo.
pause
