@echo off
setlocal

set "WIN=%USERPROFILE%\Downloads\AQUA_VERA_windows_frozen_v0_4"
set "GEN=%USERPROFILE%\Downloads\AQUA_VERA_generated_v0_3a"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_graph_gcn_gru_v0_6"

echo ============================================================
echo AQUA-VERA GRAPH GCN-GRU v0.6
echo ============================================================
echo.
echo TRAIN -> optimization
echo VALIDATION -> early stopping
echo CALIBRATION -> threshold
echo TEST -> BLOCKED
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

%PY% "%~dp001_train_graph_gcn_gru.py" ^
  --windows "%WIN%" ^
  --generated "%GEN%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo GRAPH TRAINING FAILED.
    echo Held-out test remains untouched.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS - GRAPH MODEL FROZEN PRE-TEST
echo ============================================================
echo.
echo Send:
echo %OUT%\graph_summary.txt
echo.
pause
