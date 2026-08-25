@echo off
setlocal

set "WIN=%USERPROFILE%\Downloads\AQUA_VERA_windows_frozen_v0_4"
set "GEN=%USERPROFILE%\Downloads\AQUA_VERA_generated_v0_3a"
set "BASE=%USERPROFILE%\Downloads\AQUA_VERA_baseline_mlp_v0_5"
set "GRAPH=%USERPROFILE%\Downloads\AQUA_VERA_graph_gcn_gru_v0_6"
set "CAL=%USERPROFILE%\Downloads\AQUA_VERA_calibration_degradations_v0_9b"
set "ATTR=%USERPROFILE%\Downloads\AQUA_VERA_calibration_attribution_v0_10b"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_controller_frozen_v0_11"

echo ============================================================
echo AQUA-VERA RELIABILITY CONTROLLER FREEZE v0.11
echo ============================================================
echo.
echo Training-only reconstruction.
echo Clean-calibration-only reliability thresholds.
echo Degraded test remains sealed.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py"
) else (
  set "PY=python"
)

%PY% -c "import torch,sklearn,pandas,numpy" >nul 2>&1
if errorlevel 1 (
  %PY% -m pip install torch scikit-learn pandas numpy
  if errorlevel 1 (
    echo ERROR: dependency installation failed.
    pause
    exit /b 1
  )
)

%PY% "%~dp001_freeze_controller.py" ^
  --windows "%WIN%" ^
  --generated "%GEN%" ^
  --baseline "%BASE%" ^
  --graph "%GRAPH%" ^
  --calibration-degradations "%CAL%" ^
  --calibration-attribution "%ATTR%" ^
  --out "%OUT%"

if errorlevel 1 (
  echo.
  echo CONTROLLER FREEZE FAILED.
  echo Degraded test remains sealed.
  pause
  exit /b 1
)

echo.
echo SUCCESS.
echo Send:
echo %OUT%\controller_freeze_summary.txt
echo.
pause
