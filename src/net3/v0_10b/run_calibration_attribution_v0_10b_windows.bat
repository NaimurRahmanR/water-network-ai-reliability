@echo off
setlocal

set "WIN=%USERPROFILE%\Downloads\AQUA_VERA_windows_frozen_v0_4"
set "GEN=%USERPROFILE%\Downloads\AQUA_VERA_generated_v0_3a"
set "BASE=%USERPROFILE%\Downloads\AQUA_VERA_baseline_mlp_v0_5"
set "GRAPH=%USERPROFILE%\Downloads\AQUA_VERA_graph_gcn_gru_v0_6"
set "PROTO=%USERPROFILE%\Downloads\AQUA_VERA_degradation_protocol_frozen_v0_8"
set "CAL=%USERPROFILE%\Downloads\AQUA_VERA_calibration_degradations_v0_9b"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_calibration_attribution_v0_10b"

echo ============================================================
echo AQUA-VERA CALIBRATION ATTRIBUTION v0.10b
echo ============================================================
echo.
echo Patch note:
echo v0.10 stopped because the audit compared a 20-row GCN context batch
echo against the full 600-row frozen condition lookup.
echo.
echo v0.10b audits each batch against its exact frozen window IDs.
echo Scientific method is unchanged.
echo Degraded held-out test remains sealed.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% -c "import torch,sklearn,pandas,numpy,networkx" >nul 2>&1
if errorlevel 1 (
    %PY% -m pip install torch scikit-learn pandas numpy networkx
    if errorlevel 1 (
        echo ERROR: dependency installation failed.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_calibration_attribution_v0_10b.py" ^
  --windows "%WIN%" ^
  --generated "%GEN%" ^
  --baseline "%BASE%" ^
  --graph "%GRAPH%" ^
  --degradation-protocol "%PROTO%" ^
  --calibration-degradations "%CAL%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo ATTRIBUTION RUN FAILED.
    echo Degraded test remains sealed.
    pause
    exit /b 1
)

echo.
echo SUCCESS.
echo Send:
echo %OUT%\calibration_attribution_summary.txt
echo.
pause
