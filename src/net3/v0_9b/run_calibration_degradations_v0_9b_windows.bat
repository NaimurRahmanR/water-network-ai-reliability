@echo off
setlocal

set "WIN=%USERPROFILE%\Downloads\AQUA_VERA_windows_frozen_v0_4"
set "GEN=%USERPROFILE%\Downloads\AQUA_VERA_generated_v0_3a"
set "BASE=%USERPROFILE%\Downloads\AQUA_VERA_baseline_mlp_v0_5"
set "GRAPH=%USERPROFILE%\Downloads\AQUA_VERA_graph_gcn_gru_v0_6"
set "PROTO=%USERPROFILE%\Downloads\AQUA_VERA_degradation_protocol_frozen_v0_8"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_calibration_degradations_v0_9b"

echo ============================================================
echo AQUA-VERA CALIBRATION DEGRADATIONS v0.9b
echo ============================================================
echo.
echo Patch note:
echo v0.8 protocol content is unchanged.
echo This version verifies the original frozen source hash AND the
echo Windows CRLF copy hash separately.
echo.
echo Calibration only. Degraded test remains sealed.
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

%PY% "%~dp001_generate_calibration_degradations_v0_9b.py" ^
  --windows "%WIN%" ^
  --generated "%GEN%" ^
  --baseline "%BASE%" ^
  --graph "%GRAPH%" ^
  --degradation-protocol "%PROTO%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo CALIBRATION DEGRADATION RUN FAILED.
    echo Degraded test remains sealed.
    pause
    exit /b 1
)

echo.
echo SUCCESS.
echo Send:
echo %OUT%\calibration_degradation_summary.txt
echo.
pause
