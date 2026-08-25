@echo off
setlocal

set "PRED=%USERPROFILE%\Downloads\AQUA_VERA_heldout_test_v0_7b\heldout_test_predictions.csv"
set "BENCH=%USERPROFILE%\Downloads\AQUA_VERA_benchmark_frozen"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_degradation_protocol_frozen_v0_8"

echo ============================================================
echo AQUA-VERA DEGRADATION PROTOCOL FREEZE v0.8
echo ============================================================
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% "%~dp002_freeze_degradation_protocol.py" ^
  --predictions "%PRED%" ^
  --frozen-benchmark "%BENCH%" ^
  --protocol "%~dp001_degradation_protocol_v0_8.md" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo FREEZE FAILED.
    echo Do not generate degradations.
    pause
    exit /b 1
)

echo.
echo SUCCESS.
echo Send:
echo %OUT%\degradation_protocol_summary.txt
echo.
pause
