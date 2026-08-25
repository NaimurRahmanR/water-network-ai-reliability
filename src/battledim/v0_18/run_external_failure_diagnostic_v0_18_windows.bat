@echo off
setlocal
set "MR=%USERPROFILE%\Downloads\AQUA_VERA_model_ready"
set "TRAIN=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_train_frozen_v0_16b"
set "EXT=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_external_2019_v0_17"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_external_failure_diagnostic_v0_18"

echo ============================================================
echo AQUA-VERA EXTERNAL TRANSFER FAILURE DIAGNOSTIC v0.18
echo ============================================================
echo POST-HOC READ-ONLY. NO TRAINING. NO RETUNING.
echo.

where py >nul 2>&1
if %errorlevel%==0 (set "PY=py") else (set "PY=python")

%PY% -c "import pandas,numpy,sklearn" >nul 2>&1
if errorlevel 1 (
  %PY% -m pip install pandas numpy scikit-learn
  if errorlevel 1 (
    echo ERROR: dependency installation failed.
    pause
    exit /b 1
  )
)

%PY% -u "%~dp001_external_failure_diagnostic.py" ^
 --model-ready "%MR%" ^
 --train-freeze "%TRAIN%" ^
 --external-2019 "%EXT%" ^
 --out "%OUT%"

if errorlevel 1 (
  echo DIAGNOSTIC FAILED.
  pause
  exit /b 1
)

echo.
echo SUCCESS. Send:
echo %OUT%\transfer_failure_diagnostic_summary_v0_18.txt
pause
