@echo off
setlocal

set "MR=%USERPROFILE%\Downloads\AQUA_VERA_model_ready"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_protocol_frozen_v0_15"

echo ============================================================
echo AQUA-VERA BATTLEDIM EXTERNAL PROTOCOL FREEZE v0.15
echo ============================================================
echo.
echo 2018 only.
echo 2019 remains sealed.
echo No model training in this run.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py"
) else (
  set "PY=python"
)

%PY% -c "import pandas,numpy" >nul 2>&1
if errorlevel 1 (
  %PY% -m pip install pandas numpy
  if errorlevel 1 (
    echo ERROR: dependency installation failed.
    pause
    exit /b 1
  )
)

%PY% "%~dp002_freeze_battledim_protocol.py" ^
  --model-ready "%MR%" ^
  --protocol "%~dp001_BattLeDIM_external_protocol_v0_15.md" ^
  --out "%OUT%"

if errorlevel 1 (
  echo.
  echo PROTOCOL FREEZE FAILED.
  echo Do not train or inspect 2019.
  pause
  exit /b 1
)

echo.
echo SUCCESS.
echo Send:
echo %OUT%\BattLeDIM_protocol_freeze_summary_v0_15.txt
echo.
pause
