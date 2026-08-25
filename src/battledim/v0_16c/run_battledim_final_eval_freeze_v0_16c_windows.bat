@echo off
setlocal

set "TRAIN=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_train_frozen_v0_16b"
set "DATA=%USERPROFILE%\Downloads\AQUA_VERA_data"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_final_eval_frozen_v0_16c"

echo ============================================================
echo AQUA-VERA BATTLEDIM FINAL-EVALUATION FREEZE v0.16c
echo ============================================================
echo.
echo PRE-2019.
echo This run does not check, open, parse, or hash 2019.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py"
) else (
  set "PY=python"
)

%PY% -c "import networkx" >nul 2>&1
if errorlevel 1 (
  %PY% -m pip install networkx
  if errorlevel 1 (
    echo ERROR: networkx installation failed.
    pause
    exit /b 1
  )
)

%PY% -u "%~dp001_freeze_final_eval_implementation.py" ^
  --train-freeze "%TRAIN%" ^
  --data "%DATA%" ^
  --out "%OUT%"

if errorlevel 1 (
  echo.
  echo FINAL-EVALUATION FREEZE FAILED.
  echo Do not inspect 2019.
  pause
  exit /b 1
)

echo.
echo SUCCESS.
echo Send:
echo %OUT%\BattLeDIM_final_eval_freeze_summary_v0_16c.txt
echo.
pause
