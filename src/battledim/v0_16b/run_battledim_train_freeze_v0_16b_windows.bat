@echo off
setlocal

set "MR=%USERPROFILE%\Downloads\AQUA_VERA_model_ready"
set "DATA=%USERPROFILE%\Downloads\AQUA_VERA_data"
set "PROTO=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_protocol_frozen_v0_15"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_train_frozen_v0_16b"

echo ============================================================
echo AQUA-VERA BATTLEDIM TRAIN + RELIABILITY FREEZE v0.16b
echo ============================================================
echo.
echo 2018 ONLY.
echo 2019 will NOT be checked, opened, parsed, or hashed.
echo.
echo This run trains the fixed 25-epoch MLP and fixed 28-epoch sparse
echo GCN-GRU, then freezes attribution references, reliability thresholds,
echo and pressure reconstruction.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py"
) else (
  set "PY=python"
)

%PY% -c "import torch,sklearn,pandas,numpy" >nul 2>&1
if errorlevel 1 (
  echo Installing required packages...
  %PY% -m pip install torch scikit-learn pandas numpy
  if errorlevel 1 (
    echo ERROR: dependency installation failed.
    pause
    exit /b 1
  )
)

%PY% -u "%~dp001_train_freeze_battledim_2018.py" ^
  --model-ready "%MR%" ^
  --data "%DATA%" ^
  --protocol-freeze "%PROTO%" ^
  --out "%OUT%"

if errorlevel 1 (
  echo.
  echo TRAIN/FREEZE FAILED.
  echo Do not inspect 2019.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo SUCCESS
echo ============================================================
echo.
echo Send:
echo %OUT%\BattLeDIM_train_freeze_summary_v0_16b.txt
echo.
pause
