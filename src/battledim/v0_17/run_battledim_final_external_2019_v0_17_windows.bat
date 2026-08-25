@echo off
setlocal

set "MR=%USERPROFILE%\Downloads\AQUA_VERA_model_ready"
set "DATA=%USERPROFILE%\Downloads\AQUA_VERA_data"
set "TRAIN=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_train_frozen_v0_16b"
set "FREEZE=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_final_eval_frozen_v0_16c"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_external_2019_v0_17"

echo ============================================================
echo AQUA-VERA FINAL BATTLEDIM 2019 EXTERNAL EVALUATION v0.17
echo ============================================================
echo.
echo This is the final external evaluation.
echo It will verify every frozen pre-2019 artifact before unsealing 2019.
echo No training or tuning occurs in this run.
echo.
echo The GCN and attribution stages are computationally heavy.
echo Progress is printed within each condition and checkpoints are written.
echo If the process is interrupted, rerun this same BAT to reuse completed
echo deterministic condition checkpoints; the attempt ledger will record it.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
  set "PY=py"
) else (
  set "PY=python"
)

%PY% -c "import torch,sklearn,pandas,numpy,networkx" >nul 2>&1
if errorlevel 1 (
  echo Installing required packages...
  %PY% -m pip install torch scikit-learn pandas numpy networkx
  if errorlevel 1 (
    echo ERROR: dependency installation failed.
    pause
    exit /b 1
  )
)

%PY% -u "%~dp001_run_final_external_2019.py" ^
  --model-ready "%MR%" ^
  --data "%DATA%" ^
  --train-freeze "%TRAIN%" ^
  --eval-freeze "%FREEZE%" ^
  --out "%OUT%"

if errorlevel 1 (
  echo.
  echo FINAL EXTERNAL EVALUATION STOPPED OR FAILED.
  echo Do not change any scientific setting.
  echo If this was only an interruption, rerun the same BAT; completed
  echo deterministic checkpoints will be reused and the ledger will record it.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo FINAL EXTERNAL EVALUATION COMPLETE
echo ============================================================
echo.
echo Send:
echo %OUT%\BattLeDIM_final_external_2019_summary_v0_17.txt
echo.
pause
