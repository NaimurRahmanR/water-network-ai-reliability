@echo off
setlocal

set "WIN=%USERPROFILE%\Downloads\AQUA_VERA_windows_frozen_v0_4"
set "GEN=%USERPROFILE%\Downloads\AQUA_VERA_generated_v0_3a"
set "BASE=%USERPROFILE%\Downloads\AQUA_VERA_baseline_mlp_v0_5"
set "GRAPH=%USERPROFILE%\Downloads\AQUA_VERA_graph_gcn_gru_v0_6"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_heldout_test_v0_7b"

echo ============================================================
echo AQUA-VERA HELD-OUT TEST EVALUATION v0.7b
echo ============================================================
echo.
echo v0.7 failed before prediction because the evaluator class names
echo did not match the frozen GCN-GRU checkpoint.
echo.
echo v0.7b verifies BOTH checkpoint classes BEFORE loading test.
echo Frozen models and thresholds are unchanged.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% -c "import torch, sklearn, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    %PY% -m pip install torch scikit-learn pandas numpy
    if errorlevel 1 (
        echo ERROR: dependency installation failed.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_evaluate_heldout_test_v0_7b.py" ^
  --windows "%WIN%" ^
  --generated "%GEN%" ^
  --baseline "%BASE%" ^
  --graph "%GRAPH%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo EVALUATION FAILED.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS
echo ============================================================
echo.
echo Send:
echo %OUT%\heldout_test_summary_v0_7b.txt
echo.
pause
