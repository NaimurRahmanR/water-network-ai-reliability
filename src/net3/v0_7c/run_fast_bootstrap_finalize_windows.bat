@echo off
setlocal

set "PRED=%USERPROFILE%\Downloads\AQUA_VERA_heldout_test_v0_7b\heldout_test_predictions.csv"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_heldout_test_v0_7c"

echo ============================================================
echo AQUA-VERA FAST HELD-OUT BOOTSTRAP FINALIZER v0.7c
echo ============================================================
echo.
echo This does NOT rerun either model.
echo It reuses the frozen predictions already produced by v0.7b.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% -c "import sklearn,pandas,numpy" >nul 2>&1
if errorlevel 1 (
    %PY% -m pip install scikit-learn pandas numpy
    if errorlevel 1 (
        echo ERROR: dependency installation failed.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_fast_bootstrap_finalize.py" ^
  --predictions "%PRED%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo FINALIZER FAILED.
    pause
    exit /b 1
)

echo.
echo SUCCESS.
echo Send:
echo %OUT%\heldout_test_summary_v0_7c.txt
echo.
pause
