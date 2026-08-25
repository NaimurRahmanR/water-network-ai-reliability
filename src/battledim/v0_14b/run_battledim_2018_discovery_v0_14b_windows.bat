@echo off
setlocal EnableExtensions

set "MR=%USERPROFILE%\Downloads\AQUA_VERA_model_ready"
set "DATA=%USERPROFILE%\Downloads\AQUA_VERA_data"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_BattLeDIM_2018_discovery_v0_14b"

echo ============================================================
echo AQUA-VERA BATTLEDIM 2018 DISCOVERY v0.14b
echo ============================================================
echo.
echo 2018 ONLY.
echo 2019 will not be checked, opened, parsed, or hashed.
echo No model training. No split freeze. No threshold selection.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% -c "import pandas,numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing pandas/numpy...
    %PY% -m pip install pandas numpy
    if errorlevel 1 (
        echo.
        echo ERROR: dependency installation failed.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_battledim_2018_discovery_v0_14b.py" ^
  --model-ready "%MR%" ^
  --data "%DATA%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo DISCOVERY AUDIT FAILED.
    echo The Python traceback above contains the exact cause.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS
echo ============================================================
echo.
echo Send:
echo %OUT%\BattLeDIM_2018_discovery_summary.txt
echo.
pause
