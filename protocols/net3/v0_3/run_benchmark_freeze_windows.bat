@echo off
setlocal

set "DATA=%USERPROFILE%\Downloads\AQUA_VERA_data"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_benchmark_frozen"

echo ============================================================
echo AQUA-VERA CONTROLLED BENCHMARK FREEZE v0.3
echo ============================================================
echo.
echo This freezes the full Net3 benchmark BEFORE data generation.
echo No model training is performed.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% -c "import wntr, pandas, numpy, networkx" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    %PY% -m pip install wntr==1.5.0 pandas numpy networkx
    if errorlevel 1 (
        echo ERROR: Could not install dependencies.
        pause
        exit /b 1
    )
)

%PY% "%~dp002_freeze_benchmark_design.py" ^
  --net3 "%DATA%\WNTR_Net3\Net3.inp" ^
  --protocol "%~dp001_benchmark_protocol_v0_3.md" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo BENCHMARK FREEZE FAILED.
    echo Do not generate scenarios or train a model.
    pause
    exit /b 1
)

echo.
echo SUCCESS.
echo Output:
echo %OUT%
echo.
echo Retain benchmark_summary.txt with the project record.
pause
