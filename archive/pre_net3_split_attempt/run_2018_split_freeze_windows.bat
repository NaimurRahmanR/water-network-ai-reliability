@echo off
setlocal

set "MODEL=%USERPROFILE%\Downloads\AQUA_VERA_model_ready"
set "PROTOCOL=%USERPROFILE%\Downloads\AQUA_VERA_protocol_frozen\FROZEN_PROTOCOL_RECORD.json"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_2018_split_frozen"

echo ============================================================
echo AQUA-VERA 2018 SPLIT FREEZE v0.1
echo ============================================================
echo.
echo This reads 2018 ONLY.
echo 2019 remains untouched.
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% -c "import pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing pandas and numpy...
    %PY% -m pip install pandas numpy
    if errorlevel 1 (
        echo ERROR: Could not install dependencies.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_build_2018_split_manifest.py" ^
  --model-ready "%MODEL%" ^
  --protocol-record "%PROTOCOL%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo SPLIT FREEZE FAILED.
    echo Do not train a model.
    pause
    exit /b 1
)

echo.
echo SUCCESS.
echo Output:
echo %OUT%
echo.
echo Send split_summary.txt to the research record.
pause
