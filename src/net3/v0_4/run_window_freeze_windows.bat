@echo off
setlocal

set "GEN=%USERPROFILE%\Downloads\AQUA_VERA_generated_v0_3a"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_windows_frozen_v0_4"

echo ============================================================
echo AQUA-VERA WINDOW + NORMALIZATION FREEZE v0.4
echo ============================================================
echo.
echo Generated data:
echo %GEN%
echo.
echo Output:
echo %OUT%
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% -c "import numpy, pandas" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    %PY% -m pip install numpy pandas
    if errorlevel 1 (
        echo ERROR: Could not install dependencies.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_freeze_windows_and_normalization.py" ^
  --generated "%GEN%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo WINDOW FREEZE FAILED.
    echo Do not train a model.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS
echo ============================================================
echo.
echo Send:
echo %OUT%\window_summary.txt
echo.
pause
