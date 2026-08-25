@echo off
setlocal
set "DATA=%USERPROFILE%\Downloads\AQUA_VERA_data"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_model_ready"

echo ============================================================
echo AQUA-VERA MODEL-READY PREPROCESSOR
echo ============================================================
echo Source:
echo %DATA%
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

%PY% -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo Installing pandas...
    %PY% -m pip install pandas
    if errorlevel 1 (
        echo ERROR: Could not install pandas.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_make_model_ready.py" --data "%DATA%" --out "%OUT%"
if errorlevel 1 (
    echo.
    echo PREPROCESSING FAILED. Do not proceed to modelling.
    pause
    exit /b 1
)

echo.
echo SUCCESS.
echo Send:
echo %OUT%\model_ready_summary.txt
echo.
pause
