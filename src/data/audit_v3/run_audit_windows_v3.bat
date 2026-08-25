@echo off
setlocal
set "DATA=%USERPROFILE%\Downloads\AQUA_VERA_data"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_audit_v3"

echo ============================================================
echo AQUA-VERA DATA AUDIT v3
echo ============================================================
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

%PY% -c "import pandas, yaml" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    %PY% -m pip install pandas pyyaml
    if errorlevel 1 (
        echo ERROR: Could not install pandas / pyyaml.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_audit_dataset_v3.py" --data "%DATA%" --out "%OUT%"

echo.
echo Output:
echo %OUT%\audit_summary_v3.txt
echo.
pause
