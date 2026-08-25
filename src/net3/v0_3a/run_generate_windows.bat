@echo off
setlocal

set "DATA=%USERPROFILE%\Downloads\AQUA_VERA_data"
set "FROZEN=%USERPROFILE%\Downloads\AQUA_VERA_benchmark_frozen"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_generated_v0_3a"

echo ============================================================
echo AQUA-VERA FULL SCENARIO GENERATION v0.3a
echo ============================================================
echo.
echo Frozen benchmark:
echo %FROZEN%
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

%PY% -c "import wntr, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    %PY% -m pip install wntr==1.5.0 pandas numpy
    if errorlevel 1 (
        echo ERROR: Could not install dependencies.
        pause
        exit /b 1
    )
)

%PY% "%~dp001_generate_scenarios.py" ^
  --net3 "%DATA%\WNTR_Net3\Net3.inp" ^
  --frozen "%FROZEN%" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo GENERATION FAILED.
    echo Existing verified context files are preserved and a rerun will resume.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo GENERATION SUCCESS
echo ============================================================
echo.
echo Send:
echo %OUT%\generation_summary.txt
echo.
pause
