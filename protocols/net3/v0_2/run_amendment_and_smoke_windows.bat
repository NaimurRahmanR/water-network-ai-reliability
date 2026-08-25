@echo off
setlocal

set "DATA=%USERPROFILE%\Downloads\AQUA_VERA_data"
set "PROTO=%USERPROFILE%\Downloads\AQUA_VERA_protocol_frozen\FROZEN_PROTOCOL_RECORD.json"
set "AMENDOUT=%USERPROFILE%\Downloads\AQUA_VERA_protocol_amended_v0_2"
set "SMOKEOUT=%USERPROFILE%\Downloads\AQUA_VERA_net3_smoke"

echo ============================================================
echo AQUA-VERA PROTOCOL AMENDMENT v0.2 + NET3 SMOKE TEST
echo ============================================================
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

echo [1/3] Freezing protocol amendment...
%PY% "%~dp002_freeze_amendment.py" ^
  --v01-record "%PROTO%" ^
  --amendment "%~dp001_protocol_amendment_v0_2.md" ^
  --out "%AMENDOUT%"
if errorlevel 1 (
    echo.
    echo AMENDMENT FREEZE FAILED.
    pause
    exit /b 1
)

copy /Y "%~dp001_protocol_amendment_v0_2.md" "%AMENDOUT%\01_protocol_amendment_v0_2.md" >nul

echo.
echo [2/3] Checking WNTR dependencies...
%PY% -c "import wntr, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing wntr, pandas, numpy...
    %PY% -m pip install wntr pandas numpy
    if errorlevel 1 (
        echo ERROR: Could not install WNTR dependencies.
        pause
        exit /b 1
    )
)

echo.
echo [3/3] Running Net3 clean + leak smoke test...
%PY% "%~dp003_net3_smoke_test.py" ^
  --net3 "%DATA%\WNTR_Net3\Net3.inp" ^
  --out "%SMOKEOUT%"
if errorlevel 1 (
    echo.
    echo SMOKE TEST FAILED.
    echo Do not generate the full benchmark yet.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS
echo ============================================================
echo Amendment:
echo %AMENDOUT%
echo.
echo Smoke test:
echo %SMOKEOUT%\smoke_test_summary.txt
echo.
echo Retain smoke_test_summary.txt with the project record.
pause
