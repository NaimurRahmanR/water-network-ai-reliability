@echo off
setlocal

set "MODEL=%USERPROFILE%\Downloads\AQUA_VERA_model_ready"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_protocol_frozen"

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

echo ============================================================
echo AQUA-VERA PROTOCOL FREEZE v0.1
echo ============================================================
echo.
echo This verifies the model-ready dataset hashes and freezes the
echo protocol BEFORE model training and BEFORE 2019 evaluation.
echo.

%PY% "%~dp003_freeze_protocol.py" ^
  --model-ready "%MODEL%" ^
  --protocol "%~dp001_preregistration.md" ^
  --out "%OUT%"

if errorlevel 1 (
    echo.
    echo FREEZE FAILED. Do not train anything.
    pause
    exit /b 1
)

copy /Y "%~dp001_preregistration.md" "%OUT%\01_preregistration.md" >nul
copy /Y "%~dp002_benchmark_config.json" "%OUT%\02_benchmark_config.json" >nul

echo.
echo SUCCESS.
echo Frozen protocol:
echo %OUT%
echo.
echo Next step: build the 2018 event-group split manifest.
pause
