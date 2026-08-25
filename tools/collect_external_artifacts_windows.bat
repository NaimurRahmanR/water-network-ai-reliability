@echo off
setlocal
where py >nul 2>&1
if %errorlevel%==0 (set "PY=py") else (set "PY=python")
%PY% -u "%~dp0collect_external_artifacts.py"
if errorlevel 1 (
  echo.
  echo COLLECTION FAILED.
  pause
  exit /b 1
)
echo.
echo Upload the generated AQUA_VERA_external_exact_artifacts_for_github.zip.
pause
