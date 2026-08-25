@echo off
setlocal
python "%~dp0download_aqua_vera_data.py" --output "%USERPROFILE%\Downloads\AQUA_VERA_data" --full-battledim
set EXITCODE=%ERRORLEVEL%
echo.
if %EXITCODE%==0 (
  echo FULL BattLeDIM download and checksum verification completed successfully.
  echo This includes all 17 files in Zenodo record 4017659.
) else (
  echo Downloader stopped with exit code %EXITCODE%.
  echo Run this file again; partial .part downloads will resume.
)
pause
exit /b %EXITCODE%
