@echo off
setlocal
echo WARNING: This downloads full BattLeDIM plus the full LeakDB archive.
echo Expected total download is roughly 14.5 GB, before extraction.
echo Existing checksum-valid files are skipped and partial files resume.
echo.
choice /C YN /M "Continue with the complete AQUA-VERA data download"
if errorlevel 2 exit /b 0
python "%~dp0download_aqua_vera_data.py" --output "%USERPROFILE%\Downloads\AQUA_VERA_data" --everything
set EXITCODE=%ERRORLEVEL%
echo.
if %EXITCODE%==0 (
  echo COMPLETE data download and checksum verification completed successfully.
) else (
  echo Downloader stopped with exit code %EXITCODE%.
  echo Run this file again; partial .part downloads will resume.
)
pause
exit /b %EXITCODE%
