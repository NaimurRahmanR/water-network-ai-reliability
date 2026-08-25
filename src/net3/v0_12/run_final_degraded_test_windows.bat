@echo off
setlocal
set "WIN=%USERPROFILE%\Downloads\AQUA_VERA_windows_frozen_v0_4"
set "GEN=%USERPROFILE%\Downloads\AQUA_VERA_generated_v0_3a"
set "BASE=%USERPROFILE%\Downloads\AQUA_VERA_baseline_mlp_v0_5"
set "GRAPH=%USERPROFILE%\Downloads\AQUA_VERA_graph_gcn_gru_v0_6"
set "CLEAN=%USERPROFILE%\Downloads\AQUA_VERA_heldout_test_v0_7b"
set "CALDEG=%USERPROFILE%\Downloads\AQUA_VERA_calibration_degradations_v0_9b"
set "CALATTR=%USERPROFILE%\Downloads\AQUA_VERA_calibration_attribution_v0_10b"
set "CTRL=%USERPROFILE%\Downloads\AQUA_VERA_controller_frozen_v0_11"
set "V09=%USERPROFILE%\Downloads\AQUA_VERA_Calibration_Degradations_v0_9b\01_generate_calibration_degradations_v0_9b.py"
set "V10=%USERPROFILE%\Downloads\AQUA_VERA_Calibration_Attribution_v0_10b\01_calibration_attribution_v0_10b.py"
set "V11=%USERPROFILE%\Downloads\AQUA_VERA_Controller_Freeze_v0_11\01_freeze_controller.py"
set "OUT=%USERPROFILE%\Downloads\AQUA_VERA_final_degraded_test_v0_12"

echo ============================================================
echo AQUA-VERA FINAL DEGRADED HELD-OUT TEST v0.12
echo ============================================================
echo.
echo FINAL frozen evaluation. Do not change settings after this run.
echo Full prediction metrics use all 2,700 windows per condition.
echo Attribution/controller metrics use the frozen 600-window subset.
echo.
where py >nul 2>&1
if %errorlevel%==0 (set "PY=py") else (set "PY=python")
%PY% -c "import torch,sklearn,pandas,numpy,networkx" >nul 2>&1
if errorlevel 1 (
  %PY% -m pip install torch scikit-learn pandas numpy networkx
  if errorlevel 1 (echo ERROR: dependency installation failed.&pause&exit /b 1)
)
%PY% "%~dp001_final_degraded_test_v0_12.py" ^
 --windows "%WIN%" --generated "%GEN%" --baseline "%BASE%" --graph "%GRAPH%" ^
 --clean-test "%CLEAN%" --calibration-degradations "%CALDEG%" ^
 --calibration-attribution "%CALATTR%" --controller "%CTRL%" ^
 --v09-script "%V09%" --v10-script "%V10%" --v11-script "%V11%" --out "%OUT%"
if errorlevel 1 (echo.&echo FINAL RUN FAILED. Do not change scientific settings.&pause&exit /b 1)
echo.
echo FINAL DEGRADED TEST COMPLETE
echo Send: %OUT%\final_degraded_test_summary.txt
pause
