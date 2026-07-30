@echo off
setlocal
rem run_maps.ps1 launches create_input_dataset_maps.py with input_dataset_maps.config.json.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_maps.ps1" %*
exit /b %ERRORLEVEL%

