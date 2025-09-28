@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Launch Streamlit dashboard (Windows)
REM Usage: double-click or run: scripts\run_streamlit.bat [PORT] [ADDRESS]
REM Defaults: PORT=8501, ADDRESS=localhost

set PORT=%1
if "%PORT%"=="" set PORT=8501
set ADDRESS=%2
if "%ADDRESS%"=="" set ADDRESS=localhost

REM Resolve project paths
set SCRIPT_DIR=%~dp0
set PY_SCRIPT=%SCRIPT_DIR%run_streamlit.py

REM Prefer venv python if available
set PYTHON=python
if exist "%SCRIPT_DIR%..\venv\Scripts\python.exe" set PYTHON="%SCRIPT_DIR%..\venv\Scripts\python.exe"

echo Running: %PYTHON% "%PY_SCRIPT%" --port %PORT% --address %ADDRESS%
%PYTHON% "%PY_SCRIPT%" --port %PORT% --address %ADDRESS%

endlocal
