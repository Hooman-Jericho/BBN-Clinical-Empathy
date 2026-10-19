@echo off
REM BBN Annotation Tool - Windows Batch Launcher
title BBN Clinical Empathy Annotation Tool
echo ========================================================
echo   Launching BBN Clinical Empathy Annotation Tool...
echo ========================================================
echo.

cd /d "%~dp0"

REM Check for virtual environment or system Python
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run Streamlit application
streamlit run app\main.py --server.port 8501
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Streamlit exited with an error or is not installed in the active environment.
    echo Trying with python -m streamlit...
    python -m streamlit run app\main.py --server.port 8501
)
pause
