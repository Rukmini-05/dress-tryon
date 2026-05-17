@echo off
color 0A
echo =========================================
echo    Initializing StyleAR Virtual Try-On...
echo =========================================
echo.

:: Activate the virtual environment
call venv\Scripts\activate

:: Open the default web browser after a 2-second delay
timeout /t 2 /nobreak > NUL
start http://127.0.0.1:5000

:: Start the Python backend
python app.py

pause

