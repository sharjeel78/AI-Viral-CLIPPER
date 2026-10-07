@echo off
title Viral Clipper - install
cd /d "%~dp0"
where python >nul 2>nul || (echo Python not found. Install Python 3.10-3.12 from python.org and tick "Add to PATH". & pause & exit /b 1)
where ffmpeg >nul 2>nul || (echo FFmpeg not found. Run:  winget install Gyan.FFmpeg   then open a NEW terminal and run install.bat again. & pause & exit /b 1)
if not exist venv python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Installed. Double-click run.bat to start.
pause
