@echo off
title Viral Clipper
cd /d "%~dp0"
if not exist venv (echo Run install.bat first. & pause & exit /b 1)
call venv\Scripts\activate.bat
python app.py
pause
