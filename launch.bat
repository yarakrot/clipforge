@echo off
chcp 65001 >nul
title ClipForge
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0launch.py"
) else (
    python "%~dp0launch.py"
)
echo.
pause
