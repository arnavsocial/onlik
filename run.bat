@echo off
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

set "PATH=%~dp0bin\ffmpeg;%PATH%"
.\venv\Scripts\python.exe app.py
