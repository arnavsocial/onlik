@echo off
REM ============================================
REM  Build App Bundle — Offline AI Video Editor
REM ============================================

echo.
echo Building OfflineAIVideoEditor bundle...
echo This may take several minutes.
echo.

if not exist "venv\Scripts\pyinstaller.exe" (
    echo [ERROR] PyInstaller not found. Please run setup.bat first.
    pause
    exit /b 1
)

.\venv\Scripts\pyinstaller.exe build.spec --distpath dist --workpath build_temp --clean

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo ===================================
echo  Build complete!
echo  Output: dist\OfflineAIVideoEditor\OfflineAIVideoEditor.exe
echo ===================================
pause
