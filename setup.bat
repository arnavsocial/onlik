@echo off
REM ============================================
REM  One-Time Setup — Install all dependencies
REM ============================================

echo.
echo ===================================
echo  Offline AI Video Editor — Setup
echo ===================================
echo.

REM Verify Python 3.10 exists
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 is required but not found. Please install Python 3.10 to continue.
    pause
    exit /b 1
)

echo [1/6] Creating Python 3.10 virtual environment...
py -3.10 -m venv venv

echo [2/6] Upgrading pip...
.\venv\Scripts\python.exe -m pip install --upgrade pip

echo [3/6] Installing PyTorch (CUDA 12.1)...
.\venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [WARN] CUDA build failed, attempting fallback to CPU-only torch...
    .\venv\Scripts\python.exe -m pip install torch
)

echo [4/6] Installing WhisperX and DeepFilterNet...
.\venv\Scripts\python.exe -m pip install whisperx deepfilternet

echo [5/6] Installing GUI framework and media toolkit...
.\venv\Scripts\python.exe -m pip install PySide6 ffmpeg-python

echo [6/6] Installing PyInstaller (for packaging)...
.\venv\Scripts\python.exe -m pip install pyinstaller

echo.
echo ===================================
echo  Setup complete!
echo  Run 'run.bat' to launch the app.
echo  Run 'build.bat' to create .exe.
echo ===================================
pause
