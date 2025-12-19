@echo off
TITLE Object Detection on Snapdragon X Elite NPU
echo ====================================================
echo      Snapdragon X Elite NPU Demo Launcher
echo ====================================================

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org (Tick 'Add to PATH')
    pause
    exit /b
)

echo [INFO] Installing/Verifying Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)

echo.
echo [INFO] Starting NPU Performance Test...
echo        (Keep an eye on Task Manager -> Performance -> NPU)
echo.

python npu_realtime_inference.py --duration 30

echo.
echo [DONE] Demo finished.
pause
