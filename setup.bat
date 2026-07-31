@echo off
cd /d "%~dp0"

echo === Plottle Setup ===
echo.

REM ── Check Python is available ────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found on your PATH.
    echo Please install Python 3.9 or later from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM ── Create virtual environment (skip if already exists) ───────────────────────
if exist .venv (
    echo Virtual environment already exists -- skipping creation.
) else (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        echo Make sure Python 3.9 or later is installed.
        echo.
        pause
        exit /b 1
    )
    echo Done.
)

echo.
echo Installing dependencies (this may take a few minutes on first run)...
echo.

call .venv\Scripts\activate.bat
pip install -e ".[formats,nist]"

if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    echo Check the output above for details.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo  Double-click launch.bat to start Plottle.
echo ============================================================
echo.
pause
