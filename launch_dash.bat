@echo off
cd /d "%~dp0"

if not exist .venv (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

REM Check that dash and dash-bootstrap-components are installed
python -c "import dash, dash_bootstrap_components" 2>nul
if errorlevel 1 (
    echo Installing Dash dependencies...
    pip install -r requirements_dash.txt
    if errorlevel 1 (
        echo ERROR: Failed to install Dash dependencies.
        pause
        exit /b 1
    )
)

echo Starting Plottle Dash app on http://127.0.0.1:8050
python dash_app/app.py
pause
