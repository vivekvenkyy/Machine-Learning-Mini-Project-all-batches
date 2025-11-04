@echo off
REM ============================================================================
REM Streamlit App Launcher for HistGradientBoostingRegressor Demo
REM ============================================================================

echo.
echo ========================================
echo  HistGradientBoosting Regressor Demo
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [1/3] Checking Python installation...
python --version
echo.

REM Check if requirements are installed
echo [2/3] Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    echo This may take a few minutes...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies
        echo Please run manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
) else (
    echo Dependencies are installed.
)
echo.

REM Launch the app
echo [3/3] Launching Streamlit app...
echo.
echo ========================================
echo  Opening browser at http://localhost:8501
echo  Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run app.py

pause
