@echo off
title Earnings Analyzer - Development Environment
color 0E

echo ========================================
echo  EARNINGS ANALYZER - DEV ENVIRONMENT
echo ========================================
echo.

cd /d "C:\Users\rahul\OneDrive\IMP_DOCS\AI\EarningsAnalyzer"

echo [1/3] Syncing with GitHub...
git pull origin main
if %errorlevel% neq 0 (
    echo WARNING: Could not sync with GitHub. Continuing anyway...
)
echo.

echo [2/3] Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python first.
    pause
    exit /b 1
)
echo.

echo [3/3] Starting Flask development server...
echo.
echo ========================================
echo   Server will start at: http://localhost:5000
echo   
echo   Press Ctrl+C to stop the server
echo   Close this window to stop the server
echo ========================================
echo.

python app.py