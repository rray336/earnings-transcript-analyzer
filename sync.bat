@echo off
title Earnings Analyzer - Git Sync
color 0A

echo ========================================
echo    EARNINGS ANALYZER - GIT SYNC
echo ========================================
echo.

cd /d "C:\Users\rahul\OneDrive\IMP_DOCS\AI\EarningsAnalyzer"

echo [1/4] Checking current status...
git status
echo.

echo [2/4] Pulling latest changes from GitHub...
git pull origin main
if %errorlevel% neq 0 (
    echo ERROR: Failed to pull changes. Check your internet connection.
    pause
    exit /b 1
)
echo.

echo [3/4] Current repository status:
git status
echo.

echo [4/4] Ready for development!
echo.
echo ========================================
echo Available commands:
echo   - Type 'commit' to commit and push changes
echo   - Type 'status' to check git status
echo   - Type 'log' to see recent commits
echo   - Press any key to continue to development
echo ========================================
echo.

set /p choice="Enter command or press Enter to continue: "

if /i "%choice%"=="commit" (
    call :commit_changes
) else if /i "%choice%"=="status" (
    git status
    pause
) else if /i "%choice%"=="log" (
    git log --oneline -10
    pause
) else (
    echo Ready for development!
)

goto :end

:commit_changes
echo.
echo ========================================
echo        COMMITTING CHANGES
echo ========================================
echo.

git status
echo.

set /p commit_msg="Enter commit message (or press Enter to cancel): "
if "%commit_msg%"=="" (
    echo Commit cancelled.
    goto :end
)

echo Adding all changes...
git add .

echo Committing with message: "%commit_msg%"
git commit -m "%commit_msg%"

if %errorlevel% neq 0 (
    echo ERROR: Commit failed.
    pause
    goto :end
)

echo Pushing to GitHub...
git push origin main

if %errorlevel% neq 0 (
    echo ERROR: Push failed. Check your internet connection.
    pause
    goto :end
)

echo.
echo ✅ Successfully synced with GitHub!
echo.

:end
pause