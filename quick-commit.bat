@echo off
title Quick Commit & Push
color 0B

echo ========================================
echo       QUICK COMMIT & PUSH
echo ========================================
echo.

rem Change to the correct directory
echo Navigating to project directory...
cd /d "C:\Users\rahul\OneDrive\IMP_DOCS\AI\EarningsAnalyzer"

rem Check if we're in a git repository
if not exist ".git" (
    echo ERROR: Not in a Git repository. Make sure you're in the correct folder.
    echo Current directory: %cd%
    pause
    exit /b 1
)

echo Current directory: %cd%
echo.

echo Checking for changes...
git status --porcelain > temp_status.txt
if %errorlevel% neq 0 (
    echo ERROR: Git status command failed. Is Git installed?
    del temp_status.txt 2>nul
    pause
    exit /b 1
)

rem Check if there are any changes
for /f %%i in (temp_status.txt) do set has_changes=1
del temp_status.txt

if not defined has_changes (
    echo No changes to commit. Repository is up to date.
    pause
    exit /b 0
)

echo Current changes:
git status --short
echo.

rem Get commit message
if "%~1"=="" (
    set /p "commit_msg=Enter commit message: "
) else (
    set "commit_msg=%*"
)

rem Check if commit message is empty
if "%commit_msg%"=="" (
    echo No commit message provided. Cancelling.
    pause
    exit /b 1
)

echo.
echo Adding all changes...
git add .
if %errorlevel% neq 0 (
    echo ERROR: Failed to add files to Git.
    pause
    exit /b 1
)

echo Committing with message: "%commit_msg%"
git commit -m "%commit_msg%"
if %errorlevel% neq 0 (
    echo ERROR: Commit failed.
    pause
    exit /b 1
)

echo Pushing to GitHub...
git push origin main
if %errorlevel% neq 0 (
    echo ERROR: Push failed. Possible reasons:
    echo - No internet connection
    echo - Authentication issues
    echo - Remote repository not set up
    echo.
    echo Try running: git remote -v
    pause
    exit /b 1
)

echo.
echo ✅ Successfully committed and pushed to GitHub!
echo Commit message: "%commit_msg%"
echo.
pause