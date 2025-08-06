@echo off
title Manual Git Sync (Alternative)
color 0D

echo ========================================
echo     MANUAL GIT SYNC (ALTERNATIVE)
echo ========================================
echo.

cd /d "C:\Users\rahul\OneDrive\IMP_DOCS\AI\EarningsAnalyzer"

echo This script will open a command prompt where you can run Git commands manually.
echo.
echo Common commands you'll need:
echo   git status              - Check current status
echo   git add .               - Add all changes
echo   git commit -m "message" - Commit with message
echo   git push origin main    - Push to GitHub
echo   git pull origin main    - Pull from GitHub
echo.
echo Press any key to open command prompt in this directory...
pause

echo Opening command prompt...
cmd /k "echo Welcome to Git command prompt! && echo Current directory: %cd% && echo. && echo Type 'git status' to start..."