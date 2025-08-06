@echo off
title Git Environment Check
color 0C

echo ========================================
echo       GIT ENVIRONMENT CHECK
echo ========================================
echo.

echo [1/5] Checking if Git is installed...
git --version
if %errorlevel% neq 0 (
    echo ❌ Git is NOT installed or not in PATH
    echo.
    echo SOLUTION: Install Git for Windows
    echo 1. Go to: https://git-scm.com/download/win
    echo 2. Download and install Git for Windows
    echo 3. During installation, make sure to select "Git from the command line and also from 3rd-party software"
    echo 4. Restart your computer after installation
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Git is installed and working
)
echo.

echo [2/5] Checking current directory...
cd /d "C:\Users\rahul\OneDrive\IMP_DOCS\AI\EarningsAnalyzer"
echo Current directory: %cd%
echo.

echo [3/5] Checking if this is a Git repository...
if not exist ".git" (
    echo ❌ This is NOT a Git repository
    echo.
    echo SOLUTION: Initialize Git repository
    echo Run these commands:
    echo   git init
    echo   git remote add origin YOUR_GITHUB_URL
    echo.
    pause
    exit /b 1
) else (
    echo ✅ This is a Git repository
)
echo.

echo [4/5] Checking Git configuration...
git config user.name
if %errorlevel% neq 0 (
    echo ❌ Git user.name not configured
    echo Run: git config user.name "Your Name"
) else (
    echo ✅ Git user.name: 
    git config user.name
)

git config user.email
if %errorlevel% neq 0 (
    echo ❌ Git user.email not configured  
    echo Run: git config user.email "your@email.com"
) else (
    echo ✅ Git user.email:
    git config user.email
)
echo.

echo [5/5] Checking remote repository...
git remote -v
if %errorlevel% neq 0 (
    echo ❌ No remote repository configured
    echo Run: git remote add origin YOUR_GITHUB_URL
) else (
    echo ✅ Remote repository configured
)
echo.

echo ========================================
echo           DIAGNOSIS COMPLETE
echo ========================================
echo.

echo If all checks show ✅, your Git environment is ready!
echo If any show ❌, follow the solutions above.
echo.
pause