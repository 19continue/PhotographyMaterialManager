@echo off
chcp 65001 > nul
title PMM Start
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONNOUSERSITE=1
set PYTHONPATH=

echo.
echo ================================================
echo Photography Material Manager is starting...
echo App dir: %cd%
echo Log file: %cd%\.pmm_data\logs\startup.log
echo.
echo The browser will open after the local server is ready.
echo First run may download AI dependencies and models.
echo ================================================
echo.

if not exist "python\python.exe" (
  echo Built-in Python was not found: python\python.exe
  echo Please extract the whole package before running this file.
  pause
  exit /b 1
)

"python\python.exe" -s -u "launcher.py"

if errorlevel 1 (
  echo.
  echo Start failed. Please check the log:
  echo %cd%\.pmm_data\logs\startup.log
  echo.
) else (
  echo.
  echo Service stopped.
  echo.
)
pause
