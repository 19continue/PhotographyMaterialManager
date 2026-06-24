@echo off
chcp 65001 > nul
title PMM Stop
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONNOUSERSITE=1
set PYTHONPATH=

echo.
echo Stopping Photography Material Manager...
echo.

if not exist "python\python.exe" (
  echo Built-in Python was not found: python\python.exe
  echo Please extract the whole package before running this file.
  pause
  exit /b 1
)

"python\python.exe" -s -u "stop.py"
pause
