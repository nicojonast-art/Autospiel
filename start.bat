@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Car Match Bot

rem --- Python pruefen, sonst automatisch installieren (winget) ---
python -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python fehlt - wird jetzt installiert ...
  winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
  set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
)
python -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python konnte nicht installiert werden. Bitte von python.org installieren
  echo ^(Haken bei "Add python.exe to PATH"^) und start.bat erneut starten.
  pause
  exit /b 1
)

rem --- Pakete (einmalig) ---
python -m pip install -q --disable-pip-version-check -r requirements.txt

rem --- Bot starten ---
python bot.py %*
pause
