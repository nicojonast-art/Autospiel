@echo off
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python bot.py %*
pause
