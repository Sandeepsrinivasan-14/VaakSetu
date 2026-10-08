@echo off
cd /d C:\Users\sndps\Projects\VaakSetu
set PYTHONUNBUFFERED=1
"C:\Users\sndps\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\run_pipeline.py --stage all > results\pipeline.log 2> results\pipeline.err.log
