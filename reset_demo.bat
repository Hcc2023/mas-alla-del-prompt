@echo off
REM Reinicia la demo con un clic: estado inicial + prueba de humo + abre la app.
REM Cierra la app (Ctrl+C en su terminal) antes de ejecutarlo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
python reset_demo.py
if errorlevel 1 (
    echo.
    echo La prueba de humo fallo. No se abre la app.
    pause
    exit /b 1
)
streamlit run app.py
