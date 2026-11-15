@echo off
rem Launch the training web server (the Windows counterpart of run_server.sh).
rem
rem Settings come from .env next to this file (COMFY_DIR, VENV_PYTHON) or from real environment
rem variables, which win. The interpreter is chosen like run_server.sh does:
rem   1. VENV_PYTHON, if set and it exists
rem   2. venv\Scripts\python.exe in this folder (what the setup page creates)
rem   3. ..\venv\Scripts\python.exe (ComfyUI / venv / this project as sibling folders)
rem   4. "python" from PATH
rem Arguments are passed on: run_server.bat --port 8800

setlocal EnableExtensions
set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
cd /d "%HERE%"

if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not defined %%A set "%%A=%%~B"
    )
)

set "PYTHON="
if defined VENV_PYTHON if exist "%VENV_PYTHON%" set "PYTHON=%VENV_PYTHON%"
if not defined PYTHON if exist "%HERE%\venv\Scripts\python.exe" set "PYTHON=%HERE%\venv\Scripts\python.exe"
if not defined PYTHON if exist "%HERE%\..\venv\Scripts\python.exe" set "PYTHON=%HERE%\..\venv\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"

rem Installs the server's own packages into a temporary environment if they are missing, then comes back here.
"%PYTHON%" -m backend.first_run %*
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m backend.cli --host 0.0.0.0 %*
exit /b %errorlevel%
