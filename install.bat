@echo off
setlocal EnableDelayedExpansion

pushd "%~dp0"

set "VENV_DIR=%~dp0.venv"

echo Looking for Python...

set "PYTHON="

where python >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('where python') do (
        if not defined PYTHON (
            "%%P" -c "import sys; v=sys.version_info; exit(0 if v>=(3,10) else 1)" >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON=%%P"
            )
        )
    )
)

if not defined PYTHON (
    where python3 >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('where python3') do (
            if not defined PYTHON (
                "%%P" -c "import sys; v=sys.version_info; exit(0 if v>=(3,10) else 1)" >nul 2>&1
                if not errorlevel 1 (
                    set "PYTHON=%%P"
                )
            )
        )
    )
)

if not defined PYTHON (
    echo Error: Python 3.10 or newer is required but was not found in PATH.
    echo Download Python from https://www.python.org/downloads/windows/
    goto :end_error
)

echo Using Python: !PYTHON!

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment at "%VENV_DIR%"...
    "!PYTHON!" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Error: Failed to create virtual environment.
        goto :end_error
    )
) else (
    echo Virtual environment already exists.
)

echo Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Error: pip upgrade failed.
    goto :end_error
)

echo Installing dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Dependency installation failed.
    goto :end_error
)

echo.
echo Install complete. Run run.bat to start ImageTagger.
goto :end

:end_error
echo.
echo Installation failed.
popd
pause
exit /b 1

:end
popd
pause
exit /b 0
