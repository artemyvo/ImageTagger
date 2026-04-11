@echo off
setlocal

pushd "%~dp0"

set "VENV_DIR=%~dp0.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Error: Virtual environment not found. Please run install.bat first.
    goto :end_error
)

where git >nul 2>&1
if not errorlevel 1 (
    echo Pulling latest changes...
    git pull
    if errorlevel 1 (
        echo Warning: git pull failed. Continuing with dependency update.
    )
) else (
    echo git not found in PATH, skipping repository update.
)

echo Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Error: pip upgrade failed.
    goto :end_error
)

echo Updating dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo Error: Dependency update failed.
    goto :end_error
)

echo.
echo Update complete.
goto :end

:end_error
echo.
echo Update failed.
popd
pause
exit /b 1

:end
popd
pause
exit /b 0
