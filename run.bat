@echo off
setlocal

pushd "%~dp0"

set "VENV_DIR=%~dp0.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Error: Virtual environment not found. Please run install.bat first.
    goto :end_error
)

"%VENV_DIR%\Scripts\python.exe" run.py
if errorlevel 1 (
    echo Error: ImageTagger exited with an error.
    goto :end_error
)

goto :end

:end_error
popd
pause
exit /b 1

:end
popd
exit /b 0
