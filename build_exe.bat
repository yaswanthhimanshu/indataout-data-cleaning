@echo off
REM build_exe.bat - builds dist\inDataOut.exe
REM Run this from inside your project folder, with .venv already active.

echo Installing/updating build dependencies...
set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" -m pip install pyinstaller pywebview >nul
if errorlevel 1 (
    echo Failed to install pyinstaller/pywebview. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo Building inDataOut.exe (this can take a minute or two)...
"%PYTHON_EXE%" -m PyInstaller --noconfirm indataout.spec
if errorlevel 1 (
    echo.
    echo Build failed - scroll up for the PyInstaller error.
    pause
    exit /b 1
)

echo.
echo Copying required files next to the exe...
copy /Y .env.example dist\.env.example >nul
if not exist dist\.env (
    if exist .env (
        copy /Y .env dist\.env >nul
        echo   .env copied
    ) else (
        echo   NOTE: no .env found - copy .env.example to dist\.env and fill in your
        echo   MySQL password before running inDataOut.exe.
    )
)

echo.
echo Done. Your app is at: dist\inDataOut.exe
echo Double-click it to launch inDataOut in its own window.
pause
