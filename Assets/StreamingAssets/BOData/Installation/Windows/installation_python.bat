@echo off
setlocal enabledelayedexpansion

set PYTHON_EXE="C:\Program Files\Python313\python.exe"

cd /d %~dp0

set PYTHON_INSTALLER="Installation_Objects\python-3.13.7.exe"
set REQUIREMENTS="..\requirements.txt"
set VC_REDIST_EXE="Installation_Objects\VC_redist.x64.exe"

REM Check if Visual C++ Redistributable is already installed
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel%==0 (
    echo Visual C++ Redistributable is already installed.
    goto check_python
)

REM Check if Visual C++ Redistributable is already installed
REM Check for VS 2015-2022 (v14.0 or later)
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel%==0 (
    echo Visual C++ Redistributable is already installed.
    goto check_python
)

reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel%==0 (
    echo Visual C++ Redistributable is already installed.
    goto check_python
)


REM Run the Visual C++ Redistributable installer
echo Installing Visual C++ Redistributable...
%VC_REDIST_EXE% /quiet /norestart
if %errorlevel% neq 0 (
    echo Warning: Visual C++ Redistributable installation returned error code %errorlevel%
    echo Continuing anyway...
)


REM Check if the installation was successful
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel%==0 (
    echo Visual C++ Redistributable was successfully installed.
) else (
    echo Error installing Visual C++ Redistributable.
    exit /b 1
)


:check_python
REM Check if Python is already installed
%PYTHON_EXE% --version >nul 2>&1
if %errorlevel%==0 (
    echo Python is already installed.
    goto install_packages
)

REM Run the Python installer
echo Installing Python...
%PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1

REM Wait a moment for installation to complete
timeout /t 5 /nobreak >nul

REM Check if the installation was successful
%PYTHON_EXE% --version >nul 2>&1
if %errorlevel%==0 (
    echo Python was successfully installed.
) else (
    echo Error installing Python.
    exit /b 1
)


:install_packages
REM Check if requirements file exists
if not exist %REQUIREMENTS% (
    echo Error: Requirements file not found at %REQUIREMENTS%
    pause
    exit /b 1
)


REM Update Pip
echo Updating Pip...
%PYTHON_EXE% -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Warning: Pip upgrade failed, continuing with existing version...
)

REM Install packages
echo Installing packages from %REQUIREMENTS%...
%PYTHON_EXE% -m pip install -r %REQUIREMENTS%
if %errorlevel% neq 0 (
    echo Error: Package installation failed.
    pause
    exit /b 1
)


echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
pause
exit /b 0
