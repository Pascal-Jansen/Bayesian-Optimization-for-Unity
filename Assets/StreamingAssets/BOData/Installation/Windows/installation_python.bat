@echo off

set PYTHON_EXE="C:\Program Files\Python311\python.exe"

cd %~dp0

set PYTHON_INSTALLER="Installation_Objects\python-3.11.3.exe"
set REQUIREMENTS="\..\requirements.txt"
set VC_REDIST_EXE="Installation_Objects\VC_redist.x64.exe"

REM Check if Visual C++ Redistributable is already installed
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel%==0 (
    echo Visual C++ Redistributable is already installed.
    goto continue
)

REM Run the Visual C++ Redistributable installer
echo Installing Visual C++ Redistributable...
%VC_REDIST_EXE% /quiet

REM Check if the installation was successful
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel%==0 (
    echo Visual C++ Redistributable was successfully installed.
) else (
    echo Error installing Visual C++ Redistributable.
    exit /b 1
)

:continue
REM Check if Python is already installed
%PYTHON_EXE% --version >nul 2>&1
if %errorlevel%==0 (
    echo Python is already installed.
    goto install_packages
)

REM Run the Python installer
echo Installing Python...
%PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1

REM Check if the installation was successful
%PYTHON_EXE% --version >nul 2>&1
if %errorlevel%==0 (
    echo Python was successfully installed.
) else (
    echo Error installing Python.
    exit /b 1
)


:install_packages
REM Update Pip
echo Updating Pip...
%PYTHON_EXE% -m pip install --upgrade pip

REM Install packages
echo Installing packages...
%PYTHON_EXE% -m pip install -r %REQUIREMENTS%

REM Check if the package installation was successful
%PYTHON_EXE% -m pip list | findstr "numpy scipy matplotlib pandas torch pytorch gpytorch botorch" >nul 2>&1
if %errorlevel%==0 (
    echo Packages were successfully installed.
) else (
    echo Error installing packages.
    exit /b 1
)
