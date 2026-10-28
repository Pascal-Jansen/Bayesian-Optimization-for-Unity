@echo off
REM Installs Python 3.13 (all users) and the Visual C++ runtime PyTorch needs, for BOforUnity on Windows.
REM
REM The Python packages from requirements.txt are NOT installed here: on the first Play, Unity creates a
REM private virtual environment (persistentDataPath\BOData\python-venv) and installs them into it (README 8.5).
REM
REM Usage:
REM   installation_python.bat                  install/verify Python 3.13 and the Visual C++ runtime
REM   installation_python.bat --venv DIR       additionally create a virtual environment in DIR and install
REM                                            requirements.txt into it; then set "Manually Installed Python"
REM                                            in Unity to DIR\Scripts\python.exe
REM   installation_python.bat --unattended     no "press any key" prompts (Unity runs it this way)
REM
REM The bundled installers are checked against their SHA-256 before they run with administrator rights.
REM Error levels are read with delayed expansion (exclamation marks): the percent form inside a parenthesised
REM block is expanded when the block is parsed, i.e. before the command in the block has run.

setlocal EnableExtensions DisableDelayedExpansion

REM Before parsing the arguments: SHIFT also shifts parameter 0 (the script path).
set "SCRIPT_DIR=%~dp0"

set "UNATTENDED="
set "VENV_DIR="
REM --venv DIR is made a full path relative to the folder the script was started from, not to the script's folder.
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--unattended" (
    set "UNATTENDED=1"
    shift
    goto parse_args
)
if /i "%~1"=="--venv" (
    if "%~2"=="" (
        echo Error: --venv needs a directory.
        exit /b 2
    )
    for %%I in ("%~2") do set "VENV_DIR=%%~fI"
    shift
    shift
    goto parse_args
)
echo Error: unknown argument %1
exit /b 2
:args_done
pushd "%SCRIPT_DIR%" || exit /b 1

set "TARGET_PY_MAJOR=3"
set "TARGET_PY_MINOR=13"
set "PYTHON_EXE=%ProgramFiles%\Python313\python.exe"
set "PYTHON_INSTALLER=Installation_Objects\python-3.13.7.exe"
set "PYTHON_INSTALLER_SHA256=b12e2e82461ac8e51fc43289050bc8eb937a32d84ce4d242e2c88258c37cf2bb"
set "VC_REDIST_EXE=Installation_Objects\VC_redist.x64.exe"
set "VC_REDIST_SHA256=cc0ff0eb1dc3f5188ae6300faef32bf5beeba4bdd6e8e445a9184072096b713b"
set "REQUIREMENTS=..\requirements.txt"
set "WHEELS_DIR=..\wheels"

setlocal EnableDelayedExpansion

REM ---------------------------------------------------------------- Visual C++ runtime
call :has_vc_runtime
if !errorlevel! equ 0 (
    echo Visual C++ Redistributable is already installed.
    goto check_python
)

call :verify_sha256 "%VC_REDIST_EXE%" %VC_REDIST_SHA256%
if !errorlevel! neq 0 goto fail

echo Installing Visual C++ Redistributable...
"%VC_REDIST_EXE%" /quiet /norestart
set "RC=!errorlevel!"
REM 3010/1641: installed, restart pending; 1638: a newer version is already installed.
if not "!RC!"=="0" if not "!RC!"=="3010" if not "!RC!"=="1641" if not "!RC!"=="1638" (
    echo Warning: the Visual C++ Redistributable installer returned !RC!.
)

call :has_vc_runtime
if !errorlevel! neq 0 (
    echo Error: the Visual C++ Redistributable is still missing after its installer ran ^(exit code !RC!^).
    goto fail
)
echo Visual C++ Redistributable was successfully installed.

REM ---------------------------------------------------------------- Python 3.13
:check_python
if not exist "%PYTHON_EXE%" goto install_python
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(%TARGET_PY_MAJOR%,%TARGET_PY_MINOR%) else 1)" >nul 2>&1
if !errorlevel! equ 0 (
    echo Target Python %TARGET_PY_MAJOR%.%TARGET_PY_MINOR% is already installed: !PYTHON_EXE!
    goto create_venv
)
echo A different Python version is installed at !PYTHON_EXE!. Installing Python %TARGET_PY_MAJOR%.%TARGET_PY_MINOR%...

:install_python
call :verify_sha256 "%PYTHON_INSTALLER%" %PYTHON_INSTALLER_SHA256%
if !errorlevel! neq 0 goto fail

echo Installing Python %TARGET_PY_MAJOR%.%TARGET_PY_MINOR% for all users ^(a UAC prompt may appear^)...
"%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1
set "RC=!errorlevel!"
if "!RC!"=="1602" (
    echo Error: the Python installation was cancelled.
    goto fail
)

REM Give the installer's last steps a moment. ("timeout" fails without a console, e.g. when Unity runs this.)
ping -n 4 127.0.0.1 >nul

"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(%TARGET_PY_MAJOR%,%TARGET_PY_MINOR%) else 1)" >nul 2>&1
if !errorlevel! neq 0 (
    echo Error installing Python %TARGET_PY_MAJOR%.%TARGET_PY_MINOR% ^(installer exit code !RC!^).
    goto fail
)
echo Python %TARGET_PY_MAJOR%.%TARGET_PY_MINOR% was successfully installed.

REM ---------------------------------------------------------------- optional virtual environment
:create_venv
if not defined VENV_DIR (
    echo Unity installs the Python packages into its own environment on the first Play ^(one time, a few minutes^).
    goto done
)

if not exist "%REQUIREMENTS%" (
    echo Error: requirements file not found: !REQUIREMENTS!
    goto fail
)

echo Creating the virtual environment !VENV_DIR!...
"%PYTHON_EXE%" -m venv "!VENV_DIR!"
if !errorlevel! neq 0 (
    echo Error: could not create the virtual environment.
    goto fail
)

REM Offline labs: pip also looks in Installation\wheels (PIP_FIND_LINKS is pip's own setting).
REM Built from CD with delayed expansion, which keeps an exclamation mark in the folder path intact.
if exist "%WHEELS_DIR%\" (
    set "PIP_FIND_LINKS=!CD!\%WHEELS_DIR%"
    echo Using local wheels from !PIP_FIND_LINKS!
)

echo Installing packages from requirements.txt...
"!VENV_DIR!\Scripts\python.exe" -m pip install -r "%REQUIREMENTS%" --disable-pip-version-check
if !errorlevel! neq 0 (
    echo Error: package installation failed.
    goto fail
)

"!VENV_DIR!\Scripts\python.exe" -m pip check
if !errorlevel! neq 0 (
    echo Error: installed packages have dependency conflicts.
    goto fail
)

echo In Unity, check "Manually Installed Python" and set "Path of Python Executable" to:
echo   !VENV_DIR!\Scripts\python.exe

:done
echo.
echo ========================================
echo Installation completed successfully.
echo ========================================
if not defined UNATTENDED pause
popd
exit /b 0

:fail
if not defined UNATTENDED pause
popd
exit /b 1

REM ---------------------------------------------------------------- subroutines

:has_vc_runtime
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1 && exit /b 0
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1 && exit /b 0
exit /b 1

:verify_sha256
REM Arguments: the file, its expected SHA-256 (hex).
if not exist "%~1" (
    echo Error: bundled installer not found: %~1
    exit /b 1
)
set "ACTUAL_SHA="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%~1" SHA256 2^>nul') do (
    if not defined ACTUAL_SHA set "ACTUAL_SHA=%%H"
)
if not defined ACTUAL_SHA (
    echo Error: could not compute the SHA-256 of %~1 ^(certutil failed^).
    exit /b 1
)
REM Older certutil versions separate the bytes with spaces.
set "ACTUAL_SHA=!ACTUAL_SHA: =!"
if /i "!ACTUAL_SHA!"=="%~2" exit /b 0
echo Error: %~1 is damaged or incomplete ^(SHA-256 !ACTUAL_SHA!, expected %~2^); it was not run.
echo Re-download the repository ^(if it was cloned with Git LFS: git lfs pull^) or install Python 3.13 yourself.
exit /b 1
