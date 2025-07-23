@echo off
setlocal EnableDelayedExpansion

REM ────────────────────────────────────────────────────────────────
REM 1. Download the latest stable CPython-64-bit installer
REM    (filters out alphas/betas/RCs)
REM ────────────────────────────────────────────────────────────────
echo Detecting latest Python version…

for /f "usebackq tokens=* delims=" %%V in (`
  powershell -NoProfile -Command ^
    "$vers = Invoke-WebRequest -UseBasicParsing 'https://www.python.org/ftp/python/' |" ^
    "        Select-String -Pattern '\d+\.\d+\.\d+/' -AllMatches |" ^
    "        %% { $_.Matches } | %% { $_.Value.TrimEnd('/') } |" ^
    "        Where-Object { $_ -notmatch '[abrc]' } |" ^
    "        Sort-Object {[version]$_} -Descending;" ^
    "Write-Output $vers[0]"
`) do set "PY_VERSION=%%V"

echo Latest stable CPython detected: %PY_VERSION%

set "DL_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe"
set "PYTHON_INSTALLER=Installation_Objects\python-%PY_VERSION%-amd64.exe"

if not exist "Installation_Objects" mkdir "Installation_Objects"

echo Downloading %DL_URL% …
powershell -NoProfile -Command ^
  "Invoke-WebRequest -Uri '%DL_URL%' -OutFile '%PYTHON_INSTALLER%'"

if not exist "%PYTHON_INSTALLER%" (
    echo ERROR: Failed to download Python installer.
    exit /b 1
)

REM ────────────────────────────────────────────────────────────────
REM 2. Visual C++ Redistributable check/installation (unchanged)
REM ────────────────────────────────────────────────────────────────
set "VC_REDIST_EXE=Installation_Objects\VC_redist.x64.exe"

reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Visual C++ Redistributable …
    "%VC_REDIST_EXE%" /quiet
)

REM ────────────────────────────────────────────────────────────────
REM 3. Install Python if not already present
REM ────────────────────────────────────────────────────────────────
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Python %PY_VERSION% …
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1
)

REM ────────────────────────────────────────────────────────────────
REM 4. Locate the newly-installed interpreter
REM ────────────────────────────────────────────────────────────────
for /f "usebackq tokens=*" %%P in (`where python ^| findstr /i "python%PY_VERSION%"`) do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE (
    for /f "usebackq tokens=1" %%P in (`where python`) do set "PYTHON_EXE=%%P"
)

echo Using interpreter: "%PYTHON_EXE%"

REM ────────────────────────────────────────────────────────────────
REM 5. Upgrade pip and install requirements
REM ────────────────────────────────────────────────────────────────
set "REQUIREMENTS=..\requirements.txt"

"%PYTHON_EXE%" -m pip install --upgrade pip
"%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS%"

echo Done.
endlocal
