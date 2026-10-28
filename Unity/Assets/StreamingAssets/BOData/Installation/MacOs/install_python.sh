#!/usr/bin/env bash
# Installs the bundled Python 3.13 (python.org universal2 installer) for BOforUnity on macOS.
#
# The Python packages from requirements.txt are NOT installed here: on the first Play, Unity creates a private
# virtual environment (persistentDataPath/BOData/python-venv) and installs them into it (README 8.5).
#
# Usage:
#   ./install_python.sh                 install/verify Python 3.13
#   ./install_python.sh --venv DIR      additionally create a virtual environment in DIR and install
#                                       requirements.txt into it; then set "Manually Installed Python" in Unity
#                                       to DIR/bin/python (Unity uses that environment as it is)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_INSTALLER="${SCRIPT_DIR}/Data/Installation_Objects/python-3.13.7-macos11.pkg"
# SHA-256 of the bundled installer; a truncated copy or a Git LFS pointer file must not run with admin rights.
PYTHON_INSTALLER_SHA256="f7e8c8d63ab0a4e736b5864aa369098b16af622042c079addb2f1a08400560c5"
PYTHON_EXE="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
PYTHON_TARGET_MM="3.13"

REQUIREMENTS="${INSTALLATION_DIR}/requirements.txt"
WHEELS_DIR="${INSTALLATION_DIR}/wheels"
VENV_DIR=""
# Set when this shell runs under Rosetta: the universal2 Python must still start as arm64.
ARCH_PREFIX=()

usage() {
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --venv)
                if [[ $# -lt 2 || -z "$2" ]]; then
                    echo "--venv needs a directory." >&2
                    exit 2
                fi
                VENV_DIR="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                usage
                exit 2
                ;;
        esac
    done
}

ensure_not_root() {
    if [[ "$(id -u)" -eq 0 ]]; then
        echo "Do not run this script with sudo; it asks for the administrator password where needed."
        exit 1
    fi
}

detect_rosetta() {
    if [[ "$(/usr/sbin/sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" == "1" ]]; then
        echo "This shell runs under Rosetta; starting Python as arm64."
        ARCH_PREFIX=(/usr/bin/arch -arm64)
    fi
}

run_python() {
    # "${ARCH_PREFIX[@]}" is empty unless running under Rosetta (bash 3.2-safe expansion).
    ${ARCH_PREFIX[@]+"${ARCH_PREFIX[@]}"} "$@"
}

is_target_python_installed() {
    if [[ ! -x "${PYTHON_EXE}" ]]; then
        return 1
    fi
    local installed_version
    installed_version="$(run_python "${PYTHON_EXE}" --version 2>&1 | awk '{print $2}')"
    if [[ "${installed_version}" == ${PYTHON_TARGET_MM}.* ]]; then
        echo "Found target Python version: ${installed_version} (${PYTHON_EXE})"
        return 0
    fi
    echo "Found Python ${installed_version}, but expected ${PYTHON_TARGET_MM}.x"
    return 1
}

verify_installer() {
    if [[ ! -f "${PYTHON_INSTALLER}" ]]; then
        echo "Bundled installer not found: ${PYTHON_INSTALLER}"
        exit 1
    fi
    local actual
    actual="$(/usr/bin/shasum -a 256 "${PYTHON_INSTALLER}" | awk '{print $1}')"
    if [[ "${actual}" != "${PYTHON_INSTALLER_SHA256}" ]]; then
        echo "The bundled installer is damaged or incomplete (SHA-256 ${actual}, expected ${PYTHON_INSTALLER_SHA256})."
        echo "Re-download the repository (if it was cloned with Git LFS, run 'git lfs pull') or install Python 3.13 from python.org."
        exit 1
    fi
}

verify_supported_python_architecture() {
    local python_machine
    python_machine="$(run_python "${PYTHON_EXE}" -c 'import platform; print(platform.machine())')"
    if [[ "${python_machine}" != "arm64" ]]; then
        echo "Unsupported macOS Python architecture: ${python_machine}"
        echo "The pinned PyTorch dependency currently ships Python 3.13 macOS wheels for arm64 (Apple Silicon) only."
        exit 1
    fi
}

verify_venv_support() {
    local probe
    probe="$(mktemp -d)"
    if ! run_python "${PYTHON_EXE}" -m venv "${probe}/venv" >/dev/null 2>&1; then
        rm -rf "${probe}"
        echo "${PYTHON_EXE} cannot create virtual environments; reinstall Python 3.13 from python.org."
        exit 1
    fi
    rm -rf "${probe}"
    echo "Python ${PYTHON_TARGET_MM} can create virtual environments."
}

install_requirements_into_venv() {
    if [[ ! -f "${REQUIREMENTS}" ]]; then
        echo "Requirements file not found: ${REQUIREMENTS}"
        exit 1
    fi

    echo "Creating the virtual environment ${VENV_DIR}..."
    run_python "${PYTHON_EXE}" -m venv "${VENV_DIR}"
    local venv_python="${VENV_DIR}/bin/python"

    local pip_args=(install -r "${REQUIREMENTS}" --disable-pip-version-check)
    if [[ -d "${WHEELS_DIR}" ]]; then
        echo "Using local wheels from ${WHEELS_DIR}"
        pip_args+=(--find-links "${WHEELS_DIR}")
    fi

    echo "Installing packages..."
    run_python "${venv_python}" -m pip "${pip_args[@]}"
    run_python "${venv_python}" -m pip check
    echo "Packages were successfully installed."
    echo "In Unity, check 'Manually Installed Python' and set 'Path of Python Executable' to:"
    echo "  $(cd "${VENV_DIR}" && pwd)/bin/python"
}

parse_args "$@"
ensure_not_root
detect_rosetta

# Install Python only when the target version is not already present.
if is_target_python_installed; then
    echo "Skipping Python installation."
else
    verify_installer
    echo "Installing Python..."
    sudo installer -pkg "${PYTHON_INSTALLER}" -target /

    if is_target_python_installed; then
        echo "Python was successfully installed."
    else
        echo "Error installing target Python version."
        exit 1
    fi
fi

verify_supported_python_architecture
verify_venv_support

if [[ -n "${VENV_DIR}" ]]; then
    install_requirements_into_venv
else
    echo "Unity installs the Python packages into its own environment on the first Play (one time, a few minutes)."
fi

# Remove quarantine attribute for .app files
echo "Removing quarantine attribute for .app files..."
find "${SCRIPT_DIR}" -name "*.app" -print0 | while IFS= read -r -d $'\0' app_file; do
    echo "Removing quarantine attribute for: ${app_file}"
    xattr -d com.apple.quarantine "${app_file}" 2>/dev/null || true
done
