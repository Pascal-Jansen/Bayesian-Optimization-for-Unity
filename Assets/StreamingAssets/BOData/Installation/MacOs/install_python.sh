#!/bin/bash

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_INSTALLER="$SCRIPT_DIR/Data/Installation_Objects/python-3.13.7-macos11.pkg"
PYTHON_INSTALL_DIR="/usr"
PYTHON_EXE="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
PYTHON_TARGET_MM="3.13"

REQUIREMENTS="$SCRIPT_DIR/../requirements.txt"



ensure_not_root() {
    if [ "$(id -u)" -eq 0 ]; then
        echo "Do not run this script with sudo. It installs Python packages with --user for the Unity user account."
        exit 1
    fi
}

is_target_python_installed() {
    if [ ! -x "$PYTHON_EXE" ]; then
        return 1
    fi
    local installed_version
    installed_version="$("$PYTHON_EXE" --version 2>&1 | awk '{print $2}')"
    if [[ "$installed_version" == ${PYTHON_TARGET_MM}.* ]]; then
        echo "Found target Python version: $installed_version ($PYTHON_EXE)"
        return 0
    fi
    echo "Found Python $installed_version, but expected ${PYTHON_TARGET_MM}.x"
    return 1
}

verify_supported_python_architecture() {
    local python_machine
    python_machine="$("$PYTHON_EXE" -c 'import platform; print(platform.machine())')"
    if [ "$python_machine" != "arm64" ]; then
        echo "Unsupported macOS Python architecture: $python_machine"
        echo "The pinned PyTorch dependency currently ships Python 3.13 macOS wheels for arm64 only."
        exit 1
    fi
}

install_packages() {
    if [ ! -f "$REQUIREMENTS" ]; then
        echo "Requirements file not found: $REQUIREMENTS"
        exit 1
    fi

    echo "Checking pip..."
    if ! "$PYTHON_EXE" -m pip --version > /dev/null 2>&1; then
        echo "Installing bundled pip..."
        "$PYTHON_EXE" -m ensurepip --upgrade
    fi

    # Install packages
    echo "Installing packages..."
    "$PYTHON_EXE" -m pip install --user -r "$REQUIREMENTS"

    # Check if the package installation was successful
    for package in numpy scipy matplotlib pandas torch gpytorch botorch moocore scikit-learn loguru; do
        if ! "$PYTHON_EXE" -m pip show "$package" > /dev/null 2>&1; then
            echo "Error installing package: $package"
            exit 1
        fi
    done
    "$PYTHON_EXE" -m pip check
    echo "Packages were successfully installed."
}



ensure_not_root

# Install Python only when target version is not already present
if is_target_python_installed; then
    echo "Skipping Python installation."
else
    echo "Installing Python..."
    sudo installer -pkg "$PYTHON_INSTALLER" -target /

    # Check if the installation was successful
    if is_target_python_installed; then
        echo "Python was successfully installed."
    else
        echo "Error installing target Python version."
        exit 1
    fi
fi

verify_supported_python_architecture
install_packages


# Remove quarantine attribute for .app files
echo "Removing quarantine attribute for .app files..."
find "$SCRIPT_DIR" -name "*.app" -print0 | while IFS= read -r -d $'\0' app_file; do
    echo "Removing quarantine attribute for: $app_file"
    xattr -d com.apple.quarantine "$app_file" 2>/dev/null || true
done
