#!/bin/bash

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_INSTALLER="$SCRIPT_DIR/Data/Installation_Objects/python-3.13.7-macos11.pkg"
PYTHON_INSTALL_DIR="/usr"
PYTHON_EXE="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
PYTHON_TARGET_MM="3.13"

REQUIREMENTS="$SCRIPT_DIR/../requirements.txt"



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

install_packages() {
    # Upgrade pip
    echo "Upgrading pip..."
    "$PYTHON_EXE" -m pip install --upgrade pip

    # Install packages
    echo "Installing packages..."
    "$PYTHON_EXE" -m pip install -r "$REQUIREMENTS"

    # Check if the package installation was successful
    "$PYTHON_EXE" -m pip list | grep -E "numpy|scipy|matplotlib|pandas|torch|gpytorch|botorch" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Packages were successfully installed."
    else
        echo "Error installing packages."
        exit 1
    fi
}



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

install_packages


# Remove quarantine attribute for .app files
echo "Removing quarantine attribute for .app files..."
find "$SCRIPT_DIR" -name "*.app" -print0 | while IFS= read -r -d $'\0' app_file; do
    echo "Removing quarantine attribute for: $app_file"
    xattr -d com.apple.quarantine "$app_file"
done
