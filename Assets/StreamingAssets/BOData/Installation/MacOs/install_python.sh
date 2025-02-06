#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_INSTALLER="$SCRIPT_DIR/Data/Installation_Objects/python-3.11.3-macos11.pkg"
PYTHON_INSTALL_DIR="/usr"
PYTHON_EXE="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"

REQUIREMENTS="$SCRIPT_DIR/../../../../../requirements.txt"



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



# Install Python
echo "Installing Python..."
sudo installer -pkg "$PYTHON_INSTALLER" -target /

# Check if the installation was successful
if [ -x "$PYTHON_EXE" ]; then
    echo "Python was successfully installed."
else
    echo "Error installing Python."
    exit 1
fi

install_packages


# Remove quarantine attribute for .app files
echo "Removing quarantine attribute for .app files..."
find "$SCRIPT_DIR" -name "*.app" -print0 | while IFS= read -r -d $'\0' app_file; do
    echo "Removing quarantine attribute for: $app_file"
    xattr -d com.apple.quarantine "$app_file"
done
