#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_PACKAGE="python3.11"
PYTHON_INSTALL_DIR="/usr"
PYTHON_EXE="/usr/bin/python3.11"

REQUIREMENTS="$SCRIPT_DIR/Data/Installation_Objects/requirements.txt"



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
sudo apt update && sudo apt install -y --install-recommends ${PYTHON_PACKAGE}

# Check if the installation was successful
if [ -x "${PYTHON_EXE}" ]; then
    echo "Python was successfully installed."
else
    echo "Error installing Python."
    exit 1
fi

install_packages
