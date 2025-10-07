#!/bin/bash
set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="3.13"
PYTHON_PACKAGE="python${PYTHON_VERSION}"
PYTHON_EXE="/usr/bin/python${PYTHON_VERSION}"
REQUIREMENTS="$SCRIPT_DIR/../requirements.txt"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running with appropriate privileges
check_privileges() {
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root. This script will use sudo when necessary."
        SUDO=""
    else
        SUDO="sudo"
    fi
}

# Check if Python 3.13 is already installed and get latest available version
check_python_installed() {
    if command -v "python${PYTHON_VERSION}" &> /dev/null; then
        INSTALLED_VERSION=$("python${PYTHON_VERSION}" --version 2>&1 | awk '{print $2}')
        log_info "Python ${INSTALLED_VERSION} is currently installed at $(command -v python${PYTHON_VERSION})"
        
        # Check if there's a newer version available
        log_info "Checking for updates..."
        return 1  # Continue to update check
    else
        log_warn "Python ${PYTHON_VERSION} is not installed."
        return 1
    fi
}

# Install Python 3.13
install_python() {
    log_info "Installing/updating Python ${PYTHON_VERSION}..."
    
    # Update package lists
    log_info "Updating package lists..."
    $SUDO apt update || {
        log_error "Failed to update package lists"
        exit 1
    }
    
    # Check if the package exists in repositories
    if ! apt-cache show "${PYTHON_PACKAGE}" &> /dev/null; then
        log_error "Python ${PYTHON_VERSION} not found in default repositories."
        log_info "Attempting to add deadsnakes PPA for newer Python versions..."
        
        # Install software-properties-common if not present
        $SUDO apt install -y software-properties-common
        
        # Add deadsnakes PPA
        $SUDO add-apt-repository -y ppa:deadsnakes/ppa
        $SUDO apt update
    fi
    
    # Get available version
    AVAILABLE_VERSION=$(apt-cache policy "${PYTHON_PACKAGE}" | grep Candidate | awk '{print $2}')
    log_info "Latest available version: ${AVAILABLE_VERSION}"
    
    # Install/upgrade Python and related packages
    log_info "Installing ${PYTHON_PACKAGE} and essential tools..."
    $SUDO apt install -y --install-recommends \
        "${PYTHON_PACKAGE}" \
        "${PYTHON_PACKAGE}-venv" \
        "${PYTHON_PACKAGE}-dev" \
        python3-pip \
        build-essential || {
        log_error "Failed to install Python ${PYTHON_VERSION}"
        exit 1
    }
    
    # Verify installation
    if [ -x "${PYTHON_EXE}" ]; then
        FINAL_VERSION=$("${PYTHON_EXE}" --version 2>&1 | awk '{print $2}')
        log_info "Python ${FINAL_VERSION} successfully installed at ${PYTHON_EXE}"
    else
        log_error "Python executable not found at ${PYTHON_EXE}"
        exit 1
    fi
}

# Install Python packages
install_packages() {
    log_info "Setting up Python packages..."
    
    # Check if requirements file exists
    if [ ! -f "$REQUIREMENTS" ]; then
        log_error "Requirements file not found: $REQUIREMENTS"
        exit 1
    fi
    
    # Upgrade pip
    log_info "Upgrading pip..."
    "${PYTHON_EXE}" -m pip install --upgrade pip setuptools wheel || {
        log_error "Failed to upgrade pip"
        exit 1
    }
    
    # Install packages from requirements
    log_info "Installing packages from requirements.txt..."
    "${PYTHON_EXE}" -m pip install -r "$REQUIREMENTS" || {
        log_error "Failed to install packages from requirements.txt"
        exit 1
    }
    
    # Verify installation of key packages
    log_info "Verifying package installation..."
    EXPECTED_PACKAGES="numpy scipy matplotlib pandas torch gpytorch botorch"
    MISSING_PACKAGES=""
    
    for package in $EXPECTED_PACKAGES; do
        if ! "${PYTHON_EXE}" -m pip list | grep -i "^${package}" > /dev/null 2>&1; then
            MISSING_PACKAGES="${MISSING_PACKAGES} ${package}"
        fi
    done
    
    if [ -z "$MISSING_PACKAGES" ]; then
        log_info "All expected packages were successfully installed."
        log_info "Installed packages:"
        "${PYTHON_EXE}" -m pip list | grep -E "numpy|scipy|matplotlib|pandas|torch|gpytorch|botorch"
    else
        log_error "The following packages were not installed:${MISSING_PACKAGES}"
        exit 1
    fi
}

# Create virtual environment (optional but recommended)
create_venv() {
    VENV_DIR="$SCRIPT_DIR/../venv"
    
    read -p "Would you like to create a virtual environment? (recommended) [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Creating virtual environment at ${VENV_DIR}..."
        "${PYTHON_EXE}" -m venv "$VENV_DIR" || {
            log_error "Failed to create virtual environment"
            return 1
        }
        log_info "Virtual environment created. Activate it with:"
        log_info "  source ${VENV_DIR}/bin/activate"
    fi
}

# Main execution
main() {
    log_info "Starting Python ${PYTHON_VERSION} setup script..."
    
    check_privileges
    
    # Always run install_python to ensure we have the latest version
    check_python_installed
    install_python
    
    install_packages
    
    # Optionally create virtual environment
    # Uncomment the next line if you want to prompt for venv creation
    # create_venv
    
    log_info "Setup completed successfully!"
}

# Run main function
main