#!/usr/bin/env bash
# Installs Python 3.13 with its venv module for BOforUnity on Linux.
#
# The Python packages from requirements.txt are NOT installed here: on the first Play, Unity creates a private
# virtual environment (persistentDataPath/BOData/python-venv) and installs them into it (README 8.5). This script
# makes sure that works: Python 3.13 is present and can create virtual environments.
#
# Usage:
#   ./install_python.sh                 install/verify Python 3.13 and its venv module
#   ./install_python.sh --venv DIR      additionally create a virtual environment in DIR and install
#                                       requirements.txt into it; then set "Manually Installed Python" in Unity
#                                       to DIR/bin/python (Unity uses that environment as it is)
#
# Supported automatically: Ubuntu (deadsnakes PPA when the release has no python3.13) and Debian 13+.
# Other distributions: install Python 3.13 and its venv module with the package manager, then rerun this script.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_VERSION="3.13"
PYTHON_PACKAGE="python${PYTHON_VERSION}"
REQUIREMENTS="${INSTALLATION_DIR}/requirements.txt"
WHEELS_DIR="${INSTALLATION_DIR}/wheels"
# PyPI's Linux torch wheels are CUDA builds that pull several GB of nvidia-* packages; BOforUnity's GP workloads
# run on the CPU. With PyTorch's CPU index added, pip picks torch 2.14.1+cpu, which satisfies "torch==2.14.1".
CPU_TORCH_INDEX="https://download.pytorch.org/whl/cpu"

PYTHON_EXE=""
VENV_DIR=""
SUDO=""

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

usage() {
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --venv)
                if [[ $# -lt 2 || -z "$2" ]]; then
                    log_error "--venv needs a directory."
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
                log_error "Unknown argument: $1"
                usage
                exit 2
                ;;
        esac
    done
}

check_privileges() {
    if [[ ${EUID} -eq 0 ]]; then
        log_error "Do not run this script as root; it asks for sudo where needed. Files it creates must belong to the Unity user."
        exit 1
    fi
    SUDO="sudo"
}

# Sets PYTHON_EXE to the first python3.13 on PATH (or the system location), if any.
find_python() {
    local candidate
    candidate="$(command -v "${PYTHON_PACKAGE}" 2>/dev/null || true)"
    if [[ -z "${candidate}" && -x "/usr/bin/${PYTHON_PACKAGE}" ]]; then
        candidate="/usr/bin/${PYTHON_PACKAGE}"
    fi
    if [[ -z "${candidate}" ]]; then
        return 1
    fi

    local version
    version="$("${candidate}" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || true)"
    if [[ "${version}" != ${PYTHON_VERSION}.* ]]; then
        log_warn "${candidate} reports version '${version}', expected ${PYTHON_VERSION}.x."
        return 1
    fi
    PYTHON_EXE="${candidate}"
    log_info "Using Python ${version} at ${PYTHON_EXE}"
    return 0
}

read_os_release() {
    OS_ID=""
    OS_ID_LIKE=""
    OS_NAME="this Linux distribution"
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        OS_ID="${ID:-}"
        OS_ID_LIKE="${ID_LIKE:-}"
        OS_NAME="${PRETTY_NAME:-${OS_NAME}}"
    fi
}

is_ubuntu_like() {
    [[ "${OS_ID}" == "ubuntu" || " ${OS_ID_LIKE} " == *" ubuntu "* ]]
}

is_debian_like() {
    [[ "${OS_ID}" == "debian" || " ${OS_ID_LIKE} " == *" debian "* ]] || is_ubuntu_like
}

unsupported_distribution() {
    log_error "Automatic installation of Python ${PYTHON_VERSION} is only supported on Ubuntu and Debian (found: ${OS_NAME})."
    log_error "Install Python ${PYTHON_VERSION} with its venv module using your package manager, for example:"
    log_error "  Fedora:  sudo dnf install python${PYTHON_VERSION}"
    log_error "  Arch:    sudo pacman -S python   (if it provides ${PYTHON_VERSION}), or use pyenv / uv"
    log_error "then run this script again."
    exit 1
}

install_python() {
    read_os_release
    if ! is_debian_like || ! command -v apt-get >/dev/null 2>&1; then
        unsupported_distribution
    fi

    log_info "Installing Python ${PYTHON_VERSION} on ${OS_NAME}..."
    ${SUDO} apt-get update

    if ! apt-cache show "${PYTHON_PACKAGE}" >/dev/null 2>&1; then
        if is_ubuntu_like; then
            # The deadsnakes PPA only exists for Ubuntu releases.
            log_info "Python ${PYTHON_VERSION} is not in this release's repositories; adding the deadsnakes PPA..."
            ${SUDO} apt-get install -y software-properties-common
            ${SUDO} add-apt-repository -y ppa:deadsnakes/ppa
            ${SUDO} apt-get update
        else
            log_error "Python ${PYTHON_VERSION} is not available in the repositories of ${OS_NAME}."
            log_error "Debian ships Python ${PYTHON_VERSION} from Debian 13 (trixie) on. On older releases install it with pyenv or uv,"
            log_error "then run this script again."
            exit 1
        fi
    fi

    ${SUDO} apt-get install -y "${PYTHON_PACKAGE}" "${PYTHON_PACKAGE}-venv"

    if ! find_python; then
        log_error "Python ${PYTHON_VERSION} was installed, but no ${PYTHON_PACKAGE} executable was found on PATH."
        exit 1
    fi
}

# Unity creates its private environment with "python -m venv"; Debian/Ubuntu ship that module separately.
venv_works() {
    local probe
    probe="$(mktemp -d)"
    local ok=1
    if "${PYTHON_EXE}" -m venv "${probe}/venv" >/dev/null 2>&1 && "${probe}/venv/bin/python" -m pip --version >/dev/null 2>&1; then
        ok=0
    fi
    rm -rf "${probe}"
    return "${ok}"
}

ensure_venv_support() {
    if venv_works; then
        log_info "Python ${PYTHON_VERSION} can create virtual environments."
        return 0
    fi

    read_os_release
    if is_debian_like && command -v apt-get >/dev/null 2>&1; then
        log_info "Installing the venv module (${PYTHON_PACKAGE}-venv)..."
        ${SUDO} apt-get install -y "${PYTHON_PACKAGE}-venv"
    fi

    if ! venv_works; then
        log_error "${PYTHON_EXE} cannot create virtual environments (python -m venv / ensurepip failed)."
        log_error "Install the venv module for Python ${PYTHON_VERSION} with your package manager and run this script again."
        log_error "Without it, Unity falls back to 'pip install --user', which externally managed system Pythons refuse."
        exit 1
    fi
    log_info "Python ${PYTHON_VERSION} can create virtual environments."
}

install_requirements_into_venv() {
    if [[ ! -f "${REQUIREMENTS}" ]]; then
        log_error "Requirements file not found: ${REQUIREMENTS}"
        exit 1
    fi

    log_info "Creating the virtual environment ${VENV_DIR}..."
    "${PYTHON_EXE}" -m venv "${VENV_DIR}"
    local venv_python="${VENV_DIR}/bin/python"

    local pip_args=(install -r "${REQUIREMENTS}" --disable-pip-version-check --extra-index-url "${CPU_TORCH_INDEX}")
    if [[ -d "${WHEELS_DIR}" ]]; then
        log_info "Using local wheels from ${WHEELS_DIR}"
        pip_args+=(--find-links "${WHEELS_DIR}")
    fi

    log_info "Installing packages from requirements.txt (CPU build of PyTorch)..."
    "${venv_python}" -m pip "${pip_args[@]}"
    "${venv_python}" -m pip check

    log_info "Installed packages:"
    "${venv_python}" -m pip list 2>/dev/null | grep -Ei '^(numpy|scipy|matplotlib|pandas|torch|gpytorch|botorch|moocore|scikit-learn|loguru) ' || true
    log_info "In Unity, check 'Manually Installed Python' and set 'Path of Python Executable' to:"
    log_info "  $(cd "${VENV_DIR}" && pwd)/bin/python"
}

main() {
    parse_args "$@"
    check_privileges

    log_info "Setting up Python ${PYTHON_VERSION} for BOforUnity..."
    if find_python; then
        log_info "Python ${PYTHON_VERSION} is already installed; skipping the Python installation."
    else
        install_python
    fi

    ensure_venv_support

    if [[ -n "${VENV_DIR}" ]]; then
        install_requirements_into_venv
    else
        log_info "Unity installs the Python packages into its own environment on the first Play (one time, a few minutes)."
    fi

    log_info "Setup completed successfully."
}

main "$@"
