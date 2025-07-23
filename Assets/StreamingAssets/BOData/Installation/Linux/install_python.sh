#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# 0. Paths relative to this script
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR/Data/Installation_Objects"
mkdir -p "$INSTALL_DIR"

REQUIREMENTS="$SCRIPT_DIR/../requirements.txt"

# ──────────────────────────────────────────────────────────────
# 1. Detect newest stable Python on python.org
# ──────────────────────────────────────────────────────────────
echo "Detecting latest stable CPython …"
PY_VER=$(curl -s https://www.python.org/ftp/python/ \
  | grep -oE '[0-9]+\.[0-9]+\.[0-9]+/' \
  | tr -d '/' \
  | grep -vE '[abrc]' \
  | sort -V \
  | tail -1)
echo "Latest version: $PY_VER"

MAJMIN=$(echo "$PY_VER" | cut -d. -f1,2)          # e.g. 3.12
TARBALL="Python-${PY_VER}.tar.xz"
URL="https://www.python.org/ftp/python/${PY_VER}/${TARBALL}"
SRC_DIR="$INSTALL_DIR/Python-${PY_VER}"
PKG_PATH="$INSTALL_DIR/$TARBALL"

# ──────────────────────────────────────────────────────────────
# 2. Download & extract source tarball (cached if re-run)
# ──────────────────────────────────────────────────────────────
if [[ ! -f "$PKG_PATH" ]]; then
  echo "Downloading $TARBALL …"
  curl -L "$URL" -o "$PKG_PATH"
fi

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Extracting source …"
  tar -xf "$PKG_PATH" -C "$INSTALL_DIR"
fi

# ──────────────────────────────────────────────────────────────
# 3. Install build prerequisites (Debian/Ubuntu-style);
#    silently ignore if package manager is different.
# ──────────────────────────────────────────────────────────────
if command -v apt-get &>/dev/null; then
  echo "Installing build dependencies …"
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    build-essential zlib1g-dev libssl-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev curl
fi

# ──────────────────────────────────────────────────────────────
# 4. Configure, make, altinstall (≈2–5 min on modern CPUs)
# ──────────────────────────────────────────────────────────────
cd "$SRC_DIR"
echo "Configuring CPython …"
./configure --enable-optimizations --prefix=/usr/local > /dev/null

echo "Building CPython (this may take a while) …"
make -s -j"$(nproc)"

echo "Installing CPython with altinstall …"
sudo make altinstall   # installs python3.<x> but leaves 'python3' untouched

PYTHON_EXE="/usr/local/bin/python${MAJMIN}"
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "ERROR: Interpreter not found at $PYTHON_EXE"
  exit 1
fi
echo "Python installed: $("$PYTHON_EXE" --version)"

# ──────────────────────────────────────────────────────────────
# 5. Upgrade pip & install requirements
# ──────────────────────────────────────────────────────────────
echo "Upgrading pip …"
"$PYTHON_EXE" -m pip install --upgrade pip

echo "Installing requirements …"
"$PYTHON_EXE" -m pip install -r "$REQUIREMENTS"

echo "Verifying core packages …"
"$PYTHON_EXE" -m pip show numpy scipy pandas >/dev/null || {
  echo "Package install failed."; exit 1; }

echo "✓ All done.  Using $PYTHON_EXE"
