#!/bin/bash
set -euo pipefail

# ──────────────────────────────
# 0. Paths relative to script
# ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR/Data/Installation_Objects"
mkdir -p "$INSTALL_DIR"
REQUIREMENTS="$SCRIPT_DIR/../requirements.txt"

# ──────────────────────────────
# 1. Detect latest stable Python
# ──────────────────────────────
echo "Detecting latest stable Python for macOS…"
LATEST_VER=$(curl -s https://www.python.org/ftp/python/ \
  | grep -oE '[0-9]+\.[0-9]+\.[0-9]+/' \
  | tr -d '/' \
  | grep -vE '[abrc]' \
  | sort -V \
  | tail -1)
echo "Latest version found: $LATEST_VER"

# ──────────────────────────────
# 2. Choose pkg variant by OS
# ──────────────────────────────
OS_VER=$(sw_vers -productVersion)          # e.g. 12.7.5
OS_MAJOR=${OS_VER%%.*}                     # 12
if [ "$OS_MAJOR" -ge 11 ]; then
  PKG_SUFFIX="macos11.pkg"                 # universal-2
else
  PKG_SUFFIX="macosx10.9.pkg"              # Intel-only
fi

PKG_NAME="python-${LATEST_VER}-${PKG_SUFFIX}"
PKG_URL="https://www.python.org/ftp/python/${LATEST_VER}/${PKG_NAME}"
PKG_PATH="$INSTALL_DIR/$PKG_NAME"
echo "Selected package: $PKG_NAME"

# ──────────────────────────────
# 3. Download if not cached
# ──────────────────────────────
if [ ! -f "$PKG_PATH" ]; then
  echo "Downloading $PKG_URL …"
  curl -L "$PKG_URL" -o "$PKG_PATH"
fi

# ──────────────────────────────
# 4. Install Python (needs sudo)
# ──────────────────────────────
echo "Installing Python $LATEST_VER …"
sudo installer -pkg "$PKG_PATH" -target /

# interpreter lives in Versions/<major.minor>/bin/python3
MAJMIN=$(echo "$LATEST_VER" | cut -d. -f1,2)   # e.g. 3.12
PYTHON_EXE="/Library/Frameworks/Python.framework/Versions/${MAJMIN}/bin/python3"
if [ ! -x "$PYTHON_EXE" ]; then
  echo "ERROR: Interpreter not found at $PYTHON_EXE"
  exit 1
fi
echo "Python installed: $("$PYTHON_EXE" --version)"

# ──────────────────────────────
# 5. Upgrade pip & install deps
# ──────────────────────────────
echo "Upgrading pip …"
"$PYTHON_EXE" -m pip install --upgrade pip

echo "Installing requirements …"
"$PYTHON_EXE" -m pip install -r "$REQUIREMENTS"

echo "Verifying core packages …"
"$PYTHON_EXE" -m pip show numpy scipy pandas >/dev/null || {
  echo "Package install failed."; exit 1; }

# ──────────────────────────────
# 6. Remove quarantine bit from any .app bundles
# ──────────────────────────────
echo "Removing quarantine attributes …"
find "$SCRIPT_DIR" -name "*.app" -exec xattr -d com.apple.quarantine {} + 2>/dev/null || true

echo "Setup complete."
