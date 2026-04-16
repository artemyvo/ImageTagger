#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

VENV_DIR="$(pwd)/.venv"

echo "Looking for Python..."

PYTHON=""

for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; v=sys.version_info; exit(0 if v>=(3,9) else 1)" 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.9 or newer is required but was not found in PATH."
    echo "Download Python from https://www.python.org/downloads/"
    exit 1
fi

echo "Using Python: $PYTHON"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment at \"$VENV_DIR\"..."
    if ! "$PYTHON" -m venv "$VENV_DIR" 2>/dev/null; then
        echo ""
        echo "Error: Python venv module is not available."
        echo "On Debian/Ubuntu, install it with:"
        echo "  sudo apt install python3-venv"
        echo "On Fedora/RHEL:"
        echo "  sudo dnf install python3"
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

if [ ! -f "$VENV_DIR/bin/pip" ] && ! "$VENV_DIR/bin/python" -m pip --version &>/dev/null; then
    echo "pip not found in venv, attempting to bootstrap via ensurepip..."
    if ! "$VENV_DIR/bin/python" -m ensurepip --upgrade 2>/dev/null; then
        echo ""
        echo "Error: pip could not be bootstrapped into the virtual environment."
        echo "On Debian/Ubuntu (including Ubuntu 22.04+), install the full Python package:"
        echo "  sudo apt install python3-full"
        echo "Then delete the incomplete venv and re-run this script:"
        echo "  rm -rf .venv && ./install.sh"
        exit 1
    fi
fi

echo "Upgrading pip..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "Installing dependencies..."
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo ""
echo "Install complete. Run ./run.sh to start ImageTagger."
