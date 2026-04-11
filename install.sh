#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

VENV_DIR="$(pwd)/.venv"

echo "Looking for Python..."

PYTHON=""

for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; v=sys.version_info; exit(0 if v>=(3,10) else 1)" 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.10 or newer is required but was not found in PATH."
    echo "Download Python from https://www.python.org/downloads/"
    exit 1
fi

echo "Using Python: $PYTHON"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment at \"$VENV_DIR\"..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

echo "Upgrading pip..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "Installing dependencies..."
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo ""
echo "Install complete. Run ./run.sh to start ImageTagger."
