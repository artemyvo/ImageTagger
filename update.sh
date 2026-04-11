#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

VENV_DIR="$(pwd)/.venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "Error: Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

if command -v git &>/dev/null; then
    echo "Pulling latest changes..."
    git pull || echo "Warning: git pull failed. Continuing with dependency update."
else
    echo "git not found in PATH, skipping repository update."
fi

echo "Upgrading pip..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "Updating dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade -r requirements.txt

echo ""
echo "Update complete."
