#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

VENV_DIR="$(pwd)/.venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "Error: Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

if ! "$VENV_DIR/bin/python" -c "import pytest, pytestqt, pytest_timeout" 2>/dev/null; then
    echo "Installing test dependencies..."
    "$VENV_DIR/bin/python" -m pip install -q -r requirements-dev.txt
fi

"$VENV_DIR/bin/python" -m pytest "$@"
