#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

VENV_DIR="$(pwd)/.venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "Error: Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

"$VENV_DIR/bin/python" run.py "$@"
