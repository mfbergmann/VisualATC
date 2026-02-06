#!/usr/bin/env bash
set -euo pipefail

echo "=== VisualATC Setup ==="
echo ""

# Check Python version
PYTHON=""
for cmd in python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.9+ is required but not found."
    exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PY_VERSION ($PYTHON)"

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo ""
    echo "WARNING: ffmpeg is not installed."
    echo "  Install it with:"
    echo "    macOS:   brew install ffmpeg"
    echo "    Ubuntu:  sudo apt install ffmpeg"
    echo "    Windows: choco install ffmpeg"
    echo ""
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start VisualATC:"
echo "  source venv/bin/activate"
echo "  python run.py"
echo ""
echo "Then open http://localhost:8765 in your browser."
