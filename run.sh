#!/bin/bash

# BBN Annotation Viewer - Run Script
# This script activates the virtual environment and runs the Streamlit app

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"

echo "🏥 Starting BBN Annotation Viewer..."
echo ""

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    source "$VENV_PATH/bin/activate"
fi

# Run streamlit
echo "🚀 Launching Streamlit app..."
echo ""
streamlit run "$SCRIPT_DIR/app/main.py" --server.port 8501
