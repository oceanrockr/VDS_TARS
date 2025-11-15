#!/bin/bash
# RiPIT Environment Activation Script
# Source this file to activate RiPIT for this project

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect platform
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Git Bash on Windows
    VENV_ACTIVATE="${PROJECT_ROOT}/.venv/Scripts/activate"
else
    # Linux/macOS
    VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"
fi

# Activate virtual environment
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
else
    echo "❌ Virtual environment not found: $VENV_ACTIVATE"
    return 1
fi

# Set RiPIT environment variables
export RIPIT_HOME="${PROJECT_ROOT}/.ripit"
export PYTHONPATH="${RIPIT_HOME}/ace_stub:${PYTHONPATH}"

echo "✅ RiPIT environment activated"
echo "   Project: $(basename "$PROJECT_ROOT")"
echo "   RiPIT Home: $RIPIT_HOME"
echo "   Python: $(which python3)"
echo ""
echo "To deactivate: deactivate"
