#!/bin/bash
# Claude Flow Memory CLI Wrapper
# This script integrates with npx claude-flow memory commands

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/modules/memory/claude_flow_cli.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Memory CLI script not found at $PYTHON_SCRIPT" >&2
    exit 1
fi

# Forward all arguments to the Python CLI
python3 "$PYTHON_SCRIPT" "$@"