#!/usr/bin/env python3
"""Stop hook handler for Claude Code.

Handles session end operations.
"""

import json
import sys

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Now we can import cleanly without explicit sys.path.insert calls
from modules.utils.process_manager import managed_subprocess_run


def main():
    """Main hook handler."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run session end operations with managed subprocess
    try:
        managed_subprocess_run(
            ["npx", "claude-flow@alpha", "hooks", "session-end",
             "--generate-summary", "true",
             "--persist-state", "true",
             "--export-metrics", "true"],
            check=True,
            timeout=30,  # 30 second timeout
            max_memory_mb=100,  # 100MB memory limit
            tags={"hook": "session-end", "type": "cleanup"}
        )
    except Exception as e:
        print(f"Error running session-end: {e}", file=sys.stderr)
        # Don't block on errors
    
    sys.exit(0)


if __name__ == "__main__":
    main()