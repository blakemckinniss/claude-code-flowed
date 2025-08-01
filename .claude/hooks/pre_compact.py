#!/usr/bin/env python3
"""PreCompact hook handler for Claude Code.

Provides guidance before compact operations.
"""

import json
import sys

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()


MANUAL_COMPACT_MESSAGE = """🔄 PreCompact Guidance:
📋 IMPORTANT: Review CLAUDE.md in project root for:
   • 54 available agents and concurrent usage patterns
   • Swarm coordination strategies (hierarchical, mesh, adaptive)
   • SPARC methodology workflows with batchtools optimization
   • Critical concurrent execution rules (GOLDEN RULE: 1 MESSAGE = ALL OPERATIONS)"""

AUTO_COMPACT_MESSAGE = """🔄 Auto-Compact Guidance (Context Window Full):
📋 CRITICAL: Before compacting, ensure you understand:
   • All 54 agents available in .claude/agents/ directory
   • Concurrent execution patterns from CLAUDE.md
   • Batchtools optimization for 300% performance gains
   • Swarm coordination strategies for complex tasks
⚡ Apply GOLDEN RULE: Always batch operations in single messages"""


def main():
    """Main hook handler."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    
    trigger = input_data.get("trigger", "")
    custom_instructions = input_data.get("custom_instructions", "")
    
    # Choose message based on trigger type
    if trigger == "manual":
        print(MANUAL_COMPACT_MESSAGE)
        if custom_instructions:
            print(f"🎯 Custom compact instructions: {custom_instructions}")
        print("✅ Ready for compact operation")
    else:  # auto
        print(AUTO_COMPACT_MESSAGE)
        print("✅ Auto-compact proceeding with full agent context")
    
    sys.exit(0)


if __name__ == "__main__":
    main()