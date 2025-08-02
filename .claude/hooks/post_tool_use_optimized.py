#\!/usr/bin/env python3
"""Ultra-Fast PostToolUse hook optimized for sub-100ms performance.

This optimized version focuses on:
- Lightning-fast analysis with intelligent caching
- Async execution pools for parallel processing
- Circuit breakers to prevent system overload  
- Zero-blocking behavior with fallbacks
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Import the lightning-fast processor
try:
    from modules.optimization.lightning_fast_processor import (
        get_lightning_processor,
        process_hook_fast,
        performance_timer
    )
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

# Import existing fallbacks
try:
    from modules.utils.process_manager import managed_subprocess_run
    PROCESS_MANAGER_AVAILABLE = True
except ImportError:
    PROCESS_MANAGER_AVAILABLE = False


def check_hook_file_violations_fast(tool_name: str, tool_input: Dict[str, Any]) -> Optional[str]:
    """Ultra-fast hook file violation checker."""
    # Only check file operations
    if tool_name not in {"Write", "Edit", "MultiEdit"}:
        return None
    
    # Extract file path
    file_path = tool_input.get("file_path") or tool_input.get("path", "")
    if not file_path:
        return None
    
    # Quick path check
    if ".claude/hooks" not in file_path or not file_path.endswith(".py"):
        return None
    
    # Skip path_resolver.py - it's allowed to use sys.path
    if "path_resolver.py" in file_path:
        return None
    
    # Get content for violation check
    content = ""
    if tool_name == "Write":
        content = tool_input.get("content", "")
    elif tool_name == "Edit":
        content = tool_input.get("new_string", "")
    elif tool_name == "MultiEdit":
        edits = tool_input.get("edits", [])
        content = "\n".join(edit.get("new_string", "") for edit in edits)
    
    # Fast sys.path check
    if "sys.path." in content:
        return f"""
{'='*70}
🚨 HOOK FILE VIOLATION DETECTED
{'='*70}
❌ File: {os.path.relpath(file_path, '/home/devcontainers/flowed')}
❌ Violation: sys.path manipulations are not allowed in hook files

✅ CORRECT APPROACH:
   All hook files should use centralized path management:
   from modules.utils.path_resolver import setup_hook_paths
   setup_hook_paths()

📖 See: .claude/hooks/PATH_MANAGEMENT.md for details
{'='*70}
"""
    
    return None


def run_ruff_check_fast(tool_name: str, tool_input: Dict[str, Any]) -> Optional[str]:
    """Fast Ruff code quality check."""
    if not PROCESS_MANAGER_AVAILABLE:
        return None
        
    # Only check Python file operations
    if tool_name not in {"Write", "Edit", "MultiEdit"}:
        return None

    file_path = tool_input.get("file_path") or tool_input.get("path", "")
    if not file_path or not file_path.endswith(".py"):
        return None

    # Convert to absolute path
    if not os.path.isabs(file_path):
        file_path = os.path.join("/home/devcontainers/flowed", file_path)

    if not os.path.exists(file_path):
        return None

    try:
        # Run Ruff with minimal timeout
        result = managed_subprocess_run(
            ["ruff", "check", file_path, "--output-format=json"],
            check=False, capture_output=True,
            text=True,
            timeout=3,  # Reduced timeout for speed
            max_memory_mb=25,  # Reduced memory limit
            cwd="/home/devcontainers/flowed",
            tags={"hook": "post-tool-fast", "type": "ruff-check"}
        )

        if result.stdout:
            try:
                issues = json.loads(result.stdout)
                if issues:
                    # Generate fast feedback
                    total_issues = len(issues)
                    errors = [i for i in issues if i.get("code", "").startswith("E")]
                    security = [i for i in issues if i.get("code", "").startswith("S")]
                    
                    severity = "🚨 CRITICAL" if security else "❌ ERROR" if errors else "💡 STYLE"
                    
                    return f"""
{'='*50}
🔧 RUFF FEEDBACK - {severity} ({total_issues} issues)
File: {os.path.relpath(file_path, '/home/devcontainers/flowed')}
{'='*50}

💡 QUICK FIXES:
  • ruff check {os.path.relpath(file_path, '/home/devcontainers/flowed')} --fix
  • ruff format {os.path.relpath(file_path, '/home/devcontainers/flowed')}
{'='*50}
"""
            except json.JSONDecodeError:
                pass

        return None

    except Exception:
        return None


def main():
    """Ultra-fast hook handler optimized for sub-100ms execution."""
    with performance_timer("PostToolUse Hook"):
        try:
            # Read input with timeout
            input_data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
            sys.exit(1)

        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        tool_response = input_data.get("tool_response", {})
        
        # Skip analysis for certain tools (fast exit)
        skip_tools = {"TodoWrite", "Glob", "LS"}
        if tool_name in skip_tools:
            sys.exit(0)

        # Ultra-fast hook file violation check
        hook_violation = check_hook_file_violations_fast(tool_name, tool_input)
        if hook_violation:
            print(hook_violation, file=sys.stderr)
            sys.exit(1)  # Block violating operations

        # Lightning-fast processing if available
        if LIGHTNING_AVAILABLE:
            try:
                guidance_message, processing_time_ms = process_hook_fast(
                    tool_name, tool_input, tool_response
                )
                
                if guidance_message:
                    print(guidance_message, file=sys.stderr)
                    sys.exit(2)  # Provide guidance without blocking
                
                # Log performance if under debug
                if os.environ.get("CLAUDE_HOOKS_DEBUG"):
                    print(f"⚡ Lightning processing: {processing_time_ms:.2f}ms", file=sys.stderr)
                
                # Run Ruff check in parallel (non-blocking)
                ruff_feedback = run_ruff_check_fast(tool_name, tool_input)
                if ruff_feedback:
                    print(ruff_feedback, file=sys.stderr)
                    sys.exit(2)
                
                sys.exit(0)
                
            except Exception as e:
                if os.environ.get("CLAUDE_HOOKS_DEBUG"):
                    print(f"Lightning processing failed: {e}", file=sys.stderr)
                # Fall through to basic checks

        # Fallback to basic checks only
        basic_guidance = get_basic_guidance(tool_name, tool_input, tool_response)
        if basic_guidance:
            print(basic_guidance, file=sys.stderr)
            sys.exit(2)

        sys.exit(0)


def get_basic_guidance(tool_name: str, tool_input: Dict[str, Any], 
                      tool_response: Dict[str, Any]) -> Optional[str]:
    """Basic guidance for common patterns (fallback)."""
    # Task agent spawning guidance
    if tool_name == "Task":
        return """
💡 OPTIMIZATION OPPORTUNITY
Consider using ZEN coordination for complex tasks:
  - mcp__zen__planner for task breakdown
  - mcp__claude-flow__swarm_init for parallel execution
"""
    
    # Error handling guidance
    if not tool_response.get("success", True):
        error_msg = tool_response.get("error", "").lower()
        
        if "timeout" in error_msg:
            return """
⏰ TIMEOUT DETECTED
Consider:
  - Breaking down large operations
  - Using async/parallel processing
  - Implementing retries with backoff
"""
        elif "memory" in error_msg:
            return """
💾 MEMORY ISSUE DETECTED
Consider:
  - Processing data in chunks
  - Using streaming approaches
  - Implementing memory cleanup
"""
    
    return None


if __name__ == "__main__":
    main()
