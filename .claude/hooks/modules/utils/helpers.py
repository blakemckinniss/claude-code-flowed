"""Helper utilities for the hook system."""

import json
import sys
import os

# Path setup handled by centralized resolver when importing this module
from typing import Any, Optional


def escape_json(text: str) -> str:
    """Escape text for JSON output."""
    return json.dumps(text)[1:-1]  # Remove quotes


def validate_prompt(prompt: str) -> str:
    """Validate and clean prompt input."""
    if not prompt:
        return ""
    
    # Remove any null bytes or control characters
    cleaned = prompt.replace('\0', '').strip()
    
    # Limit length to prevent abuse
    max_length = 10000
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned


def log_debug(message: str, data: Optional[Any] = None) -> None:
    """Log debug information to stderr if debugging is enabled."""
    if os.environ.get('CLAUDE_HOOKS_DEBUG', '').lower() in ('1', 'true', 'yes'):
        debug_msg = f"[HOOK DEBUG] {message}"
        if data is not None:
            debug_msg += f"\nData: {json.dumps(data, indent=2)}"
        print(debug_msg, file=sys.stderr)