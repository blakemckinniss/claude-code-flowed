"""Utility functions for the hook system."""

from .helpers import escape_json, validate_prompt, log_debug
from .custom_patterns import CustomPatternLoader

__all__ = ['CustomPatternLoader', 'escape_json', 'log_debug', 'validate_prompt']