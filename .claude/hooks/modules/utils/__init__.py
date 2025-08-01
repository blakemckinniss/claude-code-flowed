"""Utility functions for the hook system."""

from .helpers import escape_json, validate_prompt, log_debug
from .custom_patterns import CustomPatternLoader

__all__ = ['escape_json', 'validate_prompt', 'log_debug', 'CustomPatternLoader']