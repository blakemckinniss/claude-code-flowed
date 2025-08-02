"""Pre-tool analysis manager - delegates to SRP-compliant components.

REFACTORED: This manager now uses Single Responsibility Principle components:
- ValidatorRegistry: Manages validator initialization and registration
- ValidationCoordinator: Orchestrates validation execution  
- SlimmedPreToolAnalysisManager: Focuses on configuration and result processing

Original 804-line manager refactored into focused, maintainable components.
"""

# Import the new SRP-compliant manager
from .components.slimmed_manager import SlimmedPreToolAnalysisManager, GuidanceOutputHandler


# Legacy compatibility: Expose SlimmedPreToolAnalysisManager as PreToolAnalysisManager
# This maintains backward compatibility with existing pre_tool_use.py hook
PreToolAnalysisManager = SlimmedPreToolAnalysisManager


# Legacy compatibility: Expose GuidanceOutputHandler and debug classes
# to maintain compatibility with existing hook infrastructure
class DebugValidationReporter:
    """Provides detailed validation reports for debugging."""  
    
    def __init__(self, manager: SlimmedPreToolAnalysisManager):
        self.manager = manager
    
    def generate_debug_report(self, tool_name: str, tool_input: dict) -> str:
        """Generate detailed debug report for troubleshooting."""
        return self.manager.get_debug_report(tool_name, tool_input)
    
    def log_debug_info(self, tool_name: str, tool_input: dict) -> None:
        """Log debug information if debugging is enabled."""
        import os
        debug_env = os.environ.get("CLAUDE_HOOKS_DEBUG", "").lower()
        if debug_env in ["true", "1", "on"]:
            report = self.generate_debug_report(tool_name, tool_input)
            import sys
            print(f"DEBUG: {report}", file=sys.stderr)


# Export for backward compatibility
__all__ = [
    'DebugValidationReporter',
    'GuidanceOutputHandler',
    'PreToolAnalysisManager',
    'SlimmedPreToolAnalysisManager'
]