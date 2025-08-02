#!/usr/bin/env python3
"""
Claude-Flow Command Suggester Validator

Analyzes Bash tool usage and suggests claude-flow alternatives when appropriate.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import re

from ...utils.process_manager import (
    suggest_claude_flow_for_command,
    ClaudeFlowIntegration
)


class ClaudeFlowSuggesterValidator:
    """
    Suggests claude-flow commands as alternatives to regular bash commands.
    
    Priority: 500 (runs after critical validators but before most others)
    """
    
    PRIORITY = 500
    
    def __init__(self):
        self.name = "claude_flow_suggester"
        self.enabled = True
        self.claude_flow = ClaudeFlowIntegration()
        
        # Pattern matching for task descriptions
        self.task_patterns = {
            r'build.*api|create.*api|develop.*api': 'api_development',
            r'review.*code|analyze.*code|audit.*code': 'code_review',
            r'security.*audit|security.*scan|vulnerability': 'security_audit',
            r'optimize.*performance|improve.*speed|reduce.*latency': 'performance_optimization',
            r'run.*test|execute.*test|test.*suite': 'testing',
            r'sparc.*tdd|test.*driven': 'sparc_tdd',
            r'sparc.*dev|sparc.*development': 'sparc_dev'
        }
    
    def validate(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Analyze Bash commands and suggest claude-flow alternatives.
        
        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool
            
        Returns:
            Tuple of (allow, message, metadata)
        """
        if not self.enabled or tool_name != "Bash":
            return True, None, None
        
        command = tool_args.get("command", "")
        if not command:
            return True, None, None
        
        # Parse command into list format
        try:
            # Simple split - could be enhanced with shlex for complex commands
            command_parts = command.split()
        except Exception:
            return True, None, None
        
        # Skip if already a claude-flow command
        if "claude-flow" in command:
            return True, None, None
        
        # Check for optimization opportunities
        suggestions = suggest_claude_flow_for_command(command_parts)
        
        if suggestions:
            # Build informative message
            message_parts = [
                "💡 Claude-Flow Enhancement Available!",
                f"   Current command: {command}"
            ]
            
            for suggestion in suggestions['suggestions']:
                message_parts.extend([
                    f"   • {suggestion['suggestion']}",
                    f"     Benefit: {suggestion['benefit']}"
                ])
            
            message_parts.append("\n   To use claude-flow, replace your bash command with the suggested alternative.")
            
            # Check command description for task context
            description = tool_args.get("description", "")
            task_type = self._detect_task_type(description)
            
            if task_type:
                context = {
                    'task_type': task_type,
                    'description': description,
                    'parameters': {}
                }
                
                cf_suggestion = self.claude_flow.suggest_command(context)
                if cf_suggestion:
                    cmd_list, desc = cf_suggestion
                    message_parts.extend([
                        "",
                        "   🐝 Based on your task description, consider:",
                        f"      {' '.join(cmd_list)}",
                        f"      Purpose: {desc}"
                    ])
            
            # Allow the command but provide suggestions
            return True, "\n".join(message_parts), {
                "claude_flow_suggestions": suggestions,
                "task_type": task_type
            }
        
        return True, None, None
    
    def _detect_task_type(self, description: str) -> Optional[str]:
        """Detect task type from description."""
        if not description:
            return None
        
        desc_lower = description.lower()
        
        for pattern, task_type in self.task_patterns.items():
            if re.search(pattern, desc_lower):
                return task_type
        
        return None
    
    def get_config(self) -> Dict[str, Any]:
        """Get validator configuration."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "priority": self.PRIORITY,
            "description": "Suggests claude-flow commands for enhanced workflows"
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update validator configuration."""
        if "enabled" in config:
            self.enabled = bool(config["enabled"])


# Example usage in hooks
def suggest_claude_flow_for_task(task: str) -> Optional[List[str]]:
    """
    Helper function to get claude-flow command for a task.
    
    Args:
        task: Task description
        
    Returns:
        Command list or None
    """
    cf = ClaudeFlowIntegration()
    
    # Try to detect task type
    patterns = {
        r'api|rest|endpoint': 'api_development',
        r'test|spec|unit': 'testing',
        r'review|analyze|audit': 'code_review',
        r'optimize|performance|speed': 'performance_optimization'
    }
    
    task_lower = task.lower()
    task_type = None
    
    for pattern, ttype in patterns.items():
        if re.search(pattern, task_lower):
            task_type = ttype
            break
    
    if not task_type:
        task_type = 'development'  # Default
    
    context = {
        'task_type': task_type,
        'description': task,
        'parameters': {'task': task}
    }
    
    suggestion = cf.suggest_command(context)
    if suggestion:
        return suggestion[0]  # Return command list
    
    return None