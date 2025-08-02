#!/usr/bin/env python3
"""Claude-Flow Command Suggester Validator - Refactored Version.

Analyzes Bash tool usage and suggests claude-flow alternatives when appropriate
using base class functionality.
"""

from typing import Dict, Any, Optional
from .base_validators import TaskAnalysisValidator, PatternMatchingValidator
from ..core.workflow_validator import (
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)
from ...utils.process_manager import (
    suggest_claude_flow_for_command,
    ClaudeFlowIntegration
)


class RefactoredClaudeFlowSuggesterValidator(TaskAnalysisValidator):
    """Suggests claude-flow commands using base class functionality."""
    
    def __init__(self, priority: int = 500):
        super().__init__(priority)
        self.claude_flow = ClaudeFlowIntegration()
        self._initialize_patterns()
    
    def get_validator_name(self) -> str:
        return "refactored_claude_flow_suggester"
    
    def _initialize_patterns(self) -> None:
        """Initialize task patterns using base class method."""
        
        # Add task patterns for detection
        self.add_task_pattern('api_development', [
            r'build.*api', r'create.*api', r'develop.*api'
        ])
        self.add_task_pattern('code_review', [
            r'review.*code', r'analyze.*code', r'audit.*code'
        ])
        self.add_task_pattern('security_audit', [
            r'security.*audit', r'security.*scan', r'vulnerability'
        ])
        self.add_task_pattern('performance_optimization', [
            r'optimize.*performance', r'improve.*speed', r'reduce.*latency'
        ])
        self.add_task_pattern('testing', [
            r'run.*test', r'execute.*test', r'test.*suite'
        ])
        self.add_task_pattern('sparc_tdd', [
            r'sparc.*tdd', r'test.*driven'
        ])
        self.add_task_pattern('sparc_dev', [
            r'sparc.*dev', r'sparc.*development'
        ])
    
    def _validate_workflow_impl(self, tool_name: str, tool_input: Dict[str, Any], 
                               context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Analyze Bash commands and suggest claude-flow alternatives."""
        
        if tool_name != "Bash":
            return None
        
        command = tool_input.get("command", "")
        if not command or "claude-flow" in command:
            return None
        
        # Parse command into list format
        try:
            command_parts = command.split()
        except Exception:
            return None
        
        # Check for optimization opportunities
        suggestions = suggest_claude_flow_for_command(command_parts)
        
        if suggestions:
            return self._create_suggestion_response(command, tool_input, suggestions)
        
        return None
    
    def _create_suggestion_response(self, command: str, tool_input: Dict[str, Any], 
                                  suggestions: Dict[str, Any]) -> ValidationResult:
        """Create suggestion response using base class functionality."""
        
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
        
        # Check command description for task context using base class method
        description = tool_input.get("description", "")
        task_type = self.detect_task_type(description)
        
        if task_type:
            cf_suggestion = self._get_claude_flow_suggestion(task_type, description)
            if cf_suggestion:
                cmd_list, desc = cf_suggestion
                message_parts.extend([
                    "",
                    "   🐝 Based on your task description, consider:",
                    f"      {' '.join(cmd_list)}",
                    f"      Purpose: {desc}"
                ])
        
        return self.create_suggestion_result(
            message="\n".join(message_parts),
            guidance="Consider using claude-flow for enhanced workflow optimization",
            priority=50
        )
    
    def _get_claude_flow_suggestion(self, task_type: str, description: str) -> Optional[tuple]:
        """Get claude-flow suggestion for task type."""
        context = {
            'task_type': task_type,
            'description': description,
            'parameters': {}
        }
        
        return self.claude_flow.suggest_command(context)


# Helper function for external use
def suggest_claude_flow_for_task(task: str) -> Optional[list]:
    """
    Helper function to get claude-flow command for a task.
    
    Args:
        task: Task description
        
    Returns:
        Command list or None
    """
    cf = ClaudeFlowIntegration()
    
    # Simple pattern matching for task type
    task_lower = task.lower()
    
    if any(pattern in task_lower for pattern in ['api', 'rest', 'endpoint']):
        task_type = 'api_development'
    elif any(pattern in task_lower for pattern in ['test', 'spec', 'unit']):
        task_type = 'testing'
    elif any(pattern in task_lower for pattern in ['review', 'analyze', 'audit']):
        task_type = 'code_review'
    elif any(pattern in task_lower for pattern in ['optimize', 'performance', 'speed']):
        task_type = 'performance_optimization'
    else:
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