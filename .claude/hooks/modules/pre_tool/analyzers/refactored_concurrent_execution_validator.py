#!/usr/bin/env python3
"""Concurrent Execution Validator for Claude Code - Refactored Version.

Enforces concurrent execution patterns and prevents sequential operations.
This is a refactored version using the new base validator classes.
"""

from typing import Dict, Any, Optional
from .base_validators import BatchingValidator, ToolSpecificValidator
from ..core.workflow_validator import (
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class RefactoredConcurrentExecutionValidator(BatchingValidator):
    """Enforces concurrent execution patterns using base class functionality."""
    
    def __init__(self, priority: int = 875):
        super().__init__(priority)
        self.min_batch_size = 5
        self.optimal_batch_size = 10
        
        # Tool-specific thresholds
        self.thresholds = {
            "TodoWrite": {"min": 5, "priority": 95},
            "Task": {"min": 2, "priority": 90},
            "file_ops": {"min": 2, "priority": 80},
            "Bash": {"min": 2, "priority": 75}
        }
    
    def get_validator_name(self) -> str:
        return "refactored_concurrent_execution_validator"
    
    def _validate_workflow_impl(self, tool_name: str, tool_input: Dict[str, Any], 
                               context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate for concurrent execution patterns."""
        
        # Check TodoWrite batching
        if tool_name == "TodoWrite":
            return self._validate_todo_batching(tool_input)
        
        # Check Task spawning patterns
        elif tool_name == "Task":
            return self._validate_task_spawning(context)
        
        # Check file operations
        elif tool_name in ["Read", "Write", "Edit"]:
            return self._validate_file_operations(tool_name, context)
        
        # Check Bash commands
        elif tool_name == "Bash":
            return self._validate_bash_commands(context)
        
        return None
    
    def _validate_todo_batching(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate TodoWrite batching."""
        todos = tool_input.get("todos", [])
        threshold = self.thresholds["TodoWrite"]
        
        if len(todos) < threshold["min"]:
            return self.create_warning_result(
                message="🚨 BATCH VIOLATION: TodoWrite should include 5-10+ todos in ONE call!",
                violation_type=WorkflowViolationType.FRAGMENTED_WORKFLOW,
                alternative="Batch ALL todos together - status updates, new todos, completions in ONE TodoWrite call",
                guidance="✅ CORRECT: TodoWrite { todos: [5-10+ todos with all statuses/priorities] }",
                priority=threshold["priority"]
            )
        
        return None
    
    def _validate_task_spawning(self, context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate Task spawning patterns."""
        recent_tools = getattr(context, '_recent_tools', [])[-3:]
        task_count = sum(1 for tool in recent_tools if tool == "Task")
        threshold = self.thresholds["Task"]
        
        if task_count >= threshold["min"]:
            return self.create_warning_result(
                message="⚡ CONCURRENT EXECUTION REQUIRED: Spawn ALL agents in ONE message!",
                violation_type=WorkflowViolationType.FRAGMENTED_WORKFLOW,
                alternative="Batch all Task calls together in a single message for parallel execution",
                guidance="✅ Use multiple Task calls in ONE message:\n  Task('Agent 1', 'instructions', 'type-1')\n  Task('Agent 2', 'instructions', 'type-2')\n  Task('Agent 3', 'instructions', 'type-3')",
                priority=threshold["priority"]
            )
        
        return None
    
    def _validate_file_operations(self, tool_name: str, context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate file operation batching."""
        recent_tools = getattr(context, '_recent_tools', [])[-3:]
        file_op_count = sum(1 for tool in recent_tools if tool in ["Read", "Write", "Edit"])
        threshold = self.thresholds["file_ops"]
        
        if file_op_count >= threshold["min"]:
            return self.create_suggestion_result(
                message="📦 BATCH FILE OPERATIONS: Combine multiple file operations in ONE message!",
                alternative="Group all related file operations together for parallel execution",
                guidance="Read 10 files? → One message with 10 Read calls\nWrite 5 files? → One message with 5 Write calls",
                priority=threshold["priority"]
            )
        
        return None
    
    def _validate_bash_commands(self, context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate Bash command batching."""
        recent_tools = getattr(context, '_recent_tools', [])[-3:]
        bash_count = sum(1 for tool in recent_tools if tool == "Bash")
        threshold = self.thresholds["Bash"]
        
        if bash_count >= threshold["min"]:
            return self.create_suggestion_result(
                message="🖥️ BATCH BASH COMMANDS: Group terminal commands in ONE message!",
                alternative="Combine related bash commands for parallel execution",
                guidance="Multiple directories? → One message with all mkdir commands\nInstall + test + build? → One message with all npm commands",
                priority=threshold["priority"]
            )
        
        return None