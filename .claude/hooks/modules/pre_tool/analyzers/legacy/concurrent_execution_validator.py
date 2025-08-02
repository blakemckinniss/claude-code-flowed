#!/usr/bin/env python3
"""Concurrent Execution Validator for Claude Code.

Enforces concurrent execution patterns and prevents sequential operations.
"""

import re
from typing import Dict, Any, Optional
from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class ConcurrentExecutionValidator(HiveWorkflowValidator):
    """Enforces concurrent execution patterns."""
    
    def get_validator_name(self) -> str:
        return "concurrent_execution_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate for concurrent execution patterns."""
        
        # Check TodoWrite patterns
        if tool_name == "TodoWrite":
            todos = tool_input.get("todos", [])
            if len(todos) < 5:
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.FRAGMENTED_WORKFLOW,
                    message="🚨 BATCH VIOLATION: TodoWrite should include 5-10+ todos in ONE call!",
                    suggested_alternative="Batch ALL todos together - status updates, new todos, completions in ONE TodoWrite call",
                    hive_guidance="✅ CORRECT: TodoWrite { todos: [5-10+ todos with all statuses/priorities] }",
                    priority_score=95
                )
        
        # Check for sequential Task spawning patterns
        if tool_name == "Task":
            # Look at recent tools to detect sequential pattern
            recent_tools = context._recent_tools[-3:] if hasattr(context, '_recent_tools') else []
            task_count = sum(1 for tool in recent_tools if tool == "Task")
            
            if task_count >= 2:
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.FRAGMENTED_WORKFLOW,
                    message="⚡ CONCURRENT EXECUTION REQUIRED: Spawn ALL agents in ONE message!",
                    suggested_alternative="Batch all Task calls together in a single message for parallel execution",
                    hive_guidance="✅ Use multiple Task calls in ONE message:\n  Task('Agent 1', 'instructions', 'type-1')\n  Task('Agent 2', 'instructions', 'type-2')\n  Task('Agent 3', 'instructions', 'type-3')",
                    priority_score=90
                )
        
        # Check for sequential file operations
        if tool_name in ["Read", "Write", "Edit"]:
            recent_tools = context._recent_tools[-3:] if hasattr(context, '_recent_tools') else []
            file_op_count = sum(1 for tool in recent_tools if tool in ["Read", "Write", "Edit"])
            
            if file_op_count >= 2:
                return ValidationResult(
                    severity=ValidationSeverity.SUGGEST,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message="📦 BATCH FILE OPERATIONS: Combine multiple file operations in ONE message!",
                    suggested_alternative="Group all related file operations together for parallel execution",
                    hive_guidance="Read 10 files? → One message with 10 Read calls\nWrite 5 files? → One message with 5 Write calls",
                    priority_score=80
                )
        
        # Check for sequential Bash commands
        if tool_name == "Bash":
            recent_tools = context._recent_tools[-3:] if hasattr(context, '_recent_tools') else []
            bash_count = sum(1 for tool in recent_tools if tool == "Bash")
            
            if bash_count >= 2:
                return ValidationResult(
                    severity=ValidationSeverity.SUGGEST,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message="🖥️ BATCH BASH COMMANDS: Group terminal commands in ONE message!",
                    suggested_alternative="Combine related bash commands for parallel execution",
                    hive_guidance="Multiple directories? → One message with all mkdir commands\nInstall + test + build? → One message with all npm commands",
                    priority_score=75
                )
        
        return None