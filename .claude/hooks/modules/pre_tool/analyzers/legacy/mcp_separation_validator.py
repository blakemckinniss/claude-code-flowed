#!/usr/bin/env python3
"""MCP Tool Separation Validator for Claude Code.

Enforces the critical separation between MCP coordination tools
and Claude Code execution tools.
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


class MCPSeparationValidator(HiveWorkflowValidator):
    """Enforces MCP tool vs Claude Code execution separation."""
    
    def get_validator_name(self) -> str:
        return "mcp_separation_validator"
    
    def __init__(self, priority: int = 95):
        super().__init__(priority)
        
        # MCP tools that should NEVER do execution
        self.mcp_coordination_tools = {
            "mcp__claude-flow__": "coordination and planning",
            "mcp__zen__": "orchestration and analysis",
            "mcp__filesystem__": "file system navigation",
            "mcp__github__": "GitHub API operations"
        }
        
        # Claude Code execution tools
        self.execution_tools = {
            "Read": "file reading",
            "Write": "file writing",
            "Edit": "file editing",
            "MultiEdit": "multiple file edits",
            "Bash": "command execution",
            "TodoWrite": "task management",
            "Task": "agent spawning",
            "Grep": "file searching",
            "Glob": "pattern matching"
        }
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate tool usage separation."""
        
        # Check for MCP execution attempts
        if self._detects_mcp_execution_attempt(tool_name, tool_input):
            return ValidationResult(
                severity=ValidationSeverity.CRITICAL,
                violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                message="🚨 CRITICAL SEPARATION VIOLATION: MCP tools NEVER execute!",
                suggested_alternative=self._get_correct_tool_suggestion(tool_name, tool_input),
                blocking_reason="MCP tools cannot perform execution operations",
                hive_guidance=self._get_separation_guidance(),
                priority_score=100
            )
        
        # Warn about workflow patterns
        if tool_name.startswith("mcp__") and self._is_execution_context(context):
            return ValidationResult(
                severity=ValidationSeverity.WARN,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="⚠️ Remember: MCP coordinates, Claude Code executes!",
                suggested_alternative=None,
                hive_guidance=self._get_workflow_pattern(),
                priority_score=85
            )
        
        # Check for common mistakes
        if tool_name in self.execution_tools and self._should_use_mcp_first(tool_name, context):
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.MISSING_COORDINATION,
                message="💡 Consider using MCP coordination before execution",
                suggested_alternative=self._get_mcp_suggestion(tool_name),
                hive_guidance="Initialize swarm → Spawn agents → Execute with Claude Code tools",
                priority_score=60
            )
        
        return None
    
    def _detects_mcp_execution_attempt(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Detect if MCP tool is attempting execution."""
        if not tool_name.startswith("mcp__"):
            return False
        
        # Check for execution-related operations in MCP tools
        dangerous_operations = [
            "terminal_execute", "write_file", "create_file",
            "execute_code", "run_command", "build", "compile"
        ]
        
        # Check tool name
        for op in dangerous_operations:
            if op in tool_name.lower():
                return True
        
        # Check tool input for execution attempts
        input_str = str(tool_input).lower()
        for op in dangerous_operations:
            if op in input_str:
                return True
        
        return False
    
    def _is_execution_context(self, context: WorkflowContextTracker) -> bool:
        """Check if we're in an execution context."""
        recent_tools = context._recent_tools[-5:] if hasattr(context, '_recent_tools') else []
        execution_count = sum(1 for tool in recent_tools if not tool.startswith("mcp__"))
        return execution_count > 2
    
    def _should_use_mcp_first(self, tool_name: str, context: WorkflowContextTracker) -> bool:
        """Check if MCP coordination would be beneficial."""
        # Complex operations benefit from coordination
        if tool_name == "Task" and context.get_tools_since_flow() > 10:
            return True
        if tool_name == "TodoWrite" and context.get_coordination_state() == "disconnected":
            return True
        return False
    
    def _get_correct_tool_suggestion(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Get the correct tool to use instead."""
        if "file" in tool_name.lower() or "write" in tool_name.lower():
            return "Use Claude Code's Write, Edit, or MultiEdit tools for file operations"
        elif "execute" in tool_name.lower() or "terminal" in tool_name.lower():
            return "Use Claude Code's Bash tool for command execution"
        elif "todo" in tool_name.lower():
            return "Use Claude Code's TodoWrite tool for task management"
        else:
            return "Use Claude Code execution tools for actual implementation"
    
    def _get_mcp_suggestion(self, tool_name: str) -> str:
        """Get appropriate MCP coordination suggestion."""
        if tool_name == "Task":
            return "Consider: mcp__claude-flow__swarm_init → agent_spawn → task_orchestrate"
        elif tool_name == "TodoWrite":
            return "Consider: mcp__claude-flow__memory_usage to store task context"
        else:
            return "Consider using MCP tools for coordination before execution"
    
    def _get_separation_guidance(self) -> str:
        """Get separation guidance."""
        return """🎯 CLAUDE CODE IS THE ONLY EXECUTOR

✅ Claude Code ALWAYS Handles:
- 🔧 ALL file operations (Read, Write, Edit, MultiEdit)
- 💻 ALL code generation and programming
- 🖥️ ALL bash commands and system operations
- 📝 ALL TodoWrite and task management
- 🔄 ALL git operations

🧠 MCP Tools ONLY Handle:
- 🎯 Coordination only - Planning actions
- 💾 Memory management - Storing context
- 🤖 Neural features - Learning patterns
- 📊 Performance tracking - Monitoring
- 🐝 Swarm orchestration - Coordination"""
    
    def _get_workflow_pattern(self) -> str:
        """Get the correct workflow pattern."""
        return """🔄 CORRECT WORKFLOW PATTERN:

1. MCP: swarm_init (coordination setup)
2. MCP: agent_spawn (planning agents)
3. MCP: task_orchestrate (task coordination)
4. Claude Code: Task tool to spawn agents
5. Claude Code: TodoWrite with ALL todos batched
6. Claude Code: Read, Write, Edit, Bash (actual work)
7. MCP: memory_usage (store results)"""