#!/usr/bin/env python3
"""Visual Formats Validator - Refactored Version.

Provides consistent visual formatting templates for progress tracking.
"""

from typing import Dict, Any, Optional
from .base_validators import VisualFormatProvider, ToolSpecificValidator
from ..core.workflow_validator import (
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class RefactoredVisualFormatsValidator(VisualFormatProvider):
    """Provides visual format templates using base class functionality."""
    
    def __init__(self, priority: int = 650):
        super().__init__(priority)
        self._initialize_templates()
    
    def get_validator_name(self) -> str:
        return "refactored_visual_formats_validator"
    
    def _initialize_templates(self) -> None:
        """Initialize all visual format templates."""
        
        # Task progress format
        self.add_format_template("task_progress", """📊 VISUAL TASK TRACKING FORMAT:

📊 Progress Overview
   ├── Total Tasks: X
   ├── ✅ Completed: X (X%)
   ├── 🔄 In Progress: X (X%)
   ├── ⭕ Todo: X (X%)
   └── ❌ Blocked: X (X%)

📋 Todo (X)
   └── 🔴 001: [Task description] [PRIORITY] ▶

🔄 In progress (X)
   ├── 🟡 002: [Task description] ↳ X deps ▶
   └── 🔴 003: [Task description] [PRIORITY] ▶

✅ Completed (X)
   ├── ✅ 004: [Task description]
   └── ... (more completed tasks)

Priority: 🔴 HIGH, 🟡 MEDIUM, 🟢 LOW""")

        # Swarm status format
        self.add_format_template("swarm_status", """🎨 VISUAL SWARM STATUS FORMAT:

🐝 Swarm Status: ACTIVE
├── 🏗️ Topology: hierarchical
├── 👥 Agents: 6/8 active
├── ⚡ Mode: parallel execution
├── 📊 Tasks: 12 total (4 complete, 6 in-progress, 2 pending)
└── 🧠 Memory: 15 coordination points stored

Agent Activity:
├── 🟢 architect: Designing database schema...
├── 🟢 coder-1: Implementing auth endpoints...
├── 🟢 coder-2: Building user CRUD operations...
├── 🟢 analyst: Optimizing query performance...
├── 🟡 tester: Waiting for auth completion...
└── 🟢 coordinator: Monitoring progress...

Status: 🟢 Active, 🟡 Waiting, 🔴 Error, ⚫ Not started""")

        # Memory coordination format
        self.add_format_template("memory_coordination", """🔄 MEMORY COORDINATION PATTERN:

// After major decision/implementation
mcp__claude-flow__memory_usage {
  action: "store",
  key: "swarm-{id}/agent-{name}/{step}",
  value: {
    timestamp: Date.now(),
    decision: "what was decided",
    implementation: "what was built",
    nextSteps: ["step1", "step2"],
    dependencies: ["dep1", "dep2"]
  }
}

// To retrieve coordination data
mcp__claude-flow__memory_usage {
  action: "retrieve",
  key: "swarm-{id}/agent-{name}/{step}"
}

// To check all swarm progress
mcp__claude-flow__memory_usage {
  action: "list",
  pattern: "swarm-{id}/*"
}""")
    
    def _validate_workflow_impl(self, tool_name: str, tool_input: Dict[str, Any], 
                               context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Suggest visual formats when appropriate."""
        
        # TodoWrite operations should use proper visual format
        if tool_name == "TodoWrite":
            return self._validate_todo_format(tool_input)
        
        # Swarm status operations
        elif tool_name in ["mcp__claude-flow__swarm_status", "mcp__claude-flow__swarm_monitor"]:
            return self._suggest_swarm_format()
        
        # Memory coordination operations
        elif tool_name == "mcp__claude-flow__memory_usage":
            return self._suggest_memory_format(tool_input)
        
        return None
    
    def _validate_todo_format(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate TodoWrite format."""
        todos = tool_input.get("todos", [])
        
        if len(todos) > 3 and not self._has_proper_format(todos):
            return self.create_format_suggestion(
                message="📊 Use visual task tracking format for better clarity",
                template_name="task_progress",
                priority=50
            )
        
        return None
    
    def _suggest_swarm_format(self) -> ValidationResult:
        """Suggest swarm status format."""
        return self.create_format_suggestion(
            message="🎨 Use visual swarm status format for monitoring",
            template_name="swarm_status",
            priority=40
        )
    
    def _suggest_memory_format(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Suggest memory coordination format."""
        action = tool_input.get("action", "")
        
        if action in ["store", "retrieve"]:
            return self.create_format_suggestion(
                message="🔄 Follow memory coordination pattern for consistency",
                template_name="memory_coordination",
                priority=35
            )
        
        return None
    
    def _has_proper_format(self, todos: list) -> bool:
        """Check if todos have proper formatting attributes."""
        required_attrs = ["id", "content", "status", "priority"]
        return all(
            all(attr in todo for attr in required_attrs)
            for todo in todos
        )