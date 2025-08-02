#!/usr/bin/env python3
"""Visual Formats Validator for Claude Code.

Provides consistent visual formatting templates for progress tracking.
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


class VisualFormatsValidator(HiveWorkflowValidator):
    """Provides visual format templates and guidance."""
    
    def get_validator_name(self) -> str:
        return "visual_formats_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Suggest visual formats when appropriate."""
        
        # TodoWrite operations should use proper visual format
        if tool_name == "TodoWrite":
            todos = tool_input.get("todos", [])
            if len(todos) > 3 and not self._has_proper_format(todos):
                return ValidationResult(
                    severity=ValidationSeverity.SUGGEST,
                    violation_type=None,
                    message="📊 Use visual task tracking format for better clarity",
                    suggested_alternative=None,
                    hive_guidance=self._get_task_progress_format(),
                    priority_score=50
                )
        
        # Swarm status operations
        if tool_name in ["mcp__claude-flow__swarm_status", "mcp__claude-flow__swarm_monitor"]:
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=None,
                message="🎨 Use visual swarm status format for monitoring",
                suggested_alternative=None,
                hive_guidance=self._get_swarm_status_format(),
                priority_score=40
            )
        
        # Memory coordination operations
        if tool_name == "mcp__claude-flow__memory_usage":
            action = tool_input.get("action", "")
            if action in ["store", "retrieve"]:
                return ValidationResult(
                    severity=ValidationSeverity.SUGGEST,
                    violation_type=None,
                    message="🔄 Follow memory coordination pattern for consistency",
                    suggested_alternative=None,
                    hive_guidance=self._get_memory_format(),
                    priority_score=35
                )
        
        return None
    
    def _has_proper_format(self, todos: list) -> bool:
        """Check if todos have proper formatting attributes."""
        required_attrs = ["id", "content", "status", "priority"]
        return all(
            all(attr in todo for attr in required_attrs)
            for todo in todos
        )
    
    def _get_task_progress_format(self) -> str:
        """Get task progress visual format template."""
        return """📊 VISUAL TASK TRACKING FORMAT:

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

Priority: 🔴 HIGH, 🟡 MEDIUM, 🟢 LOW"""
    
    def _get_swarm_status_format(self) -> str:
        """Get swarm status visual format template."""
        return """🎨 VISUAL SWARM STATUS FORMAT:

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

Status: 🟢 Active, 🟡 Waiting, 🔴 Error, ⚫ Not started"""
    
    def _get_memory_format(self) -> str:
        """Get memory coordination format."""
        return """🔄 MEMORY COORDINATION PATTERN:

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
}"""