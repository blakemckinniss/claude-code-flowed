"""Core workflow validation system for pre-tool analysis.

Validates adherence to Queen ZEN → Flow Workers → Storage Workers → Execution Drones
hierarchy and provides proactive guidance before tool execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import re


class ValidationSeverity(Enum):
    """Severity levels for workflow validation."""
    ALLOW = 0          # Tool is optimal - proceed
    SUGGEST = 1        # Tool is suboptimal - suggest improvement
    WARN = 2           # Tool may cause issues - provide warning
    BLOCK = 3          # Tool violates hive hierarchy - block execution
    CRITICAL = 4       # Tool would cause hive chaos - immediate intervention


class WorkflowViolationType(Enum):
    """Types of workflow violations."""
    BYPASSING_ZEN = "bypassing_zen"               # Skipping Queen ZEN entirely
    WORKER_INSUBORDINATION = "worker_insubordination"  # Workers acting independently  
    INEFFICIENT_EXECUTION = "inefficient_execution"    # Native tools when MCP available
    MISSING_COORDINATION = "missing_coordination"      # No hive memory/coordination
    DANGEROUS_OPERATION = "dangerous_operation"        # Operations that could harm hive
    FRAGMENTED_WORKFLOW = "fragmented_workflow"        # Scattered operations vs batched


@dataclass
class ValidationResult:
    """Result of workflow validation."""
    severity: ValidationSeverity
    violation_type: Optional[WorkflowViolationType]
    message: str
    suggested_alternative: Optional[str] = None
    blocking_reason: Optional[str] = None
    hive_guidance: Optional[str] = None
    priority_score: int = 0


class WorkflowContextTracker:
    """Tracks workflow context to understand current hive state."""
    
    def __init__(self):
        self._recent_tools: List[str] = []
        self._zen_last_used = -1
        self._flow_last_used = -1
        self._current_session_tools = 0
        self._coordination_state = "disconnected"  # disconnected, zen_active, flow_active, coordinated
        
    def add_tool_context(self, tool_name: str) -> None:
        """Add tool to context tracking."""
        self._recent_tools.append(tool_name)
        if len(self._recent_tools) > 10:  # Keep last 10 tools
            self._recent_tools.pop(0)
        
        self._current_session_tools += 1
        
        # Update coordination state
        if tool_name.startswith("mcp__zen__"):
            self._zen_last_used = self._current_session_tools
            self._coordination_state = "zen_active"
        elif tool_name.startswith("mcp__claude-flow__"):
            self._flow_last_used = self._current_session_tools
            if self._coordination_state == "zen_active":
                self._coordination_state = "coordinated"
            else:
                self._coordination_state = "flow_active"
        elif not tool_name.startswith("mcp__"):
            # Native tool - check if coordination is recent
            if self._is_coordination_recent():
                pass  # Keep current state
            else:
                self._coordination_state = "disconnected"
    
    def _is_coordination_recent(self, window: int = 5) -> bool:
        """Check if coordination is recent enough to be relevant."""
        recent_window = self._current_session_tools - window
        return (self._zen_last_used > recent_window or 
                self._flow_last_used > recent_window)
    
    def get_coordination_state(self) -> str:
        """Get current coordination state."""
        return self._coordination_state
    
    def get_tools_since_zen(self) -> int:
        """Get number of tools since last ZEN usage."""
        if self._zen_last_used == -1:
            return self._current_session_tools
        return self._current_session_tools - self._zen_last_used
    
    def get_tools_since_flow(self) -> int:
        """Get number of tools since last Flow usage."""
        if self._flow_last_used == -1:
            return self._current_session_tools
        return self._current_session_tools - self._flow_last_used
    
    def get_recent_pattern(self) -> str:
        """Get recent tool usage pattern."""
        return " → ".join(self._recent_tools[-5:])
    
    def has_zen_coordination(self, window: int = 5) -> bool:
        """Check if ZEN coordination is recent."""
        return self._zen_last_used > (self._current_session_tools - window)
    
    def has_flow_coordination(self, window: int = 5) -> bool:
        """Check if Flow coordination is recent."""
        return self._flow_last_used > (self._current_session_tools - window)


class HiveWorkflowValidator(ABC):
    """Base class for hive workflow validators."""
    
    def __init__(self, priority: int = 0):
        self.priority = priority
        self.context_tracker = WorkflowContextTracker()
    
    @abstractmethod
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate tool usage against hive workflow principles."""
        pass
    
    @abstractmethod
    def get_validator_name(self) -> str:
        """Get name of this validator."""
        pass
    
    def update_context(self, tool_name: str) -> None:
        """Update context tracking."""
        self.context_tracker.add_tool_context(tool_name)


class ZenHierarchyValidator(HiveWorkflowValidator):
    """Validates adherence to Queen ZEN hierarchy."""
    
    def get_validator_name(self) -> str:
        return "zen_hierarchy_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate ZEN hierarchy adherence."""
        
        # Allow ZEN tools - they're always appropriate
        if tool_name.startswith("mcp__zen__"):
            return ValidationResult(
                severity=ValidationSeverity.ALLOW,
                violation_type=None,
                message="👑 Queen ZEN command - optimal hive coordination",
                hive_guidance="Perfect! Queen ZEN leads the hive with supreme wisdom."
            )
        
        # Check for Flow Workers acting without ZEN command
        if tool_name.startswith("mcp__claude-flow__"):
            if not context.has_zen_coordination(3):
                return self._create_flow_insubordination_result(tool_name, context)
        
        # Check for direct Filesystem usage without coordination
        if tool_name.startswith("mcp__filesystem__"):
            if not context.has_zen_coordination(2) and not context.has_flow_coordination(2):
                return self._create_storage_bypass_result(tool_name, context)
        
        # Check for native tools without any coordination
        if not tool_name.startswith("mcp__"):
            return self._validate_native_tool_usage(tool_name, tool_input, context)
        
        return None
    
    def _create_flow_insubordination_result(self, tool_name: str, context: WorkflowContextTracker) -> ValidationResult:
        """Create result for Flow Worker acting without ZEN."""
        tools_since_zen = context.get_tools_since_zen()
        
        if tools_since_zen > 5:
            severity = ValidationSeverity.BLOCK
            message = f"🚨 HIVE VIOLATION: Flow Worker '{tool_name}' attempting independent action after {tools_since_zen} tools without Queen ZEN!"
            blocking_reason = "Flow Workers must receive royal commands from Queen ZEN before coordination"
        else:
            severity = ValidationSeverity.WARN
            message = f"⚠️ Flow Worker '{tool_name}' acting without recent Queen ZEN guidance"
        
        return ValidationResult(
            severity=severity,
            violation_type=WorkflowViolationType.WORKER_INSUBORDINATION,
            message=message,
            suggested_alternative="mcp__zen__chat { prompt: \"Guide Flow Worker coordination strategy\" }",
            blocking_reason=blocking_reason if severity == ValidationSeverity.BLOCK else None,
            hive_guidance="Queen ZEN must issue royal decrees before Flow Workers coordinate operations",
            priority_score=80 if severity == ValidationSeverity.BLOCK else 60
        )
    
    def _create_storage_bypass_result(self, tool_name: str, context: WorkflowContextTracker) -> ValidationResult:
        """Create result for Storage Worker bypassing hierarchy."""
        return ValidationResult(
            severity=ValidationSeverity.WARN,
            violation_type=WorkflowViolationType.BYPASSING_ZEN,
            message=f"📁 Storage Worker '{tool_name}' activated without hive coordination",
            suggested_alternative="mcp__zen__planner { step: \"Plan file operations with hive intelligence\" }",
            hive_guidance="Queen ZEN should coordinate Storage Worker operations for optimal efficiency",
            priority_score=40
        )
    
    def _validate_native_tool_usage(self, tool_name: str, tool_input: Dict[str, Any], 
                                  context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate native tool usage patterns."""
        coordination_state = context.get_coordination_state()
        tools_since_zen = context.get_tools_since_zen()
        
        # Check for complex operations without ZEN planning
        if self._is_complex_operation(tool_name, tool_input):
            if coordination_state == "disconnected":
                return ValidationResult(
                    severity=ValidationSeverity.BLOCK,
                    violation_type=WorkflowViolationType.BYPASSING_ZEN,
                    message=f"🚨 Complex operation '{tool_name}' requires Queen ZEN's strategic planning!",
                    suggested_alternative="mcp__zen__thinkdeep { step: \"Plan complex operation strategy\", thinking_mode: \"high\" }",
                    blocking_reason="Complex operations must have Queen ZEN's strategic oversight",
                    hive_guidance="Queen ZEN's deep analysis prevents chaos and ensures optimal results",
                    priority_score=90
                )
        
        # Check for extended native usage without coordination
        if tools_since_zen > 3 and coordination_state == "disconnected":
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.MISSING_COORDINATION,
                message=f"🐝 Extended native tool usage ({tools_since_zen} tools) without hive coordination",
                suggested_alternative="mcp__zen__chat { prompt: \"Coordinate current workflow with hive intelligence\" }",
                hive_guidance="Queen ZEN's guidance optimizes workflow efficiency and prevents errors",
                priority_score=30
            )
        
        return None
    
    def _is_complex_operation(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Determine if this is a complex operation requiring ZEN planning."""
        complex_indicators = [
            # Large file operations
            tool_name == "Write" and len(tool_input.get("content", "")) > 1000,
            tool_name == "MultiEdit" and len(tool_input.get("edits", [])) > 3,
            
            # System-level bash commands
            tool_name == "Bash" and any(cmd in tool_input.get("command", "")
                                      for cmd in ["npm install", "git clone", "docker", "sudo", "rm -rf"]),
            
            # Project initialization
            tool_name == "Write" and any(file in tool_input.get("file_path", "")
                                       for file in ["package.json", "requirements.txt", "Cargo.toml", "docker-compose"]),
            
            # Subagent deployment
            tool_name == "Task" and len(tool_input.get("description", "")) > 100
        ]
        
        return any(complex_indicators)


class EfficiencyOptimizer(HiveWorkflowValidator):
    """Optimizes workflow efficiency by suggesting MCP alternatives."""
    
    def get_validator_name(self) -> str:
        return "efficiency_optimizer"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Suggest more efficient MCP alternatives."""
        
        # Suggest filesystem MCP for file operations
        if tool_name in ["Read", "Write", "Edit"] and not context.has_flow_coordination(3):
            return self._suggest_filesystem_mcp(tool_name, tool_input)
        
        # Suggest batching for repeated operations
        if self._detect_batching_opportunity(tool_name, context):
            return self._suggest_batching_optimization(tool_name, context)
        
        # Suggest memory coordination for multiple operations
        if self._needs_memory_coordination(tool_name, context):
            return self._suggest_memory_coordination(tool_name, context)
        
        return None
    
    def _suggest_filesystem_mcp(self, tool_name: str, tool_input: Dict[str, Any]) -> ValidationResult:
        """Suggest filesystem MCP for better coordination."""
        mcp_alternatives = {
            "Read": "mcp__filesystem__read_text_file",
            "Write": "mcp__filesystem__write_file", 
            "Edit": "mcp__filesystem__edit_file"
        }
        
        return ValidationResult(
            severity=ValidationSeverity.SUGGEST,
            violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
            message=f"🔧 Consider using {mcp_alternatives.get(tool_name)} for better hive coordination",
            suggested_alternative=f"mcp__zen__planner followed by {mcp_alternatives.get(tool_name)}",
            hive_guidance="MCP tools provide superior coordination and error handling",
            priority_score=20
        )
    
    def _detect_batching_opportunity(self, tool_name: str, context: WorkflowContextTracker) -> bool:
        """Detect if this tool could be part of a batch operation."""
        recent_tools = context._recent_tools[-3:]
        return (len([t for t in recent_tools if t in ["Read", "Write", "Edit"]]) >= 2 and
                tool_name in ["Read", "Write", "Edit"])
    
    def _suggest_batching_optimization(self, tool_name: str, context: WorkflowContextTracker) -> ValidationResult:
        """Suggest batching multiple file operations."""
        return ValidationResult(
            severity=ValidationSeverity.SUGGEST,
            violation_type=WorkflowViolationType.FRAGMENTED_WORKFLOW,
            message="📦 Multiple file operations detected - consider batching for efficiency",
            suggested_alternative="mcp__filesystem__read_multiple_files or mcp__zen__planner for coordinated operations",
            hive_guidance="Queen ZEN can coordinate batched operations for 300% efficiency improvement",
            priority_score=35
        )
    
    def _needs_memory_coordination(self, tool_name: str, context: WorkflowContextTracker) -> bool:
        """Check if memory coordination would be beneficial."""
        return (tool_name == "Task" and 
                context.get_tools_since_flow() > 2 and
                any("Task" in tool for tool in context._recent_tools[-3:]))
    
    def _suggest_memory_coordination(self, tool_name: str, context: WorkflowContextTracker) -> ValidationResult:
        """Suggest memory coordination for complex workflows."""
        return ValidationResult(
            severity=ValidationSeverity.SUGGEST,
            violation_type=WorkflowViolationType.MISSING_COORDINATION,
            message="🧠 Multiple complex operations - hive memory coordination recommended",
            suggested_alternative="mcp__claude-flow__memory_usage for cross-operation coordination",
            hive_guidance="Hive memory prevents duplicate work and coordinates complex workflows",
            priority_score=45
        )


class SafetyValidator(HiveWorkflowValidator):
    """Validates operations for safety and prevents hive damage."""
    
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",           # Dangerous deletion
        r"sudo\s+rm",              # Sudo deletion
        r">\s*/dev/null\s+2>&1",   # Silencing all output
        r"curl\s+.*\|\s*sh",       # Piping to shell
        r"wget\s+.*\|\s*sh",       # Piping to shell
        r"chmod\s+777",            # Overly permissive permissions
    ]
    
    def get_validator_name(self) -> str:
        return "safety_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate operations for safety."""
        
        if tool_name == "Bash":
            return self._validate_bash_safety(tool_input.get("command", ""))
        
        elif tool_name in ["Write", "Edit"]:
            return self._validate_file_operation_safety(tool_name, tool_input)
        
        return None
    
    def _validate_bash_safety(self, command: str) -> Optional[ValidationResult]:
        """Validate bash command safety."""
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return ValidationResult(
                    severity=ValidationSeverity.BLOCK,
                    violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                    message=f"🚨 DANGEROUS COMMAND BLOCKED: {command[:50]}...",
                    blocking_reason="Command could damage the hive or system integrity",
                    hive_guidance="Queen ZEN recommends safer alternatives or explicit confirmation",
                    priority_score=100
                )
        
        # Check for commands that should use ZEN planning
        complex_commands = ["docker", "npm install", "pip install", "git clone"]
        if any(cmd in command for cmd in complex_commands):
            return ValidationResult(
                severity=ValidationSeverity.WARN,
                violation_type=WorkflowViolationType.BYPASSING_ZEN,
                message=f"⚠️ Complex command should have Queen ZEN's planning: {command[:30]}...",
                suggested_alternative="mcp__zen__thinkdeep for command planning and optimization",
                hive_guidance="Queen ZEN can optimize complex commands and prevent errors",
                priority_score=50
            )
        
        return None
    
    def _validate_file_operation_safety(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate file operation safety."""
        file_path = (tool_input.get("file_path") or 
                    tool_input.get("path") or "")
        
        # Check for dangerous file paths
        dangerous_paths = ["/etc/", "/usr/", "/sys/", "/proc/", "/dev/"]
        if any(dangerous_path in file_path for dangerous_path in dangerous_paths):
            return ValidationResult(
                severity=ValidationSeverity.BLOCK,
                violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                message=f"🚨 DANGEROUS FILE OPERATION BLOCKED: {file_path}",
                blocking_reason="Operation could affect system files or directories",
                hive_guidance="Queen ZEN recommends working within safe project directories",
                priority_score=95
            )
        
        return None