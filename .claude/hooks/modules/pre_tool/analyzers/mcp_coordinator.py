"""MCP Coordination Validator for proactive tool optimization.

Analyzes tool usage patterns and proactively suggests MCP alternatives
that align with Queen ZEN → Flow Workers → Storage Workers hierarchy.
"""

from typing import Dict, Any, Optional
from ..core.workflow_validator import (
    HiveWorkflowValidator, 
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class MCPCoordinationValidator(HiveWorkflowValidator):
    """Validates and optimizes MCP tool coordination patterns."""
    
    # MCP tool alternatives mapping
    MCP_ALTERNATIVES = {
        # Native → Filesystem MCP alternatives
        "Read": "mcp__filesystem__read_text_file",
        "Write": "mcp__filesystem__write_file", 
        "Edit": "mcp__filesystem__edit_file",
        "MultiEdit": "mcp__filesystem__edit_file",
        "Glob": "mcp__filesystem__search_files",
        "LS": "mcp__filesystem__list_directory",
        
        # Native → ZEN MCP alternatives for complex operations
        "Task": "mcp__zen__thinkdeep → Task coordination",
        "Bash": "mcp__zen__planner → Bash optimization",
        
        # Direct MCP → ZEN-coordinated alternatives
        "mcp__filesystem__write_file": "mcp__zen__planner → mcp__filesystem__write_file",
        "mcp__claude-flow__swarm_init": "mcp__zen__chat → mcp__claude-flow__swarm_init"
    }
    
    # Complex operations that should always use ZEN planning
    COMPLEX_OPERATIONS = {
        "npm install", "pip install", "docker", "git clone", 
        "sudo", "rm -rf", "chmod", "chown", "systemctl"
    }
    
    def __init__(self, priority: int = 900):
        super().__init__(priority)
    
    def get_validator_name(self) -> str:
        return "mcp_coordination_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate MCP coordination patterns and suggest optimizations."""
        
        # Prioritize ZEN-coordinated MCP tools - highest efficiency
        if tool_name.startswith("mcp__zen__"):
            return self._validate_zen_optimization(tool_name, tool_input, context)
        
        # Check Flow Worker coordination patterns
        elif tool_name.startswith("mcp__claude-flow__"):
            return self._validate_flow_coordination(tool_name, tool_input, context)
        
        # Check Storage Worker patterns
        elif tool_name.startswith("mcp__filesystem__"):
            return self._validate_storage_coordination(tool_name, tool_input, context)
        
        # Check native tools for MCP optimization opportunities
        elif not tool_name.startswith("mcp__"):
            return self._suggest_mcp_optimization(tool_name, tool_input, context)
        
        # Other MCP tools - general validation
        else:
            return self._validate_general_mcp(tool_name, tool_input, context)
    
    def _validate_zen_optimization(self, tool_name: str, tool_input: Dict[str, Any], 
                                  context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate ZEN tool usage for optimal hive coordination."""
        
        # ZEN tools are always optimal - provide encouragement
        zen_tool_benefits = {
            "mcp__zen__chat": "Supreme hive intelligence guidance",
            "mcp__zen__thinkdeep": "Deep strategic analysis with expert validation", 
            "mcp__zen__planner": "Optimal step-by-step workflow coordination",
            "mcp__zen__consensus": "Multi-model decision validation",
            "mcp__zen__debug": "Systematic issue investigation",
            "mcp__zen__analyze": "Comprehensive code analysis",
            "mcp__zen__testgen": "Intelligent test generation"
        }
        
        benefit = zen_tool_benefits.get(tool_name, "Queen ZEN's supreme coordination")
        
        return ValidationResult(
            severity=ValidationSeverity.ALLOW,
            violation_type=None,
            message=f"👑 OPTIMAL: {tool_name} - {benefit}",
            hive_guidance="Perfect! Queen ZEN leads with supreme hive intelligence.",
            priority_score=0
        )
    
    def _validate_flow_coordination(self, tool_name: str, tool_input: Dict[str, Any], 
                                   context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate Flow Worker coordination patterns."""
        
        # Check if Flow Worker has recent ZEN guidance
        if not context.has_zen_coordination(3):
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.WORKER_INSUBORDINATION,
                message=f"🤖 Flow Worker '{tool_name}' can be enhanced with Queen ZEN's guidance",
                suggested_alternative=f"mcp__zen__chat {{ prompt: \"Guide {tool_name} coordination strategy\" }}",
                hive_guidance="Queen ZEN's pre-coordination increases Flow Worker effectiveness by 40%",
                priority_score=30
            )
        
        # Flow Workers with ZEN coordination are excellent
        return ValidationResult(
            severity=ValidationSeverity.ALLOW,
            violation_type=None,
            message=f"🤖 COORDINATED: Flow Worker '{tool_name}' operating under Queen ZEN's guidance",
            hive_guidance="Excellent! Flow Workers coordinated by Queen ZEN achieve optimal results.",
            priority_score=0
        )
    
    def _validate_storage_coordination(self, tool_name: str, tool_input: Dict[str, Any], 
                                      context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate Storage Worker coordination patterns."""
        
        # Check for complex file operations without coordination
        if self._is_complex_file_operation(tool_name, tool_input):
            if not context.has_zen_coordination(2) and not context.has_flow_coordination(2):
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.BYPASSING_ZEN,
                    message=f"📁 Complex Storage operation '{tool_name}' benefits from hive coordination",
                    suggested_alternative="mcp__zen__planner → coordinate Storage Worker operations",
                    hive_guidance="Queen ZEN's coordination prevents Storage Worker errors and optimizes file operations",
                    priority_score=50
                )
        
        # Standard Storage Worker operations are acceptable
        return ValidationResult(
            severity=ValidationSeverity.ALLOW,
            violation_type=None,
            message=f"📁 Storage Worker '{tool_name}' - acceptable operation",
            hive_guidance="Storage Workers handle file operations efficiently",
            priority_score=0
        )
    
    def _suggest_mcp_optimization(self, tool_name: str, tool_input: Dict[str, Any], 
                                 context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Suggest MCP alternatives for native tools."""
        
        # Check for direct MCP alternatives
        if tool_name in self.MCP_ALTERNATIVES:
            mcp_alternative = self.MCP_ALTERNATIVES[tool_name]
            
            # Different severities based on coordination state
            if context.get_coordination_state() == "disconnected":
                severity = ValidationSeverity.SUGGEST
                priority = 40
                prefix = "🔄 OPTIMIZATION:"
            else:
                severity = ValidationSeverity.ALLOW
                priority = 10  
                prefix = "💡 TIP:"
            
            return ValidationResult(
                severity=severity,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION if severity == ValidationSeverity.SUGGEST else None,
                message=f"{prefix} Consider {mcp_alternative} for enhanced hive coordination",
                suggested_alternative=mcp_alternative,
                hive_guidance="MCP tools provide superior error handling, coordination, and performance",
                priority_score=priority
            )
        
        # Check for complex Bash operations
        if tool_name == "Bash" and self._is_complex_bash_command(tool_input.get("command", "")):
            return ValidationResult(
                severity=ValidationSeverity.WARN,
                violation_type=WorkflowViolationType.BYPASSING_ZEN,
                message="⚡ Complex Bash command detected - Queen ZEN can optimize and secure execution",
                suggested_alternative="mcp__zen__thinkdeep { step: \"Analyze and optimize bash command\", thinking_mode: \"medium\" }",
                hive_guidance="Queen ZEN's analysis prevents command errors and optimizes execution strategy",
                priority_score=60
            )
        
        return None
    
    def _validate_general_mcp(self, tool_name: str, tool_input: Dict[str, Any], 
                             context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate other MCP tools for coordination opportunities."""
        
        # Suggest ZEN coordination for complex MCP operations
        if self._is_complex_mcp_operation(tool_name, tool_input):
            if not context.has_zen_coordination(5):
                return ValidationResult(
                    severity=ValidationSeverity.SUGGEST,
                    violation_type=WorkflowViolationType.MISSING_COORDINATION,
                    message=f"🔗 MCP tool '{tool_name}' can be enhanced with Queen ZEN's strategic planning",
                    suggested_alternative=f"mcp__zen__planner → {tool_name} coordination",
                    hive_guidance="Queen ZEN's pre-planning optimizes complex MCP operations",
                    priority_score=25
                )
        
        return None
    
    def _is_complex_file_operation(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Determine if this is a complex file operation."""
        complex_indicators = [
            # Large files
            tool_name in ["mcp__filesystem__write_file"] and len(tool_input.get("content", "")) > 2000,
            
            # Multiple files
            tool_name in ["mcp__filesystem__read_multiple_files"] and len(tool_input.get("paths", [])) > 3,
            
            # System or config files
            any(path in tool_input.get("path", "")  
                for path in ["/etc/", "/usr/", "docker-compose", "package.json", ".env"]),
                
            # Critical project files
            any(file in tool_input.get("path", "")
                for file in ["package.json", "requirements.txt", "Cargo.toml", "tsconfig.json"])
        ]
        
        return any(complex_indicators)
    
    def _is_complex_bash_command(self, command: str) -> bool:
        """Determine if this is a complex bash command."""
        return any(cmd in command.lower() for cmd in self.COMPLEX_OPERATIONS)
    
    def _is_complex_mcp_operation(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Determine if this is a complex MCP operation."""
        complex_mcp_patterns = [
            # GitHub operations
            tool_name.startswith("mcp__github__") and "create" in tool_name,
            
            # Large-scale tree-sitter operations
            tool_name == "mcp__tree_sitter__analyze_project",
            
            # Web crawling/extraction
            tool_name in ["mcp__tavily-remote__tavily_crawl", "mcp__tavily-remote__tavily_extract"],
            
            # Playwright automation
            tool_name.startswith("mcp__playwright__") and len(str(tool_input)) > 100
        ]
        
        return any(complex_mcp_patterns)