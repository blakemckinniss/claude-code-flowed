"""Hive Workflow Optimizer - Advanced workflow pattern analysis.

Implements the Queen's hive intelligence to detect and optimize
complex workflow patterns before execution.
"""

from typing import Dict, Any, Optional, List
from ..core.workflow_validator import (
    HiveWorkflowValidator, 
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class HiveWorkflowOptimizer(HiveWorkflowValidator):
    """Advanced hive workflow optimization and pattern analysis."""
    
    # Workflow patterns that benefit from hive coordination
    HIVE_PATTERNS = {
        "batch_operations": {
            "tools": ["Read", "Write", "Edit"],
            "threshold": 2,
            "optimization": "Batch file operations for 300% efficiency gain"
        },
        "project_initialization": {
            "tools": ["Write", "Bash"],
            "indicators": ["package.json", "npm install", "git init"],
            "optimization": "Queen ZEN can orchestrate complete project setup"
        },
        "testing_workflow": {
            "tools": ["Write", "Bash", "Task"],
            "indicators": ["test", "spec", "npm test", "pytest"],
            "optimization": "Coordinated testing with intelligent agent swarms"
        },
        "deployment_preparation": {
            "tools": ["Bash", "Write", "Edit"],
            "indicators": ["docker", "deploy", "build", "ci/cd"],
            "optimization": "Queen ZEN ensures deployment readiness and safety"
        }
    }
    
    def __init__(self, priority: int = 750):
        super().__init__(priority)  
        self._detected_patterns: List[str] = []
        self._workflow_context = {
            "project_phase": "unknown",  # initialization, development, testing, deployment
            "complexity_level": "simple",  # simple, moderate, complex, enterprise
            "coordination_opportunity": 0  # 0-100 score
        }
    
    def get_validator_name(self) -> str:
        return "hive_workflow_optimizer"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Analyze workflow patterns and suggest hive optimizations."""
        
        # Update workflow context based on tool usage
        self._update_workflow_context(tool_name, tool_input, context)
        
        # Detect workflow patterns
        detected_pattern = self._detect_workflow_pattern(tool_name, tool_input, context)
        
        if detected_pattern:
            return self._optimize_workflow_pattern(detected_pattern, tool_name, tool_input, context)
        
        # Check for missed optimization opportunities
        optimization_opportunity = self._assess_optimization_opportunity(tool_name, tool_input, context)
        
        if optimization_opportunity:
            return optimization_opportunity
        
        return None
    
    def _update_workflow_context(self, tool_name: str, tool_input: Dict[str, Any], 
                                context: WorkflowContextTracker) -> None:
        """Update workflow context understanding."""
        
        # Detect project phase
        if any(indicator in str(tool_input).lower() 
               for indicator in ["package.json", "requirements.txt", "cargo.toml"]):
            self._workflow_context["project_phase"] = "initialization"
        elif any(indicator in str(tool_input).lower()
                for indicator in ["test", "spec", "jest", "pytest"]):
            self._workflow_context["project_phase"] = "testing"
        elif any(indicator in str(tool_input).lower()
                for indicator in ["docker", "deploy", "build", "ci"]):
            self._workflow_context["project_phase"] = "deployment"
        else:
            self._workflow_context["project_phase"] = "development"
        
        # Assess complexity level
        tools_in_window = len(context._recent_tools)
        coordination_state = context.get_coordination_state()
        
        if tools_in_window >= 5 and coordination_state == "disconnected":
            self._workflow_context["complexity_level"] = "complex"
        elif tools_in_window >= 3:
            self._workflow_context["complexity_level"] = "moderate"
        else:
            self._workflow_context["complexity_level"] = "simple"
        
        # Calculate coordination opportunity score
        base_score = min(tools_in_window * 10, 50)  # Base on tool count
        coordination_bonus = 30 if coordination_state == "disconnected" else 10
        phase_bonus = {"initialization": 20, "testing": 15, "deployment": 25, "development": 5}
        
        self._workflow_context["coordination_opportunity"] = (
            base_score + coordination_bonus + phase_bonus.get(self._workflow_context["project_phase"], 0)
        )
    
    def _detect_workflow_pattern(self, tool_name: str, tool_input: Dict[str, Any], 
                                context: WorkflowContextTracker) -> Optional[str]:
        """Detect specific workflow patterns."""
        
        recent_tools = context._recent_tools + [tool_name]
        recent_inputs = str(tool_input).lower()
        
        # Check each defined pattern
        for pattern_name, pattern_config in self.HIVE_PATTERNS.items():
            if self._matches_pattern(pattern_name, pattern_config, recent_tools, recent_inputs):
                if pattern_name not in self._detected_patterns:
                    self._detected_patterns.append(pattern_name)
                return pattern_name
        
        return None
    
    def _matches_pattern(self, pattern_name: str, pattern_config: Dict[str, Any], 
                        recent_tools: List[str], recent_inputs: str) -> bool:
        """Check if current workflow matches a specific pattern."""
        
        # Check tool presence
        required_tools = pattern_config.get("tools", [])
        tools_present = sum(1 for tool in required_tools if tool in recent_tools[-5:])
        threshold = pattern_config.get("threshold", len(required_tools))
        
        if tools_present < threshold:
            return False
        
        # Check indicators if present
        if "indicators" in pattern_config:
            indicators_present = any(indicator in recent_inputs 
                                   for indicator in pattern_config["indicators"])
            if not indicators_present:
                return False
        
        return True
    
    def _optimize_workflow_pattern(self, pattern_name: str, tool_name: str, 
                                  tool_input: Dict[str, Any], context: WorkflowContextTracker) -> ValidationResult:
        """Provide optimization suggestions for detected patterns."""
        
        pattern_config = self.HIVE_PATTERNS[pattern_name]
        optimization_message = pattern_config["optimization"]
        
        # Determine severity based on coordination state and complexity
        coordination_state = context.get_coordination_state()
        complexity_level = self._workflow_context["complexity_level"]
        
        if coordination_state == "disconnected" and complexity_level in ["complex", "enterprise"]:
            severity = ValidationSeverity.WARN
            priority = 70
            prefix = "🚨 HIVE OPTIMIZATION NEEDED:"
        elif coordination_state != "coordinated":
            severity = ValidationSeverity.SUGGEST  
            priority = 50
            prefix = "🐝 HIVE OPTIMIZATION:"
        else:
            severity = ValidationSeverity.ALLOW
            priority = 20
            prefix = "💡 HIVE TIP:"
        
        # Generate pattern-specific suggestions
        suggestions = self._generate_pattern_suggestions(pattern_name, tool_name, tool_input)
        
        return ValidationResult(
            severity=severity,
            violation_type=WorkflowViolationType.FRAGMENTED_WORKFLOW if severity != ValidationSeverity.ALLOW else None,
            message=f"{prefix} {pattern_name.replace('_', ' ').title()} pattern detected",
            suggested_alternative=suggestions["primary"],
            hive_guidance=f"{optimization_message}. {suggestions['guidance']}",
            priority_score=priority
        )
    
    def _generate_pattern_suggestions(self, pattern_name: str, tool_name: str, 
                                     tool_input: Dict[str, Any]) -> Dict[str, str]:
        """Generate specific suggestions for workflow patterns."""
        
        suggestions = {
            "batch_operations": {
                "primary": "mcp__zen__planner { step: \"Coordinate batch file operations\", thinking_mode: \"medium\" }",
                "guidance": "Queen ZEN can batch operations for 3x efficiency and prevent file conflicts"
            },
            "project_initialization": {
                "primary": "mcp__zen__thinkdeep { step: \"Plan complete project setup\", thinking_mode: \"high\" }",
                "guidance": "Queen ZEN orchestrates complete project initialization with best practices"
            },
            "testing_workflow": {
                "primary": "mcp__zen__testgen { step: \"Generate comprehensive test strategy\" }",
                "guidance": "Queen ZEN creates optimal testing workflows with intelligent coverage"
            },
            "deployment_preparation": {
                "primary": "mcp__zen__consensus { step: \"Validate deployment readiness\" }",
                "guidance": "Queen ZEN ensures deployment safety and readiness validation"
            }
        }
        
        return suggestions.get(pattern_name, {
            "primary": "mcp__zen__chat { prompt: \"Optimize current workflow pattern\" }",
            "guidance": "Queen ZEN can analyze and optimize this workflow pattern"
        })
    
    def _assess_optimization_opportunity(self, tool_name: str, tool_input: Dict[str, Any], 
                                       context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Assess general optimization opportunities."""
        
        opportunity_score = self._workflow_context["coordination_opportunity"]
        
        # High opportunity score suggests significant optimization potential
        if opportunity_score >= 60:
            phase = self._workflow_context["project_phase"]
            complexity = self._workflow_context["complexity_level"]
            
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.MISSING_COORDINATION,
                message=f"🎯 HIGH OPTIMIZATION OPPORTUNITY: {phase} phase, {complexity} complexity (score: {opportunity_score})",
                suggested_alternative=f"mcp__zen__planner {{ step: \"Optimize {phase} workflow\", thinking_mode: \"high\" }}",
                hive_guidance=f"Queen ZEN can provide {opportunity_score}% efficiency improvement for {phase} workflows",
                priority_score=opportunity_score
            )
        
        # Medium opportunity - gentle suggestion
        elif opportunity_score >= 40:
            return ValidationResult(
                severity=ValidationSeverity.ALLOW,
                violation_type=None,
                message=f"💡 Workflow optimization available (score: {opportunity_score})",
                suggested_alternative="mcp__zen__chat { prompt: \"Review current workflow for optimization opportunities\" }",
                hive_guidance="Queen ZEN can identify workflow efficiency improvements",
                priority_score=20
            )
        
        return None