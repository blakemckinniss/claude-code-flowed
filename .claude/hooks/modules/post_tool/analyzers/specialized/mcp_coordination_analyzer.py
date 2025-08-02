"""MCP Coordination Analyzer.

Specialized analyzer for MCP tool usage patterns, ensuring proper
Queen ZEN → Flow → Storage → Execution hierarchy and coordination.
"""

import re
from typing import Dict, Any, List, Optional, Set

from ...core.tool_analyzer_base import (
    BaseToolAnalyzer, ToolContext, FeedbackResult, FeedbackSeverity, ToolCategory
)


class MCPCoordinationAnalyzer(BaseToolAnalyzer):
    """Analyzer for MCP tool coordination and workflow optimization."""
    
    def __init__(self, priority: int = 900):
        """Initialize MCP coordination analyzer."""
        super().__init__(priority)
        
        # Define hierarchy levels
        self.zen_tools = ["mcp__zen__chat", "mcp__zen__thinkdeep", "mcp__zen__planner", 
                         "mcp__zen__consensus", "mcp__zen__analyze", "mcp__zen__debug"]
        self.flow_tools = ["mcp__claude-flow__swarm_init", "mcp__claude-flow__agent_spawn",
                          "mcp__claude-flow__task_orchestrate", "mcp__claude-flow__memory_usage"]
        self.storage_tools = ["mcp__filesystem__read_file", "mcp__filesystem__write_file",
                             "mcp__filesystem__list_directory"]
        
        # Ideal workflow patterns
        self.optimal_patterns = [
            # Full hierarchy: ZEN → Flow → Storage → Execution
            r"mcp__zen__.* → mcp__claude-flow__.* → mcp__filesystem__.* → (Write|Edit|Bash)",
            # ZEN direct coordination
            r"mcp__zen__.* → (mcp__filesystem__|Write|Edit|Bash)",
            # Flow with storage
            r"mcp__claude-flow__.* → mcp__filesystem__.* → (Write|Edit)",
        ]
        
        # Coordination drift patterns
        self.drift_patterns = [
            {
                "pattern": r"(mcp__filesystem__|Write|Edit|Bash){3,}",
                "type": "missing_coordination",
                "severity": FeedbackSeverity.WARNING,
                "message": "Extended execution without MCP coordination"
            },
            {
                "pattern": r"mcp__claude-flow__.* → (Write|Edit|Bash)",
                "type": "bypassed_zen",
                "severity": FeedbackSeverity.WARNING,
                "message": "Flow workers acting without Queen ZEN coordination"
            }
        ]
    
    def get_analyzer_name(self) -> str:
        return "mcp_coordination_analyzer"
    
    def get_supported_tools(self) -> List[str]:
        return ["mcp__*"]
    
    def get_tool_categories(self) -> List[ToolCategory]:
        return [ToolCategory.MCP_COORDINATION]
    
    async def _analyze_tool_impl(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze MCP tool coordination patterns."""
        tool_name = context.tool_name
        
        # Skip non-MCP tools
        if not tool_name.startswith("mcp__"):
            return None
        
        # Analyze coordination hierarchy
        hierarchy_result = self._analyze_coordination_hierarchy(context)
        if hierarchy_result:
            return hierarchy_result
        
        # Analyze workflow patterns
        pattern_result = self._analyze_workflow_patterns(context)
        if pattern_result:
            return pattern_result
        
        # Analyze tool parameter optimization
        param_result = self._analyze_tool_parameters(context)
        if param_result:
            return param_result
        
        # Check for missing coordination opportunities
        coordination_result = self._check_coordination_opportunities(context)
        if coordination_result:
            return coordination_result
        
        return None
    
    def _analyze_coordination_hierarchy(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze adherence to Queen ZEN coordination hierarchy."""
        tool_name = context.tool_name
        workflow_history = context.workflow_history
        
        # Check if we're following proper hierarchy
        if any(tool_name.startswith(prefix) for prefix in ["mcp__claude-flow__", "mcp__filesystem__"]):
            # Check if ZEN coordination preceded this
            recent_zen = any(tool.startswith("mcp__zen__") for tool in workflow_history[-5:])
            
            if not recent_zen:
                hierarchy_level = "Flow Workers" if tool_name.startswith("mcp__claude-flow__") else "Storage Workers"
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message=f"{hierarchy_level} operating without Queen ZEN coordination",
                    suggestions=[
                        "Start with Queen ZEN guidance: mcp__zen__chat or mcp__zen__thinkdeep",
                        "Use mcp__zen__planner for complex workflow coordination",
                        "Queen ZEN should command all hive operations"
                    ],
                    metadata={
                        "hierarchy_level": hierarchy_level,
                        "missing_coordination": "zen"
                    },
                    analyzer_name=self.get_analyzer_name()
                )
        
        return None
    
    def _analyze_workflow_patterns(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze workflow patterns for optimization opportunities."""
        workflow_history = context.workflow_history
        if len(workflow_history) < 3:
            return None
        
        # Create workflow sequence
        recent_sequence = " → ".join(workflow_history[-6:])
        
        # Check for drift patterns
        for pattern_info in self.drift_patterns:
            if re.search(pattern_info["pattern"], recent_sequence):
                return FeedbackResult(
                    severity=pattern_info["severity"],
                    message=pattern_info["message"],
                    suggestions=self._get_coordination_suggestions(pattern_info["type"]),
                    metadata={
                        "drift_type": pattern_info["type"],
                        "workflow_sequence": recent_sequence
                    },
                    analyzer_name=self.get_analyzer_name()
                )
        
        return None
    
    def _analyze_tool_parameters(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze MCP tool parameters for optimization."""
        tool_name = context.tool_name
        tool_input = context.tool_input
        
        # ZEN tool parameter optimization
        if tool_name.startswith("mcp__zen__"):
            return self._analyze_zen_parameters(tool_name, tool_input)
        
        # Claude Flow parameter optimization
        elif tool_name.startswith("mcp__claude-flow__"):
            return self._analyze_flow_parameters(tool_name, tool_input)
        
        return None
    
    def _analyze_zen_parameters(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[FeedbackResult]:
        """Analyze ZEN tool parameters."""
        if tool_name == "mcp__zen__thinkdeep":
            thinking_mode = tool_input.get("thinking_mode", "medium")
            if thinking_mode == "minimal":
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message="Queen ZEN using minimal thinking mode - may limit analysis depth",
                    suggestions=[
                        "Consider 'medium' or 'high' thinking mode for complex tasks",
                        "Use 'max' thinking mode for enterprise-level decisions"
                    ],
                    analyzer_name=self.get_analyzer_name()
                )
        
        elif tool_name == "mcp__zen__consensus":
            models = tool_input.get("models", [])
            if len(models) < 2:
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message="Queen ZEN consensus with insufficient model diversity",
                    suggestions=[
                        "Use at least 2-3 models for reliable consensus",
                        "Include models with different stances (for/against/neutral)"
                    ],
                    analyzer_name=self.get_analyzer_name()
                )
        
        return None
    
    def _analyze_flow_parameters(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[FeedbackResult]:
        """Analyze Claude Flow parameters."""
        if tool_name == "mcp__claude-flow__swarm_init":
            max_agents = tool_input.get("maxAgents", 5)
            if max_agents > 10:
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message=f"Large swarm size ({max_agents}) may impact performance",
                    suggestions=[
                        "Consider starting with 3-5 agents and scaling up",
                        "Use hierarchical topology for large agent counts"
                    ],
                    analyzer_name=self.get_analyzer_name()
                )
        
        elif tool_name == "mcp__claude-flow__agent_spawn":
            agent_type = tool_input.get("type", "")
            if not agent_type:
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message="Agent spawned without specific type designation",
                    suggestions=[
                        "Specify agent type for optimal specialization",
                        "Use types like 'coder', 'reviewer', 'analyst', 'architect'"
                    ],
                    analyzer_name=self.get_analyzer_name()
                )
        
        return None
    
    def _check_coordination_opportunities(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Check for missed coordination opportunities."""
        workflow_history = context.workflow_history
        
        # Check if we have multiple execution tools without coordination
        execution_tools = ["Write", "Edit", "Bash", "Task"]
        recent_execution = [tool for tool in workflow_history[-5:] if tool in execution_tools]
        
        if len(recent_execution) >= 3:
            # Check if we have MCP coordination
            recent_mcp = [tool for tool in workflow_history[-5:] if tool.startswith("mcp__")]
            
            if not recent_mcp:
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message=f"Multiple execution operations ({len(recent_execution)}) without MCP coordination",
                    suggestions=[
                        "Use mcp__zen__planner to coordinate execution sequence",
                        "Initialize swarm with mcp__claude-flow__swarm_init",
                        "Consider using mcp__zen__chat for workflow guidance"
                    ],
                    metadata={
                        "execution_count": len(recent_execution),
                        "missing_coordination": True
                    },
                    analyzer_name=self.get_analyzer_name()
                )
        
        return None
    
    def _get_coordination_suggestions(self, drift_type: str) -> List[str]:
        """Get coordination suggestions based on drift type."""
        suggestions = {
            "missing_coordination": [
                "Start with Queen ZEN: mcp__zen__chat or mcp__zen__thinkdeep",
                "Use mcp__claude-flow__swarm_init for multi-agent coordination",
                "Plan workflow with mcp__zen__planner"
            ],
            "bypassed_zen": [
                "Queen ZEN must command all hive operations",
                "Use mcp__zen__consensus for complex decisions",
                "Establish hierarchy: ZEN → Flow → Storage → Execution"
            ],
            "fragmented_workflow": [
                "Coordinate operations through mcp__zen__planner",
                "Use mcp__claude-flow__memory_usage for state management",
                "Batch related operations for efficiency"
            ]
        }
        
        return suggestions.get(drift_type, [
            "Consider using MCP coordination tools",
            "Follow Queen ZEN → Flow → Storage → Execution hierarchy"
        ])


class MCPParameterValidator(BaseToolAnalyzer):
    """Validator for MCP tool parameters and configurations."""
    
    def __init__(self, priority: int = 700):
        super().__init__(priority)
        
        # Parameter validation rules
        self.validation_rules = {
            "mcp__zen__thinkdeep": {
                "required": ["step", "step_number", "total_steps", "next_step_required", "findings", "model"],
                "optional": ["thinking_mode", "confidence", "use_websearch"]
            },
            "mcp__claude-flow__swarm_init": {
                "required": ["topology"],
                "optional": ["maxAgents", "strategy"]
            },
            "mcp__zen__consensus": {
                "required": ["step", "step_number", "total_steps", "next_step_required", "findings"],
                "optional": ["models", "images"]
            }
        }
    
    def get_analyzer_name(self) -> str:
        return "mcp_parameter_validator"
    
    def get_supported_tools(self) -> List[str]:
        return list(self.validation_rules.keys())
    
    def get_tool_categories(self) -> List[ToolCategory]:
        return [ToolCategory.MCP_COORDINATION]
    
    async def _analyze_tool_impl(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Validate MCP tool parameters."""
        tool_name = context.tool_name
        tool_input = context.tool_input
        
        if tool_name not in self.validation_rules:
            return None
        
        rules = self.validation_rules[tool_name]
        issues = []
        
        # Check required parameters
        for required_param in rules["required"]:
            if required_param not in tool_input:
                issues.append(f"Missing required parameter: {required_param}")
        
        # Check parameter values
        validation_result = self._validate_parameter_values(tool_name, tool_input)
        if validation_result:
            issues.extend(validation_result)
        
        if issues:
            return FeedbackResult(
                severity=FeedbackSeverity.ERROR,
                message=f"Invalid parameters for {tool_name}",
                suggestions=[f"Fix: {issue}" for issue in issues[:3]],
                metadata={
                    "tool_name": tool_name,
                    "parameter_issues": issues
                },
                analyzer_name=self.get_analyzer_name()
            )
        
        return None
    
    def _validate_parameter_values(self, tool_name: str, tool_input: Dict[str, Any]) -> List[str]:
        """Validate specific parameter values."""
        issues = []
        
        if tool_name == "mcp__zen__thinkdeep":
            # Validate thinking_mode
            thinking_mode = tool_input.get("thinking_mode")
            if thinking_mode and thinking_mode not in ["minimal", "low", "medium", "high", "max"]:
                issues.append(f"Invalid thinking_mode: {thinking_mode}")
            
            # Validate step_number
            step_number = tool_input.get("step_number")
            if step_number and (not isinstance(step_number, int) or step_number < 1):
                issues.append("step_number must be positive integer")
        
        elif tool_name == "mcp__claude-flow__swarm_init":
            # Validate topology
            topology = tool_input.get("topology")
            if topology and topology not in ["hierarchical", "mesh", "ring", "star"]:
                issues.append(f"Invalid topology: {topology}")
            
            # Validate maxAgents
            max_agents = tool_input.get("maxAgents")
            if max_agents and (not isinstance(max_agents, int) or max_agents < 1 or max_agents > 100):
                issues.append("maxAgents must be between 1 and 100")
        
        return issues