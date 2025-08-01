"""Workflow Pattern Analyzer - Detects optimal vs suboptimal workflow patterns."""

from typing import Dict, Any, Optional, List, Set
from ..core.drift_detector import DriftAnalyzer, DriftEvidence, DriftType, DriftSeverity
import re


class WorkflowPatternAnalyzer(DriftAnalyzer):
    """Analyzes workflow patterns for adherence to hive intelligence principles."""
    
    # Optimal workflow patterns (Queen ZEN → Flow → Storage → Execution)
    OPTIMAL_PATTERNS = [
        # Full hive coordination
        r"mcp__zen__.* → mcp__claude-flow__.* → mcp__filesystem__.* → (Write|Edit|Read)",
        
        # ZEN planning then execution
        r"mcp__zen__(thinkdeep|planner).* → (Write|Edit|Bash|Task)",
        
        # Flow coordination with memory
        r"mcp__claude-flow__(swarm_init|memory_usage).* → mcp__filesystem__.* → (Write|Edit)",
        
        # ZEN consensus for decisions
        r"mcp__zen__consensus.* → mcp__claude-flow__.* → (Write|Edit|Bash)"
    ]
    
    # Suboptimal patterns that should trigger guidance
    SUBOPTIMAL_PATTERNS = [
        {
            "pattern": r"(Write|Edit|Read|Bash){3,}",  # 3+ consecutive native tools
            "drift_type": DriftType.NO_MCP_COORDINATION,
            "severity": DriftSeverity.MODERATE,
            "message": "Extended native tool sequence without hive coordination"
        },
        {
            "pattern": r"mcp__filesystem__.* → (Write|Edit) → mcp__filesystem__.* → (Write|Edit)",
            "drift_type": DriftType.FRAGMENTED_WORKFLOW,
            "severity": DriftSeverity.MINOR,
            "message": "Fragmented file operations could be batched with hive intelligence"
        },
        {
            "pattern": r"Task → Task → Task",  # Multiple sequential subagents
            "drift_type": DriftType.MISSING_HIVE_MEMORY,
            "severity": DriftSeverity.MODERATE,
            "message": "Multiple subagents without hive memory coordination"
        }
    ]
    
    def __init__(self, priority: int = 700):
        super().__init__(priority)
        self._workflow_segments: List[str] = []
        self._batch_opportunities = 0
        
    def get_analyzer_name(self) -> str:
        return "workflow_pattern_analyzer"
    
    def analyze_drift(self, tool_name: str, tool_input: Dict[str, Any], 
                     tool_response: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze workflow patterns for optimization opportunities."""
        self.add_tool_usage(tool_name, tool_input, tool_response)
        self._workflow_segments.append(tool_name)
        
        # Analyze recent workflow segment
        recent_workflow = " → ".join(self.get_recent_tools(6))
        
        # Check for suboptimal patterns
        for pattern_info in self.SUBOPTIMAL_PATTERNS:
            if re.search(pattern_info["pattern"], recent_workflow):
                return self._create_pattern_evidence(
                    pattern_info, recent_workflow, tool_name
                )
        
        # Check for batch opportunities
        batch_evidence = self._analyze_batch_opportunities(tool_name, tool_input)
        if batch_evidence:
            return batch_evidence
        
        # Check for memory coordination opportunities
        memory_evidence = self._analyze_memory_coordination(tool_name)
        if memory_evidence:
            return memory_evidence
        
        return None
    
    def _create_pattern_evidence(self, pattern_info: Dict, workflow: str, current_tool: str) -> DriftEvidence:
        """Create evidence for detected suboptimal patterns."""
        return DriftEvidence(
            drift_type=pattern_info["drift_type"],
            severity=pattern_info["severity"],
            tool_sequence=self.get_recent_tools(),
            missing_tools=self._suggest_missing_tools(pattern_info["drift_type"]),
            evidence_details=f"{pattern_info['message']}: {workflow}",
            correction_guidance=self._get_correction_guidance(pattern_info["drift_type"]),
            priority_score=self.calculate_priority_score(
                pattern_info["severity"].value, 
                len(self._workflow_segments)
            )
        )
    
    def _analyze_batch_opportunities(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze opportunities for batching operations."""
        recent_tools = self.get_recent_tools(4)
        
        # Detect repeated similar operations that could be batched
        if tool_name in ["Write", "Edit", "Read"]:
            similar_count = sum(1 for t in recent_tools if t in ["Write", "Edit", "Read"])
            
            if similar_count >= 3:  # 3+ file operations
                self._batch_opportunities += 1
                
                return DriftEvidence(
                    drift_type=DriftType.FRAGMENTED_WORKFLOW,
                    severity=DriftSeverity.MINOR,
                    tool_sequence=recent_tools,
                    missing_tools=["mcp__filesystem__read_multiple_files", "mcp__zen__planner"],
                    evidence_details=f"Multiple file operations ({similar_count}) could be batched for efficiency",
                    correction_guidance="Queen ZEN can plan batched operations through Filesystem MCP",
                    priority_score=25
                )
        
        return None
    
    def _analyze_memory_coordination(self, tool_name: str) -> Optional[DriftEvidence]:
        """Analyze need for memory coordination."""
        recent_tools = self.get_recent_tools(5)
        
        # Check for multiple Task tools without memory coordination
        task_count = sum(1 for t in recent_tools if t == "Task")
        has_memory = any("memory" in t for t in recent_tools)
        
        if task_count >= 2 and not has_memory:
            return DriftEvidence(
                drift_type=DriftType.MISSING_HIVE_MEMORY,
                severity=DriftSeverity.MODERATE,
                tool_sequence=recent_tools,
                missing_tools=["mcp__claude-flow__memory_usage", "mcp__zen__consensus"],
                evidence_details=f"Multiple subagents ({task_count}) without hive memory coordination",
                correction_guidance="Hive memory essential for coordinating multiple worker castes",
                priority_score=45
            )
        
        return None
    
    def _suggest_missing_tools(self, drift_type: DriftType) -> List[str]:
        """Suggest missing tools based on drift type."""
        suggestions = {
            DriftType.NO_MCP_COORDINATION: [
                "mcp__zen__chat", "mcp__zen__thinkdeep", "mcp__claude-flow__swarm_init"
            ],
            DriftType.FRAGMENTED_WORKFLOW: [
                "mcp__zen__planner", "mcp__filesystem__read_multiple_files"
            ],
            DriftType.MISSING_HIVE_MEMORY: [
                "mcp__claude-flow__memory_usage", "mcp__zen__consensus"
            ]
        }
        return suggestions.get(drift_type, ["mcp__zen__chat"])
    
    def _get_correction_guidance(self, drift_type: DriftType) -> str:
        """Get correction guidance for drift type."""
        guidance = {
            DriftType.NO_MCP_COORDINATION: "Queen ZEN's hive coordination provides superior workflow optimization",
            DriftType.FRAGMENTED_WORKFLOW: "Hive intelligence can batch and optimize these operations",
            DriftType.MISSING_HIVE_MEMORY: "Hive memory ensures coordinated decision-making across worker castes"
        }
        return guidance.get(drift_type, "Consider using Queen ZEN's hive coordination")


class BatchingOpportunityAnalyzer(DriftAnalyzer):
    """Detects opportunities for batching operations through MCP tools."""
    
    def __init__(self, priority: int = 500):
        super().__init__(priority)
        self._consecutive_file_ops = 0
        self._consecutive_bash_ops = 0
        self._file_paths: Set[str] = set()
        
    def get_analyzer_name(self) -> str:
        return "batching_opportunity_analyzer"
    
    def analyze_drift(self, tool_name: str, tool_input: Dict[str, Any], 
                     tool_response: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze opportunities for batching operations."""
        self.add_tool_usage(tool_name, tool_input, tool_response)
        
        # Track file operations
        if tool_name in ["Read", "Write", "Edit"]:
            self._consecutive_file_ops += 1
            file_path = tool_input.get("file_path", "")
            if file_path:
                self._file_paths.add(file_path)
        else:
            self._consecutive_file_ops = 0
        
        # Track bash operations
        if tool_name == "Bash":
            self._consecutive_bash_ops += 1
        else:
            self._consecutive_bash_ops = 0
        
        # Analyze batching opportunities
        if self._consecutive_file_ops >= 3:
            return self._create_file_batching_evidence()
        
        if self._consecutive_bash_ops >= 3:
            return self._create_bash_batching_evidence()
        
        return None
    
    def _create_file_batching_evidence(self) -> DriftEvidence:
        """Create evidence for file operation batching opportunity."""
        return DriftEvidence(
            drift_type=DriftType.FRAGMENTED_WORKFLOW,
            severity=DriftSeverity.MINOR,
            tool_sequence=self.get_recent_tools(),
            missing_tools=["mcp__filesystem__read_multiple_files", "mcp__zen__planner"],
            evidence_details=f"Sequential file operations on {len(self._file_paths)} files could be batched",
            correction_guidance="Queen ZEN can coordinate batched file operations for 300% efficiency gain",
            priority_score=30
        )
    
    def _create_bash_batching_evidence(self) -> DriftEvidence:
        """Create evidence for bash command batching opportunity."""
        return DriftEvidence(
            drift_type=DriftType.FRAGMENTED_WORKFLOW,
            severity=DriftSeverity.MINOR,
            tool_sequence=self.get_recent_tools(),
            missing_tools=["mcp__zen__planner"],
            evidence_details=f"Multiple bash commands could be coordinated through hive intelligence",
            correction_guidance="Queen ZEN can plan optimal command sequences and error handling",
            priority_score=25
        )


class MemoryCoordinationAnalyzer(DriftAnalyzer):
    """Analyzes need for hive memory coordination."""
    
    def __init__(self, priority: int = 600):
        super().__init__(priority)
        self._decision_points = 0
        self._context_switches = 0
        
    def get_analyzer_name(self) -> str:
        return "memory_coordination_analyzer"
    
    def analyze_drift(self, tool_name: str, tool_input: Dict[str, Any], 
                     tool_response: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze need for memory coordination."""
        self.add_tool_usage(tool_name, tool_input, tool_response)
        
        # Detect decision points that need memory
        if self._is_decision_point(tool_name, tool_input):
            self._decision_points += 1
        
        # Detect context switches
        if self._is_context_switch(tool_name):
            self._context_switches += 1
        
        # Check if memory coordination is needed
        recent_tools = self.get_recent_tools(5)
        has_memory = any("memory" in t for t in recent_tools)
        
        if (self._decision_points >= 2 or self._context_switches >= 3) and not has_memory:
            return self._create_memory_coordination_evidence()
        
        return None
    
    def _is_decision_point(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Check if this tool represents a decision point."""
        decision_indicators = [
            tool_name == "Task",  # Subagent spawning
            tool_name == "Write" and len(tool_input.get("content", "")) > 500,  # Large content creation
            tool_name == "Bash" and any(cmd in tool_input.get("command", "") 
                                      for cmd in ["git", "npm", "pip", "cargo"]),  # Project commands
        ]
        return any(decision_indicators)
    
    def _is_context_switch(self, tool_name: str) -> bool:
        """Check if this represents a context switch."""
        recent_tools = self.get_recent_tools(3)
        if len(recent_tools) < 2:
            return False
        
        # Different tool categories indicate context switches
        categories = {
            "file": ["Read", "Write", "Edit", "MultiEdit"],
            "search": ["Grep", "Glob"],
            "execution": ["Bash", "Task"],
            "mcp_zen": [t for t in recent_tools if t.startswith("mcp__zen__")],
            "mcp_flow": [t for t in recent_tools if t.startswith("mcp__claude-flow__")],
            "mcp_fs": [t for t in recent_tools if t.startswith("mcp__filesystem__")]
        }
        
        current_category = None
        for category, tools in categories.items():
            if tool_name in tools or any(tool_name.startswith(prefix) for prefix in tools if prefix.startswith("mcp__")):
                current_category = category
                break
        
        previous_category = None
        for category, tools in categories.items():
            if recent_tools[-1] in tools or any(recent_tools[-1].startswith(prefix) for prefix in tools if prefix.startswith("mcp__")):
                previous_category = category
                break
        
        return current_category != previous_category and current_category is not None and previous_category is not None
    
    def _create_memory_coordination_evidence(self) -> DriftEvidence:
        """Create evidence for memory coordination need."""
        return DriftEvidence(
            drift_type=DriftType.MISSING_HIVE_MEMORY,
            severity=DriftSeverity.MODERATE,
            tool_sequence=self.get_recent_tools(),
            missing_tools=["mcp__claude-flow__memory_usage", "mcp__zen__consensus"],
            evidence_details=f"Complex workflow with {self._decision_points} decision points and {self._context_switches} context switches needs hive memory",
            correction_guidance="Hive memory ensures consistent decision-making and context preservation",
            priority_score=50
        )