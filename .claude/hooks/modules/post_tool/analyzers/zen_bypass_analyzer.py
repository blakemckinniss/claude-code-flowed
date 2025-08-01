"""ZEN Bypass Analyzer - Detects when Queen ZEN is bypassed in workflows."""

from typing import Dict, Any, Optional
from ..core.drift_detector import DriftAnalyzer, DriftEvidence, DriftType, DriftSeverity


class ZenBypassAnalyzer(DriftAnalyzer):
    """Detects when Claude Code bypasses Queen ZEN coordination."""
    
    def __init__(self, priority: int = 1000):
        super().__init__(priority)
        self._non_zen_sequence_count = 0
        self._last_zen_usage = -1
        self._current_index = 0
        
    def get_analyzer_name(self) -> str:
        return "zen_bypass_analyzer"
    
    def analyze_drift(self, tool_name: str, tool_input: Dict[str, Any], 
                     tool_response: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze for ZEN bypass patterns."""
        self.add_tool_usage(tool_name, tool_input, tool_response)
        self._current_index += 1
        
        # Track ZEN usage
        if tool_name.startswith("mcp__zen__"):
            self._last_zen_usage = self._current_index
            self._non_zen_sequence_count = 0
            return None  # No drift when using ZEN
        
        # Increment non-ZEN sequence count
        self._non_zen_sequence_count += 1
        
        # Analyze different bypass scenarios
        drift_evidence = self._analyze_bypass_scenarios(tool_name, tool_input)
        
        return drift_evidence
    
    def _analyze_bypass_scenarios(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze specific bypass scenarios."""
        recent_tools = self.get_recent_tools(5)
        
        # Scenario 1: Direct Flow usage without ZEN
        if tool_name.startswith("mcp__claude-flow__") and not self.has_zen_coordination(5):
            return self._create_flow_bypass_evidence(tool_name, recent_tools)
        
        # Scenario 2: Direct Filesystem usage without ZEN coordination
        if tool_name.startswith("mcp__filesystem__") and not self.has_zen_coordination(3):
            return self._create_filesystem_bypass_evidence(tool_name, recent_tools)
        
        # Scenario 3: Extended native tool sequence without ZEN guidance
        if self._non_zen_sequence_count >= 4 and not tool_name.startswith("mcp__"):
            return self._create_extended_bypass_evidence(tool_name, recent_tools)
        
        # Scenario 4: Complex task initiation without ZEN planning
        if self._is_complex_task_start(tool_name, tool_input) and not self.has_zen_coordination(2):
            return self._create_complex_task_bypass_evidence(tool_name, recent_tools)
        
        return None
    
    def _create_flow_bypass_evidence(self, tool_name: str, recent_tools: list) -> DriftEvidence:
        """Create evidence for Flow Worker acting without Queen ZEN."""
        return DriftEvidence(
            drift_type=DriftType.BYPASSED_ZEN,
            severity=DriftSeverity.MODERATE,
            tool_sequence=recent_tools,
            missing_tools=["mcp__zen__chat", "mcp__zen__thinkdeep"],
            evidence_details=f"Flow Worker '{tool_name}' deployed without Queen ZEN's royal command",
            correction_guidance="Queen ZEN must issue royal decrees before Flow Worker deployment",
            priority_score=self.calculate_priority_score(DriftSeverity.MODERATE.value, self._non_zen_sequence_count)
        )
    
    def _create_filesystem_bypass_evidence(self, tool_name: str, recent_tools: list) -> DriftEvidence:
        """Create evidence for direct Filesystem usage without ZEN."""
        severity = DriftSeverity.MINOR if self._non_zen_sequence_count < 3 else DriftSeverity.MODERATE
        
        return DriftEvidence(
            drift_type=DriftType.BYPASSED_ZEN,
            severity=severity,
            tool_sequence=recent_tools,
            missing_tools=["mcp__zen__chat", "mcp__zen__planner"],
            evidence_details=f"Storage Worker '{tool_name}' activated without Queen ZEN's strategic guidance",
            correction_guidance="Queen ZEN should plan file operations for optimal hive coordination",
            priority_score=self.calculate_priority_score(severity.value, self._non_zen_sequence_count)
        )
    
    def _create_extended_bypass_evidence(self, tool_name: str, recent_tools: list) -> DriftEvidence:
        """Create evidence for extended sequence without ZEN."""
        severity = DriftSeverity.MAJOR if self._non_zen_sequence_count >= 6 else DriftSeverity.MODERATE
        
        return DriftEvidence(
            drift_type=DriftType.BYPASSED_ZEN,
            severity=severity,
            tool_sequence=recent_tools,
            missing_tools=["mcp__zen__chat", "mcp__zen__thinkdeep", "mcp__zen__planner"],
            evidence_details=f"Extended sequence of {self._non_zen_sequence_count} tools without Queen ZEN's wisdom",
            correction_guidance="Queen ZEN's strategic oversight required for complex multi-tool operations",
            priority_score=self.calculate_priority_score(severity.value, self._non_zen_sequence_count)
        )
    
    def _create_complex_task_bypass_evidence(self, tool_name: str, recent_tools: list) -> DriftEvidence:
        """Create evidence for complex task without ZEN planning."""
        return DriftEvidence(
            drift_type=DriftType.BYPASSED_ZEN,
            severity=DriftSeverity.MAJOR,
            tool_sequence=recent_tools,
            missing_tools=["mcp__zen__thinkdeep", "mcp__zen__planner", "mcp__zen__consensus"],
            evidence_details=f"Complex task '{tool_name}' initiated without Queen ZEN's deep strategic analysis",
            correction_guidance="Queen ZEN's thinkdeep analysis essential for complex task coordination",
            priority_score=self.calculate_priority_score(DriftSeverity.MAJOR.value, self._non_zen_sequence_count)
        )
    
    def _is_complex_task_start(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Determine if this appears to be the start of a complex task."""
        complex_indicators = [
            # Multi-file operations
            tool_name == "Write" and len(tool_input.get("content", "")) > 1000,
            tool_name == "MultiEdit" and len(tool_input.get("edits", [])) > 3,
            
            # Directory structure creation
            tool_name == "Bash" and any(cmd in tool_input.get("command", "") 
                                      for cmd in ["mkdir -p", "npm init", "git init"]),
            
            # Project setup patterns
            tool_name == "Write" and any(filename in tool_input.get("file_path", "")
                                       for filename in ["package.json", "requirements.txt", "Cargo.toml"]),
            
            # Task tool usage (subagents)
            tool_name == "Task"
        ]
        
        return any(complex_indicators)


class FlowCoordinationAnalyzer(DriftAnalyzer):
    """Analyzes Flow Worker coordination patterns."""
    
    def __init__(self, priority: int = 800):
        super().__init__(priority)
        self._flow_without_zen_count = 0
        
    def get_analyzer_name(self) -> str:
        return "flow_coordination_analyzer"
    
    def analyze_drift(self, tool_name: str, tool_input: Dict[str, Any], 
                     tool_response: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze Flow Worker coordination patterns."""
        self.add_tool_usage(tool_name, tool_input, tool_response)
        
        # Check for Flow usage without ZEN
        if tool_name.startswith("mcp__claude-flow__"):
            if not self.has_zen_coordination(3):
                self._flow_without_zen_count += 1
                return self._create_uncoordinated_flow_evidence(tool_name)
        
        # Check for swarm operations without proper hierarchy
        if tool_name.startswith("mcp__claude-flow__swarm"):
            return self._analyze_swarm_coordination(tool_name, tool_input)
        
        return None
    
    def _create_uncoordinated_flow_evidence(self, tool_name: str) -> DriftEvidence:
        """Create evidence for uncoordinated Flow Worker usage."""
        severity = (DriftSeverity.MAJOR if self._flow_without_zen_count > 2 
                   else DriftSeverity.MODERATE)
        
        return DriftEvidence(
            drift_type=DriftType.BYPASSED_ZEN,
            severity=severity,
            tool_sequence=self.get_recent_tools(),
            missing_tools=["mcp__zen__chat", "mcp__zen__thinkdeep"],
            evidence_details=f"Flow Worker '{tool_name}' operating independently without Queen ZEN's command",
            correction_guidance="Flow Workers must receive orders from Queen ZEN before coordinating operations",
            priority_score=self.calculate_priority_score(severity.value, self._flow_without_zen_count)
        )
    
    def _analyze_swarm_coordination(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze swarm coordination patterns."""
        # Check for proper swarm hierarchy
        topology = tool_input.get("topology", "")
        max_agents = tool_input.get("maxAgents", 0)
        
        # Detect potentially chaotic swarm configurations
        if topology not in ["hierarchical", "mesh"] and max_agents > 5:
            return DriftEvidence(
                drift_type=DriftType.WRONG_TOOL_ORDER,
                severity=DriftSeverity.MODERATE,
                tool_sequence=self.get_recent_tools(),
                missing_tools=["mcp__zen__planner"],
                evidence_details=f"Swarm topology '{topology}' with {max_agents} agents may lack proper hierarchy",
                correction_guidance="Queen ZEN should plan optimal swarm topology before deployment",
                priority_score=50
            )
        
        return None


class NativeToolOveruseAnalyzer(DriftAnalyzer):
    """Detects overuse of native tools without MCP coordination."""
    
    def __init__(self, priority: int = 600):
        super().__init__(priority)
        self._consecutive_native_count = 0
        
    def get_analyzer_name(self) -> str:
        return "native_overuse_analyzer"
    
    def analyze_drift(self, tool_name: str, tool_input: Dict[str, Any], 
                     tool_response: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze native tool overuse patterns."""
        self.add_tool_usage(tool_name, tool_input, tool_response)
        
        # Track consecutive native tool usage
        if tool_name.startswith("mcp__"):
            self._consecutive_native_count = 0
            return None
        else:
            self._consecutive_native_count += 1
        
        # Trigger at different thresholds
        if self._consecutive_native_count == 3:  # Early warning
            return self._create_early_warning_evidence()
        elif self._consecutive_native_count == 5:  # Moderate concern
            return self._create_moderate_concern_evidence()
        elif self._consecutive_native_count >= 7:  # Major drift
            return self._create_major_drift_evidence()
        
        return None
    
    def _create_early_warning_evidence(self) -> DriftEvidence:
        """Create early warning for native tool overuse."""
        return DriftEvidence(
            drift_type=DriftType.EXCESSIVE_NATIVE_TOOLS,
            severity=DriftSeverity.MINOR,
            tool_sequence=self.get_recent_tools(),
            missing_tools=["mcp__zen__chat", "mcp__claude-flow__swarm_init"],
            evidence_details=f"3 consecutive native tools - consider MCP coordination for efficiency",
            correction_guidance="Queen ZEN's hive coordination can optimize these operations",
            priority_score=20
        )
    
    def _create_moderate_concern_evidence(self) -> DriftEvidence:
        """Create moderate concern for extended native usage."""
        return DriftEvidence(
            drift_type=DriftType.EXCESSIVE_NATIVE_TOOLS,
            severity=DriftSeverity.MODERATE,
            tool_sequence=self.get_recent_tools(),
            missing_tools=["mcp__zen__thinkdeep", "mcp__claude-flow__memory_usage"],
            evidence_details=f"5 consecutive native tools without hive coordination",
            correction_guidance="Hive intelligence can provide better coordination and memory management",
            priority_score=40
        )
    
    def _create_major_drift_evidence(self) -> DriftEvidence:
        """Create major drift evidence for excessive native usage."""
        return DriftEvidence(
            drift_type=DriftType.EXCESSIVE_NATIVE_TOOLS,
            severity=DriftSeverity.MAJOR,
            tool_sequence=self.get_recent_tools(),
            missing_tools=["mcp__zen__consensus", "mcp__claude-flow__swarm_init"],
            evidence_details=f"Extended sequence of {self._consecutive_native_count} native tools - hive coordination required",
            correction_guidance="Queen ZEN's royal intervention needed to restore hive efficiency",
            priority_score=70
        )