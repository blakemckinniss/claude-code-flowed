"""Core drift detection system for post-tool analysis.

Detects when Claude Code is drifting away from the proper MCP hierarchy:
Queen ZEN → Flow Workers → Storage Workers → Execution Drones
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import re


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = 0
    MINOR = 1      # Gentle reminder
    MODERATE = 2   # Clear guidance needed
    MAJOR = 3      # Strong correction required
    CRITICAL = 4   # Immediate intervention


class DriftType(Enum):
    """Types of drift from proper MCP workflow."""
    BYPASSED_ZEN = "bypassed_zen"              # Went straight to Flow/FS without ZEN
    SKIPPED_FLOW = "skipped_flow"              # Went ZEN → Filesystem directly
    NO_MCP_COORDINATION = "no_mcp_coordination" # Using only Claude Code tools
    WRONG_TOOL_ORDER = "wrong_tool_order"      # Tools used in wrong sequence
    EXCESSIVE_NATIVE_TOOLS = "excessive_native" # Too many native tools vs MCP
    MISSING_HIVE_MEMORY = "missing_hive_memory" # No memory coordination
    FRAGMENTED_WORKFLOW = "fragmented_workflow" # Scattered, uncoordinated actions


@dataclass
class DriftEvidence:
    """Evidence of workflow drift."""
    drift_type: DriftType
    severity: DriftSeverity
    tool_sequence: List[str]
    missing_tools: List[str]
    evidence_details: str
    correction_guidance: str
    priority_score: int = 0


class DriftAnalyzer(ABC):
    """Base class for drift detection analyzers."""
    
    def __init__(self, priority: int = 0):
        self.priority = priority
        self._tool_history: List[Dict[str, Any]] = []
        self._mcp_usage_count = 0
        self._native_usage_count = 0
        
    @abstractmethod
    def analyze_drift(self, tool_name: str, tool_input: Dict[str, Any], 
                     tool_response: Dict[str, Any]) -> Optional[DriftEvidence]:
        """Analyze a tool use for potential drift."""
        pass
    
    @abstractmethod
    def get_analyzer_name(self) -> str:
        """Get the name of this analyzer."""
        pass
    
    def add_tool_usage(self, tool_name: str, tool_input: Dict[str, Any], 
                      tool_response: Dict[str, Any]) -> None:
        """Track tool usage for pattern analysis."""
        self._tool_history.append({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "timestamp": None  # Could add actual timestamp if needed
        })
        
        # Track MCP vs native tool usage
        if tool_name.startswith("mcp__"):
            self._mcp_usage_count += 1
        else:
            self._native_usage_count += 1
    
    def get_recent_tools(self, count: int = 10) -> List[str]:
        """Get recent tool names for pattern analysis."""
        return [t["tool_name"] for t in self._tool_history[-count:]]
    
    def get_mcp_ratio(self) -> float:
        """Get ratio of MCP tools to total tools used."""
        total = self._mcp_usage_count + self._native_usage_count
        return self._mcp_usage_count / total if total > 0 else 0.0
    
    def has_zen_coordination(self, lookback: int = 5) -> bool:
        """Check if recent tools included ZEN coordination."""
        recent_tools = self.get_recent_tools(lookback)
        return any("mcp__zen__" in tool for tool in recent_tools)
    
    def has_flow_coordination(self, lookback: int = 5) -> bool:
        """Check if recent tools included Flow coordination."""
        recent_tools = self.get_recent_tools(lookback)
        return any("mcp__claude-flow__" in tool for tool in recent_tools)
    
    def calculate_priority_score(self, base_severity: int, tool_count: int) -> int:
        """Calculate priority score for drift correction."""
        # Higher scores = more urgent correction needed
        score = base_severity * 10
        
        # Escalate based on tool count without coordination
        if tool_count > 5:
            score += 20
        elif tool_count > 3:
            score += 10
        
        # Escalate if MCP ratio is too low
        mcp_ratio = self.get_mcp_ratio()
        if mcp_ratio < 0.3:  # Less than 30% MCP usage
            score += 15
        elif mcp_ratio < 0.5:  # Less than 50% MCP usage
            score += 10
        
        return score


class HiveWorkflowValidator:
    """Validates adherence to Queen ZEN's hive workflow."""
    
    IDEAL_WORKFLOW_PATTERNS = [
        # Pattern 1: Full hive coordination
        ["mcp__zen__", "mcp__claude-flow__", "mcp__filesystem__", "Write|Edit|Read"],
        
        # Pattern 2: ZEN direct to filesystem (acceptable for simple tasks)
        ["mcp__zen__", "mcp__filesystem__", "Write|Edit|Read"],
        
        # Pattern 3: ZEN planning then native execution (acceptable)
        ["mcp__zen__", "Write|Edit|Read|Bash"],
        
        # Pattern 4: Flow coordination with memory
        ["mcp__claude-flow__", "mcp__filesystem__", "Write|Edit"],
    ]
    
    DRIFT_PATTERNS = [
        # Pattern 1: Direct to filesystem without any MCP coordination
        {
            "pattern": ["mcp__filesystem__", "Write|Edit"],
            "drift_type": DriftType.NO_MCP_COORDINATION,
            "severity": DriftSeverity.MODERATE,
            "message": "Bypassed Queen ZEN and Flow Workers - hive coordination required!"
        },
        
        # Pattern 2: Only native tools for extended sequence
        {
            "pattern": ["Write", "Edit", "Read", "Bash"],
            "drift_type": DriftType.NO_MCP_COORDINATION,
            "severity": DriftSeverity.MAJOR,
            "message": "Extended native tool usage without hive coordination - Queen ZEN's guidance needed!"
        },
        
        # Pattern 3: Flow without ZEN (worker acting independently)
        {
            "pattern": ["mcp__claude-flow__", "Write|Edit"],
            "drift_type": DriftType.BYPASSED_ZEN,
            "severity": DriftSeverity.MODERATE,
            "message": "Flow Workers acting independently - Queen ZEN must command the hive!"
        }
    ]
    
    def validate_workflow_adherence(self, tool_sequence: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate if tool sequence follows hive workflow patterns."""
        # Check against ideal patterns
        for ideal_pattern in self.IDEAL_WORKFLOW_PATTERNS:
            if self._matches_pattern(tool_sequence, ideal_pattern):
                return True, "Following Queen ZEN's hive workflow correctly"
        
        # Check for drift patterns
        for drift_pattern in self.DRIFT_PATTERNS:
            if self._matches_pattern(tool_sequence, drift_pattern["pattern"]):
                return False, drift_pattern["message"]
        
        return True, None  # No clear drift detected
    
    def _matches_pattern(self, tool_sequence: List[str], pattern: List[str]) -> bool:
        """Check if tool sequence matches a given pattern."""
        if len(tool_sequence) < len(pattern):
            return False
        
        # Check if the sequence contains the pattern in order
        pattern_idx = 0
        for tool in tool_sequence:
            if pattern_idx < len(pattern):
                pattern_regex = pattern[pattern_idx]
                if re.search(pattern_regex, tool):
                    pattern_idx += 1
        
        return pattern_idx == len(pattern)


class DriftGuidanceGenerator:
    """Generates guidance messages to correct workflow drift."""
    
    GUIDANCE_TEMPLATES = {
        DriftType.BYPASSED_ZEN: {
            DriftSeverity.MINOR: "👑 Consider starting with Queen ZEN's guidance (mcp__zen__chat) for better coordination",
            DriftSeverity.MODERATE: "👑 HIVE PROTOCOL VIOLATION: Queen ZEN must command before worker deployment!",
            DriftSeverity.MAJOR: "🚨 CRITICAL: Hive chaos detected! Queen ZEN's royal decree required immediately!",
            DriftSeverity.CRITICAL: "💥 HIVE EMERGENCY: All workers must return to Queen ZEN for immediate coordination!"
        },
        
        DriftType.SKIPPED_FLOW: {
            DriftSeverity.MINOR: "🤖 Consider using Flow Workers (mcp__claude-flow__swarm_init) for better coordination",
            DriftSeverity.MODERATE: "🤖 Flow Workers bypassed - hive coordination may be suboptimal",
            DriftSeverity.MAJOR: "🚨 Flow Workers required for complex operations - deploy immediately!",
            DriftSeverity.CRITICAL: "💥 HIVE BREAKDOWN: Flow Workers must coordinate all complex operations!"
        },
        
        DriftType.NO_MCP_COORDINATION: {
            DriftSeverity.MINOR: "🐝 Hive tools available: Consider mcp__zen__ or mcp__claude-flow__ for coordination",
            DriftSeverity.MODERATE: "🐝 HIVE GUIDANCE: Queen ZEN's MCP tools provide superior coordination",
            DriftSeverity.MAJOR: "🚨 HIVE VIOLATION: MCP coordination mandatory for complex workflows!",
            DriftSeverity.CRITICAL: "💥 HIVE EMERGENCY: Return to MCP coordination immediately!"
        },
        
        DriftType.EXCESSIVE_NATIVE_TOOLS: {
            DriftSeverity.MINOR: "⚡ Consider batching operations through MCP tools for efficiency",
            DriftSeverity.MODERATE: "⚡ EFFICIENCY WARNING: MCP tools provide better batching and coordination",
            DriftSeverity.MAJOR: "🚨 PERFORMANCE ALERT: Excessive native tool usage - MCP coordination required!",
            DriftSeverity.CRITICAL: "💥 WORKFLOW BREAKDOWN: Switch to MCP-coordinated operations now!"
        }
    }
    
    CORRECTION_COMMANDS = {
        DriftType.BYPASSED_ZEN: [
            "mcp__zen__chat { prompt: \"Guide the current task with hive coordination\" }",
            "mcp__zen__thinkdeep { step: \"Plan systematic approach\", thinking_mode: \"high\" }"
        ],
        
        DriftType.SKIPPED_FLOW: [
            "mcp__claude-flow__swarm_init { topology: \"hierarchical\", maxAgents: 5 }",
            "mcp__claude-flow__memory_usage { action: \"store\", key: \"workflow/coordination\" }"
        ],
        
        DriftType.NO_MCP_COORDINATION: [
            "mcp__zen__chat { prompt: \"Coordinate this workflow with hive intelligence\" }",
            "mcp__claude-flow__swarm_init { topology: \"mesh\", strategy: \"parallel\" }"
        ]
    }
    
    def generate_guidance(self, drift_evidence: DriftEvidence) -> str:
        """Generate guidance message for detected drift."""
        drift_type = drift_evidence.drift_type
        severity = drift_evidence.severity
        
        # Get base message
        base_message = self.GUIDANCE_TEMPLATES.get(drift_type, {}).get(
            severity, "🐝 Consider using Queen ZEN's hive coordination for better results"
        )
        
        # Add specific evidence and corrections
        guidance_parts = [
            f"🔍 DRIFT DETECTED: {drift_evidence.evidence_details}",
            f"👑 {base_message}",
            "",
            "🛠️ RECOMMENDED CORRECTIONS:"
        ]
        
        # Add correction commands
        corrections = self.CORRECTION_COMMANDS.get(drift_type, [])
        for i, correction in enumerate(corrections[:2], 1):  # Show top 2 corrections
            guidance_parts.append(f"  {i}. {correction}")
        
        if drift_evidence.missing_tools:
            guidance_parts.extend([
                "",
                f"🎯 MISSING HIVE TOOLS: {', '.join(drift_evidence.missing_tools)}"
            ])
        
        # Add workflow reminder for severe drift
        if severity in [DriftSeverity.MAJOR, DriftSeverity.CRITICAL]:
            guidance_parts.extend([
                "",
                "👑 HIVE HIERARCHY REMINDER:",
                "  Queen ZEN → Flow Workers → Storage Workers → Execution Drones",
                "  🚨 All workers must serve Queen ZEN's royal decrees!"
            ])
        
        return "\n".join(guidance_parts)