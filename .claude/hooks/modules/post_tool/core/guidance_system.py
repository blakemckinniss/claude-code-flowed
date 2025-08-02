"""Guidance System - Provides non-blocking guidance using sys.exit(2) strategically."""

import sys
import json

# Path setup handled by centralized resolver when importing this module
from typing import List, Optional, Dict, Any
from .drift_detector import DriftEvidence, DriftSeverity, DriftGuidanceGenerator


class GuidanceEscalationManager:
    """Manages escalation of guidance based on drift severity and frequency."""
    
    def __init__(self):
        self._guidance_history: List[str] = []
        self._severity_counts = {severity: 0 for severity in DriftSeverity}
        self._consecutive_ignored = 0
        self._last_guidance_tool_count = 0
        
    def should_provide_guidance(self, evidence: DriftEvidence, current_tool_count: int) -> bool:
        """Determine if guidance should be provided based on escalation rules."""
        severity = evidence.severity
        
        # Always guide critical issues
        if severity == DriftSeverity.CRITICAL:
            return True
        
        # Escalate based on consecutive ignored guidance
        if self._consecutive_ignored >= 3 and severity in [DriftSeverity.MAJOR, DriftSeverity.MODERATE]:
            return True
        
        # Escalate based on frequency of same severity
        self._severity_counts[severity] += 1
        if self._severity_counts[severity] >= 3 and severity == DriftSeverity.MAJOR:
            return True
        
        # Provide guidance based on tool count threshold
        tools_since_last = current_tool_count - self._last_guidance_tool_count
        thresholds = {
            DriftSeverity.MINOR: 8,      # Every 8 tools for minor issues
            DriftSeverity.MODERATE: 5,   # Every 5 tools for moderate issues  
            DriftSeverity.MAJOR: 3,      # Every 3 tools for major issues
            DriftSeverity.CRITICAL: 1    # Immediately for critical issues
        }
        
        if tools_since_last >= thresholds.get(severity, 5):
            self._last_guidance_tool_count = current_tool_count
            return True
        
        return False
    
    def record_guidance_provided(self, evidence: DriftEvidence) -> None:
        """Record that guidance was provided."""
        self._guidance_history.append(evidence.drift_type.value)
        self._consecutive_ignored = 0  # Reset since we're providing guidance
    
    def record_guidance_ignored(self) -> None:
        """Record that guidance appears to have been ignored."""
        self._consecutive_ignored += 1
    
    def get_escalation_level(self) -> str:
        """Get current escalation level for guidance tone."""
        if self._consecutive_ignored >= 5:
            return "emergency"
        elif self._consecutive_ignored >= 3:
            return "urgent"
        elif self._consecutive_ignored >= 1:
            return "firm"
        else:
            return "gentle"


class GuidanceFormatter:
    """Formats guidance messages with appropriate tone and structure."""
    
    TONE_PREFIXES = {
        "gentle": "🐝 HIVE GUIDANCE",
        "firm": "👑 QUEEN ZEN'S ADVICE", 
        "urgent": "🚨 HIVE PROTOCOL ALERT",
        "emergency": "💥 HIVE EMERGENCY"
    }
    
    TONE_CLOSINGS = {
        "gentle": "Queen ZEN's wisdom is always available to guide the hive.",
        "firm": "The hive works best when following Queen ZEN's royal protocols.",
        "urgent": "Immediate course correction will restore hive efficiency!",
        "emergency": "CRITICAL: All workers must return to Queen ZEN's command immediately!"
    }
    
    def format_guidance(self, evidence: DriftEvidence, escalation_level: str) -> str:
        """Format guidance message with appropriate tone."""
        generator = DriftGuidanceGenerator()
        base_guidance = generator.generate_guidance(evidence)
        
        # Add tone-appropriate header
        prefix = self.TONE_PREFIXES.get(escalation_level, "🐝 HIVE GUIDANCE")
        closing = self.TONE_CLOSINGS.get(escalation_level, "Queen ZEN's wisdom guides the hive.")
        
        # Format final message
        formatted_parts = [
            f"═══ {prefix} ═══",
            "",
            base_guidance,
            "",
            f"⚡ {closing}",
            ""
        ]
        
        # Add escalation-specific elements
        if escalation_level in ["urgent", "emergency"]:
            formatted_parts.insert(-2, "🔄 IMMEDIATE ACTION REQUIRED - HIVE EFFICIENCY DEPENDS ON COMPLIANCE!")
            formatted_parts.insert(-2, "")
        
        return "\n".join(formatted_parts)


class NonBlockingGuidanceProvider:
    """Provides non-blocking guidance using strategic sys.exit(2) calls."""
    
    def __init__(self):
        self.escalation_manager = GuidanceEscalationManager()
        self.formatter = GuidanceFormatter()
        
    def provide_guidance(self, evidence_list: List[DriftEvidence], current_tool_count: int) -> Optional[Dict[str, Any]]:
        """Provide guidance if warranted, return None if no guidance needed."""
        if not evidence_list:
            return None
        
        # Sort evidence by priority score (highest first)
        sorted_evidence = sorted(evidence_list, key=lambda e: e.priority_score, reverse=True)
        primary_evidence = sorted_evidence[0]
        
        # Check if guidance should be provided
        if not self.escalation_manager.should_provide_guidance(primary_evidence, current_tool_count):
            self.escalation_manager.record_guidance_ignored()
            return None
        
        # Record guidance being provided
        self.escalation_manager.record_guidance_provided(primary_evidence)
        
        # Get escalation level and format guidance
        escalation_level = self.escalation_manager.get_escalation_level()
        guidance_message = self.formatter.format_guidance(primary_evidence, escalation_level)
        
        # Determine exit strategy based on severity
        should_block = self._should_block_execution(primary_evidence, escalation_level)
        
        return {
            "message": guidance_message,
            "should_block": should_block,
            "evidence": primary_evidence,
            "escalation_level": escalation_level
        }
    
    def _should_block_execution(self, evidence: DriftEvidence, escalation_level: str) -> bool:
        """Determine if execution should be blocked (sys.exit(2))."""
        # Block for critical severity or emergency escalation
        if evidence.severity == DriftSeverity.CRITICAL:
            return True
        
        if escalation_level == "emergency":
            return True
        
        # Block for high-priority major issues
        if evidence.severity == DriftSeverity.MAJOR and evidence.priority_score >= 70:
            return True
        
        # Otherwise, provide non-blocking guidance
        return False


class GuidanceOutputHandler:
    """Handles the actual output and exit behavior for guidance."""
    
    @staticmethod
    def handle_guidance_output(guidance_info: Dict[str, Any]) -> None:
        """Handle guidance output and exit appropriately."""
        if guidance_info["should_block"]:
            # Blocking guidance - sys.exit(2) to provide feedback to Claude
            print(guidance_info["message"], file=sys.stderr)
            sys.exit(2)
        else:
            # Non-blocking guidance - JSON output for sophisticated control
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "guidanceMessage": guidance_info["message"],
                    "driftDetected": True,
                    "escalationLevel": guidance_info["escalation_level"],
                    "evidence": {
                        "driftType": guidance_info["evidence"].drift_type.value,
                        "severity": guidance_info["evidence"].severity.name,
                        "priorityScore": guidance_info["evidence"].priority_score
                    }
                },
                "continue": True,  # Don't stop Claude, just provide guidance
                "suppressOutput": False  # Show in transcript
            }
            
            print(json.dumps(output, indent=2))
            sys.exit(0)
    
    @staticmethod
    def handle_no_guidance() -> None:
        """Handle case where no guidance is needed."""
        # Success - no blocking needed and no guidance to show
        sys.exit(0)


class ContextualGuidanceEnhancer:
    """Enhances guidance with contextual information about the current state."""
    
    def enhance_guidance(self, guidance_message: str, tool_name: str, 
                        tool_input: Dict[str, Any], recent_tools: List[str]) -> str:
        """Enhance guidance with contextual information."""
        context_parts = [
            guidance_message,
            "",
            "📍 CURRENT CONTEXT:",
            f"  • Last Tool: {tool_name}",
            f"  • Recent Sequence: {' → '.join(recent_tools[-5:])}"
        ]
        
        # Add specific context based on tool type
        if tool_name == "Write":
            file_path = tool_input.get("file_path", "unknown")
            context_parts.append(f"  • File: {file_path}")
        elif tool_name == "Bash":
            command = tool_input.get("command", "")[:50] + "..." if len(tool_input.get("command", "")) > 50 else tool_input.get("command", "")
            context_parts.append(f"  • Command: {command}")
        elif tool_name == "Task":
            description = tool_input.get("description", "")[:50] + "..." if len(tool_input.get("description", "")) > 50 else tool_input.get("description", "")
            context_parts.append(f"  • Task: {description}")
        
        # Add MCP tool suggestions based on context
        context_parts.extend([
            "",
            "🎯 IMMEDIATE HIVE ACTIONS:",
            "  1. mcp__zen__chat { prompt: \"Guide the current workflow\" }",
            "  2. mcp__zen__thinkdeep { step: \"Analyze and optimize approach\" }",
            "  3. mcp__claude-flow__memory_usage { action: \"store\", key: \"workflow/state\" }"
        ])
        
        return "\n".join(context_parts)