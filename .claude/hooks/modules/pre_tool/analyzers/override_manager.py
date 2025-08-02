"""Override manager for anti-pattern validators.

Provides a controlled way to bypass validators when necessary,
with proper justification and tracking.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class OverrideManager:
    """Manages override requests for anti-pattern validators."""
    
    # Magic keywords that trigger override consideration
    OVERRIDE_TRIGGERS = [
        "FORCE:",
        "OVERRIDE:",
        "BYPASS:",
        "IGNORE_VALIDATION:",
        "QUEEN_ZEN_APPROVED:",
        "EMERGENCY:",
        "HOTFIX:",
        "CRITICAL_FIX:"
    ]
    
    # Validators that can never be overridden (for safety)
    NON_OVERRIDABLE = [
        "safety_validator",  # Never bypass safety checks
    ]
    
    def __init__(self):
        self.override_log_path = Path("/home/devcontainers/flowed/.claude/hooks/logs/overrides.json")
        self.override_history: List[Dict[str, Any]] = []
        self._load_override_history()
    
    def check_for_override_request(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[Tuple[bool, str]]:
        """Check if the tool input contains an override request.
        
        Returns:
            Tuple of (should_override, justification) or None
        """
        # Check various fields for override triggers
        fields_to_check = [
            tool_input.get("command", ""),
            tool_input.get("content", ""),
            tool_input.get("description", ""),
            tool_input.get("message", ""),
            tool_input.get("comment", ""),
        ]
        
        for field in fields_to_check:
            if not isinstance(field, str):
                continue
                
            for trigger in self.OVERRIDE_TRIGGERS:
                if trigger in field:
                    # Extract justification
                    justification = self._extract_justification(field, trigger)
                    if justification:
                        return (True, justification)
        
        return None
    
    def should_allow_override(self, validator_name: str, justification: str, 
                            severity: ValidationSeverity) -> bool:
        """Determine if an override should be allowed.
        
        Rules:
        1. Safety validators cannot be overridden
        2. CRITICAL severity requires Queen ZEN approval
        3. Must have valid justification
        """
        # Never override safety validators
        if validator_name in self.NON_OVERRIDABLE:
            return False
        
        # Check if justification is sufficient
        if len(justification.strip()) < 10:
            return False
        
        # CRITICAL severity requires special approval
        if severity == ValidationSeverity.CRITICAL:
            return "QUEEN_ZEN_APPROVED" in justification or "EMERGENCY" in justification
        
        return True
    
    def log_override(self, validator_name: str, tool_name: str, 
                    justification: str, severity: ValidationSeverity) -> None:
        """Log an override for audit purposes."""
        override_entry = {
            "timestamp": datetime.now().isoformat(),
            "validator": validator_name,
            "tool": tool_name,
            "justification": justification,
            "severity": severity.name,
            "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown")
        }
        
        self.override_history.append(override_entry)
        self._save_override_history()
    
    def get_override_statistics(self) -> Dict[str, Any]:
        """Get statistics about overrides."""
        if not self.override_history:
            return {"total_overrides": 0}
        
        stats = {
            "total_overrides": len(self.override_history),
            "by_validator": {},
            "by_severity": {},
            "recent_overrides": self.override_history[-5:]
        }
        
        for entry in self.override_history:
            validator = entry.get("validator", "unknown")
            severity = entry.get("severity", "unknown")
            
            stats["by_validator"][validator] = stats["by_validator"].get(validator, 0) + 1
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        
        return stats
    
    def _extract_justification(self, text: str, trigger: str) -> str:
        """Extract justification from override request."""
        # Find the trigger position
        trigger_pos = text.find(trigger)
        if trigger_pos == -1:
            return ""
        
        # Extract text after trigger
        justification = text[trigger_pos + len(trigger):].strip()
        
        # Take first line or up to 200 characters
        if '\n' in justification:
            justification = justification.split('\n')[0]
        
        return justification[:200]
    
    def _load_override_history(self) -> None:
        """Load override history from file."""
        try:
            if self.override_log_path.exists():
                with open(self.override_log_path) as f:
                    self.override_history = json.load(f)
        except Exception:
            self.override_history = []
    
    def _save_override_history(self) -> None:
        """Save override history to file."""
        try:
            self.override_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.override_log_path, 'w') as f:
                json.dump(self.override_history, f, indent=2)
        except Exception:
            pass  # Fail silently - logging shouldn't break operations


class OverrideValidator(HiveWorkflowValidator):
    """Special validator that checks for override requests."""
    
    def __init__(self, priority: int = 999):  # Highest priority
        super().__init__(priority)
        self.override_manager = OverrideManager()
    
    def get_validator_name(self) -> str:
        return "override_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Check for override requests and provide guidance."""
        
        # Check if there's an override request
        override_info = self.override_manager.check_for_override_request(tool_name, tool_input)
        
        if override_info:
            _, justification = override_info
            
            # Return info-level result to inform about override capability
            return ValidationResult(
                severity=ValidationSeverity.ALLOW,
                violation_type=None,
                message=f"🔓 Override request detected: {justification[:50]}...",
                suggested_alternative=None,
                hive_guidance="Override will be considered if other validators block this operation",
                priority_score=1  # Low score - this is just informational
            )
        
        return None