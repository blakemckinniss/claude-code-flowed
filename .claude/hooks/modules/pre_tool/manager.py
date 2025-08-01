"""Pre-tool analysis manager - coordinates all workflow validators.

Central coordination system for Queen ZEN → Flow Workers → Storage Workers
hierarchy validation and proactive optimization guidance.
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional, Type
from .core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult, 
    ValidationSeverity,
    WorkflowContextTracker
)
from .analyzers import (
    ZenHierarchyValidator,
    EfficiencyOptimizer,
    SafetyValidator,
    MCPCoordinationValidator,
    HiveWorkflowOptimizer,
    NeuralPatternValidator,
    GitHubCoordinatorAnalyzer,
    GitHubPRAnalyzer,
    GitHubIssueAnalyzer,
    GitHubReleaseAnalyzer,
    GitHubRepoAnalyzer,
    GitHubSyncAnalyzer
)


class PreToolAnalysisConfig:
    """Configuration for pre-tool analysis system."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default config path relative to hooks directory  
            hooks_dir = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(hooks_dir, "pre_tool_config.json")
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load pre-tool config: {e}", file=sys.stderr)
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "enabled_validators": [
                "zen_hierarchy_validator",
                "mcp_coordination_validator", 
                "hive_workflow_optimizer",
                "neural_pattern_validator",
                "github_coordinator_analyzer",
                "github_pr_analyzer",
                "github_issue_analyzer",
                "github_release_analyzer",
                "github_repo_analyzer",
                "github_sync_analyzer",
                "efficiency_optimizer",
                "safety_validator"
            ],
            "validation_settings": {
                "block_dangerous_operations": True,
                "suggest_optimizations": True,
                "require_zen_for_complex": True,
                "optimization_threshold": 30
            },
            "hive_intelligence": {
                "auto_suggest_coordination": True,
                "pattern_detection_enabled": True,
                "workflow_optimization": True,
                "memory_integration": True
            },
            "blocking_behavior": {
                "block_on_critical": True,
                "block_on_dangerous": True,
                "allow_overrides": False
            }
        }
    
    def is_validator_enabled(self, validator_name: str) -> bool:
        """Check if a validator is enabled."""
        return validator_name in self._config.get("enabled_validators", [])
    
    def get_validation_settings(self) -> Dict[str, Any]:
        """Get validation settings."""
        return self._config.get("validation_settings", {})
    
    def get_hive_intelligence_settings(self) -> Dict[str, Any]:
        """Get hive intelligence settings."""
        return self._config.get("hive_intelligence", {})
    
    def get_blocking_settings(self) -> Dict[str, Any]:
        """Get blocking behavior settings."""
        return self._config.get("blocking_behavior", {})


class PreToolAnalysisManager:
    """Coordinates all pre-tool workflow validation and optimization."""
    
    # Registry of available validators
    VALIDATOR_REGISTRY: Dict[str, Type[HiveWorkflowValidator]] = {
        "zen_hierarchy_validator": ZenHierarchyValidator,
        "mcp_coordination_validator": MCPCoordinationValidator,
        "hive_workflow_optimizer": HiveWorkflowOptimizer,
        "neural_pattern_validator": NeuralPatternValidator,
        "github_coordinator_analyzer": GitHubCoordinatorAnalyzer,
        "github_pr_analyzer": GitHubPRAnalyzer,
        "github_issue_analyzer": GitHubIssueAnalyzer,
        "github_release_analyzer": GitHubReleaseAnalyzer,
        "github_repo_analyzer": GitHubRepoAnalyzer,
        "github_sync_analyzer": GitHubSyncAnalyzer,
        "efficiency_optimizer": EfficiencyOptimizer,
        "safety_validator": SafetyValidator
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = PreToolAnalysisConfig(config_path)
        self.validators: List[HiveWorkflowValidator] = []
        self.context_tracker = WorkflowContextTracker()
        self.validation_count = 0
        
        self._initialize_validators()
    
    def _initialize_validators(self) -> None:
        """Initialize enabled validators."""
        for validator_name, validator_class in self.VALIDATOR_REGISTRY.items():
            if self.config.is_validator_enabled(validator_name):
                priority = self._get_validator_priority(validator_name)
                validator = validator_class(priority=priority)
                self.validators.append(validator)
        
        # Sort by priority (highest first)
        self.validators.sort(key=lambda v: v.priority, reverse=True)
    
    def _get_validator_priority(self, validator_name: str) -> int:
        """Get priority for a validator."""
        priorities = {
            "zen_hierarchy_validator": 1000,       # Highest - Queen ZEN is supreme
            "safety_validator": 950,               # Very High - Safety first
            "mcp_coordination_validator": 900,     # High - MCP optimization critical  
            "neural_pattern_validator": 850,       # High - Neural learning intelligence
            "github_coordinator_analyzer": 825,   # High - GitHub workflow intelligence
            "github_pr_analyzer": 820,            # High - PR workflow intelligence
            "github_issue_analyzer": 815,         # High - Issue workflow intelligence  
            "github_release_analyzer": 810,       # High - Release workflow intelligence
            "github_repo_analyzer": 805,          # High - Repository health intelligence
            "github_sync_analyzer": 800,          # High - Multi-repo sync intelligence
            "hive_workflow_optimizer": 750,        # High - Workflow intelligence
            "efficiency_optimizer": 600            # Medium - General efficiency
        }
        return priorities.get(validator_name, 500)
    
    def validate_tool_usage(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate tool usage and provide guidance if needed.
        
        Returns guidance dict if validation requires intervention, None if tool should proceed.
        """
        self.validation_count += 1
        
        # Update context tracking
        self.context_tracker.add_tool_context(tool_name)
        
        # Collect validation results from all validators
        all_results: List[ValidationResult] = []
        
        for validator in self.validators:
            try:
                result = validator.validate_workflow(tool_name, tool_input, self.context_tracker)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Warning: Validator {validator.get_validator_name()} failed: {e}", file=sys.stderr)
        
        # Process validation results
        if all_results:
            return self._process_validation_results(all_results, tool_name, tool_input)
        
        return None
    
    def _process_validation_results(self, results: List[ValidationResult], 
                                   tool_name: str, tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process validation results and determine guidance response."""
        
        # Sort by priority score (highest first) and severity
        results.sort(key=lambda r: (r.severity.value, -r.priority_score), reverse=True)
        
        # Get the highest priority result
        primary_result = results[0]
        
        # Determine if we should block execution
        should_block = self._should_block_execution(primary_result)
        
        # Generate guidance message
        guidance_message = self._generate_guidance_message(primary_result, results, tool_name)
        
        # Return guidance information
        return {
            "severity": primary_result.severity.name,
            "should_block": should_block,
            "message": guidance_message,
            "suggested_alternative": primary_result.suggested_alternative,
            "hive_guidance": primary_result.hive_guidance,
            "violation_type": primary_result.violation_type.value if primary_result.violation_type else None,
            "priority_score": primary_result.priority_score,
            "total_validators_triggered": len(results)
        }
    
    def _should_block_execution(self, result: ValidationResult) -> bool:
        """Determine if execution should be blocked."""
        blocking_settings = self.config.get_blocking_settings()
        
        # Always block critical severity if configured
        if result.severity == ValidationSeverity.CRITICAL and blocking_settings.get("block_on_critical", True):
            return True
        
        # Block dangerous operations if configured
        if result.severity == ValidationSeverity.BLOCK and blocking_settings.get("block_on_dangerous", True):
            return True
        
        # Check for specific blocking reasons
        if result.blocking_reason and not blocking_settings.get("allow_overrides", False):
            return True
        
        return False
    
    def _generate_guidance_message(self, primary_result: ValidationResult, 
                                  all_results: List[ValidationResult], tool_name: str) -> str:
        """Generate comprehensive guidance message."""
        
        # Start with primary message  
        message_parts = [f"👑 QUEEN ZEN'S HIVE INTELLIGENCE:"]
        message_parts.append(f"   {primary_result.message}")
        
        # Add hive guidance if available
        if primary_result.hive_guidance:
            message_parts.append(f"   🐝 {primary_result.hive_guidance}")
        
        # Add suggested alternative if available
        if primary_result.suggested_alternative:
            message_parts.append(f"   💡 Suggested: {primary_result.suggested_alternative}")
        
        # Add additional insights from other validators (max 2)
        other_results = [r for r in all_results[1:3] if r.severity != ValidationSeverity.ALLOW]
        for result in other_results:
            if result.message != primary_result.message:
                message_parts.append(f"   ➕ {result.message}")
        
        # Add workflow context if relevant
        context_info = self._get_workflow_context_info()
        if context_info:
            message_parts.append(f"   📊 Context: {context_info}")
        
        return "\n".join(message_parts)
    
    def _get_workflow_context_info(self) -> str:
        """Get relevant workflow context information."""
        coord_state = self.context_tracker.get_coordination_state()
        tools_since_zen = self.context_tracker.get_tools_since_zen()
        recent_pattern = self.context_tracker.get_recent_pattern()
        
        context_parts = []
        
        if coord_state == "disconnected" and tools_since_zen > 3:
            context_parts.append(f"{tools_since_zen} tools since Queen ZEN")
        elif coord_state == "coordinated":
            context_parts.append("Optimal hive coordination active")
        
        if recent_pattern:
            context_parts.append(f"Pattern: {recent_pattern}")
        
        return " | ".join(context_parts)
    
    def get_validator_status(self) -> Dict[str, Any]:
        """Get status of all validators for debugging."""
        return {
            "total_validations": self.validation_count,
            "active_validators": [v.get_validator_name() for v in self.validators],
            "validator_priorities": {v.get_validator_name(): v.priority for v in self.validators},
            "coordination_state": self.context_tracker.get_coordination_state(),
            "tools_since_zen": self.context_tracker.get_tools_since_zen(),
            "tools_since_flow": self.context_tracker.get_tools_since_flow(),
            "config_path": self.config.config_path
        }


class GuidanceOutputHandler:
    """Handles guidance output and exit behavior."""
    
    @staticmethod
    def handle_validation_guidance(guidance_info: Dict[str, Any]) -> None:
        """Handle validation guidance output and exit appropriately."""
        
        # Output guidance message to stderr for Claude Code to see
        print(guidance_info["message"], file=sys.stderr)
        
        # Exit with appropriate code
        if guidance_info["should_block"]:
            # Block tool execution with sys.exit(2) 
            sys.exit(2)
        else:
            # Allow tool execution to proceed
            sys.exit(0)
    
    @staticmethod
    def handle_no_guidance() -> None:
        """Handle case where no guidance is needed."""
        # Tool usage is optimal - allow execution
        sys.exit(0)


class DebugValidationReporter:
    """Provides detailed validation reports for debugging."""  
    
    def __init__(self, manager: PreToolAnalysisManager):
        self.manager = manager
    
    def generate_debug_report(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Generate detailed debug report for troubleshooting."""
        report_parts = [
            "=== PRE-TOOL VALIDATION DEBUG REPORT ===",
            f"Tool: {tool_name}",
            f"Total Validations: {self.manager.validation_count}",
            f"Active Validators: {len(self.manager.validators)}",
            ""
        ]
        
        # Validator details
        for validator in self.manager.validators:
            report_parts.extend([
                f"Validator: {validator.get_validator_name()}",
                f"  Priority: {validator.priority}",
                ""
            ])
        
        # Context tracking details
        tracker = self.manager.context_tracker
        report_parts.extend([
            "Context Tracking:",
            f"  Coordination State: {tracker.get_coordination_state()}",
            f"  Tools Since ZEN: {tracker.get_tools_since_zen()}",
            f"  Tools Since Flow: {tracker.get_tools_since_flow()}", 
            f"  Recent Pattern: {tracker.get_recent_pattern()}",
            ""
        ])
        
        return "\n".join(report_parts)
    
    def log_debug_info(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """Log debug information if debugging is enabled."""
        debug_env = os.environ.get("CLAUDE_HOOKS_DEBUG", "").lower()
        if debug_env in ["true", "1", "on"]:
            report = self.generate_debug_report(tool_name, tool_input)
            print(f"DEBUG: {report}", file=sys.stderr)