#!/usr/bin/env python3
"""Slimmed Pre-Tool Analysis Manager - Single Responsibility Component.

Facade pattern implementation that delegates to ValidatorRegistry and ValidationCoordinator.
Focuses solely on configuration, result processing, and guidance generation.
"""

import json
import os
import sys
from typing import Dict, Any, Optional, List, Type
from ..core.workflow_validator import (
    HiveWorkflowValidator, 
    ValidationResult, 
    ValidationSeverity,
    WorkflowContextTracker
)
from .validator_registry import ValidatorRegistry
from .validation_coordinator import ValidationCoordinator


class PreToolAnalysisConfig:
    """Configuration management for pre-tool analysis system."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default config path relative to hooks directory  
            hooks_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(hooks_dir, "pre_tool_config.json")
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
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
                "safety_validator",
                "concurrent_execution_validator",
                "agent_patterns_validator",
                "visual_formats_validator",
                "mcp_separation_validator",
                "duplication_detection_validator",
                "rogue_system_validator",
                "conflicting_architecture_validator",
                "overwrite_protection_validator",
                "claude_flow_suggester",
                "multi_model_consensus_validator"
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
            },
            "caching": {
                "enable_validation_cache": True,
                "cache_stale_results": True,
                "max_cache_entries": 2000
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


class SlimmedPreToolAnalysisManager:
    """Slimmed manager focusing on facade coordination using single responsibility components."""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 validator_registry: Optional[ValidatorRegistry] = None,
                 validation_coordinator: Optional[ValidationCoordinator] = None):
        """Initialize slimmed manager with dependency injection support.
        
        Args:
            config_path: Optional path to configuration file
            validator_registry: Optional validator registry (for testing)
            validation_coordinator: Optional validation coordinator (for testing)
        """
        self.config = PreToolAnalysisConfig(config_path)
        self.context_tracker = WorkflowContextTracker()
        self.validation_count = 0
        
        # Import validator classes for registry
        validator_classes = self._import_validator_classes()
        
        # Initialize components (with dependency injection support)
        self.validator_registry = validator_registry or ValidatorRegistry(validator_classes)
        self.validation_coordinator = validation_coordinator or ValidationCoordinator()
        
        # Initialize validators
        self._initialize_system()
    
    def _import_validator_classes(self) -> Dict[str, Type[HiveWorkflowValidator]]:
        """Import and return validator classes for the registry."""
        from ..analyzers import (
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
            GitHubSyncAnalyzer,
            RogueSystemValidator,
            ConflictingArchitectureValidator,
            OverwriteProtectionValidator
        )
        # Import refactored validators (Phase 1 & 2)
        from ..analyzers.refactored_concurrent_execution_validator import RefactoredConcurrentExecutionValidator
        from ..analyzers.refactored_agent_patterns_validator import RefactoredAgentPatternsValidator
        from ..analyzers.refactored_visual_formats_validator import RefactoredVisualFormatsValidator
        from ..analyzers.refactored_mcp_separation_validator import RefactoredMCPSeparationValidator
        from ..analyzers.refactored_duplication_detection_validator import RefactoredDuplicationDetectionValidator
        from ..analyzers.refactored_claude_flow_suggester import RefactoredClaudeFlowSuggesterValidator
        from ..analyzers.refactored_conflicting_architecture_validator import RefactoredConflictingArchitectureValidator
        from ..analyzers.refactored_overwrite_protection_validator import RefactoredOverwriteProtectionValidator
        
        # Import Phase 3 ZEN integration validator
        from ..analyzers.multi_model_consensus_validator import MultiModelConsensusValidator
        
        # Registry of available validators - using refactored versions where available
        return {
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
            "safety_validator": SafetyValidator,
            # Using refactored validators for improved performance and maintainability
            "concurrent_execution_validator": RefactoredConcurrentExecutionValidator,
            "agent_patterns_validator": RefactoredAgentPatternsValidator,
            "visual_formats_validator": RefactoredVisualFormatsValidator,
            "mcp_separation_validator": RefactoredMCPSeparationValidator,
            "duplication_detection_validator": RefactoredDuplicationDetectionValidator,
            "claude_flow_suggester": RefactoredClaudeFlowSuggesterValidator,
            "conflicting_architecture_validator": RefactoredConflictingArchitectureValidator,
            "overwrite_protection_validator": RefactoredOverwriteProtectionValidator,
            "rogue_system_validator": RogueSystemValidator,
            # Phase 3 ZEN integration: Multi-model consensus validation
            "multi_model_consensus_validator": MultiModelConsensusValidator
        }
    
    def _initialize_system(self) -> None:
        """Initialize the validator system."""
        # Get enabled validators from config
        enabled_validators = [
            name for name in self.config._config.get("enabled_validators", [])
            if self.config.is_validator_enabled(name)
        ]
        
        # Register validators
        self.validator_registry.register_validators(enabled_validators)
        
        # Register validators with parallel framework for coordination
        validators = self.validator_registry.get_validators()
        self.validation_coordinator.register_validators_with_parallel_framework(validators)
    
    def validate_tool_usage(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate tool usage and provide guidance if needed.
        
        Args:
            tool_name: Name of the tool being validated
            tool_input: Tool input parameters
            
        Returns:
            Guidance dict if validation requires intervention, None if tool should proceed
        """
        self.validation_count += 1
        
        # Get validators from registry
        validators = self.validator_registry.get_validators()
        if not validators:
            return None
        
        # Coordinate validation execution
        validation_results = self.validation_coordinator.coordinate_validation(
            tool_name, tool_input, validators, self.context_tracker
        )
        
        # Process results and generate guidance
        if validation_results:
            return self._process_validation_results(validation_results, tool_name, tool_input)
        
        return None
    
    def _process_validation_results(self, 
                                  results: List[ValidationResult], 
                                  tool_name: str, 
                                  tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process validation results and determine guidance response."""
        
        # Import override manager
        from ..analyzers.override_manager import OverrideManager
        override_manager = OverrideManager()
        
        # Sort by priority score (highest first) and severity
        results.sort(key=lambda r: (r.severity.value, -r.priority_score), reverse=True)
        
        # Get the highest priority result
        primary_result = results[0]
        
        # Check for override request
        override_info = override_manager.check_for_override_request(tool_name, tool_input)
        
        # Determine if we should block execution
        should_block = self._should_block_execution(primary_result)
        
        # If blocking and override requested, check if override is allowed
        if should_block and override_info:
            override_allowed, justification = override_info
            validator_name = self._get_validator_name_from_result(primary_result, results)
            
            if override_manager.should_allow_override(validator_name, justification, primary_result.severity):
                # Log the override
                override_manager.log_override(validator_name, tool_name, justification, primary_result.severity)
                
                # Allow execution with override notice
                should_block = False
                primary_result.message = f"🔓 OVERRIDE GRANTED: {primary_result.message}\n   📝 Justification: {justification}"
        
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
            "total_validators_triggered": len(results),
            "override_applied": override_info is not None and not should_block
        }
    
    def _get_validator_name_from_result(self, 
                                       primary_result: ValidationResult, 
                                       all_results: List[ValidationResult]) -> str:
        """Get the validator name that produced the primary result."""
        # This is a simplified approach - in production, results would carry validator info
        # For now, infer from the message patterns
        message = primary_result.message.lower()
        
        if "duplication" in message:
            return "duplication_detection_validator"
        elif "rogue system" in message:
            return "rogue_system_validator"
        elif "architecture" in message or "conflict" in message:
            return "conflicting_architecture_validator"
        elif "overwrite" in message:
            return "overwrite_protection_validator"
        elif "dangerous" in message:
            return "safety_validator"
        
        return "unknown_validator"
    
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
    
    def _generate_guidance_message(self, 
                                  primary_result: ValidationResult, 
                                  all_results: List[ValidationResult], 
                                  tool_name: str) -> str:
        """Generate comprehensive guidance message."""
        
        # Start with primary message  
        message_parts = ["👑 QUEEN ZEN'S HIVE INTELLIGENCE:"]
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
        """Get comprehensive status of the validation system."""
        status = {
            "total_validations": self.validation_count,
            "active_validators": self.validator_registry.get_validator_names(),
            "validator_count": self.validator_registry.get_validator_count(),
            "coordination_state": self.context_tracker.get_coordination_state(),
            "tools_since_zen": self.context_tracker.get_tools_since_zen(),
            "tools_since_flow": self.context_tracker.get_tools_since_flow(),
            "config_path": self.config.config_path
        }
        
        # Add coordination statistics
        coordination_stats = self.validation_coordinator.get_coordination_stats()
        status["coordination_performance"] = coordination_stats
        
        return status
    
    def get_debug_report(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Generate detailed debug report for troubleshooting."""
        validators = self.validator_registry.get_validators()
        
        report_parts = [
            "=== SLIMMED PRE-TOOL VALIDATION DEBUG REPORT ===",
            f"Tool: {tool_name}",
            f"Total Validations: {self.validation_count}",
            f"Active Validators: {len(validators)}",
            ""
        ]
        
        # Validator details from registry
        for validator in validators:
            report_parts.extend([
                f"Validator: {validator.get_validator_name()}",
                f"  Priority: {validator.priority}",
                ""
            ])
        
        # Context tracking details
        report_parts.extend([
            "Context Tracking:",
            f"  Coordination State: {self.context_tracker.get_coordination_state()}",
            f"  Tools Since ZEN: {self.context_tracker.get_tools_since_zen()}",
            f"  Tools Since Flow: {self.context_tracker.get_tools_since_flow()}", 
            f"  Recent Pattern: {self.context_tracker.get_recent_pattern()}",
            ""
        ])
        
        # Component status
        coordination_stats = self.validation_coordinator.get_coordination_stats()
        report_parts.extend([
            "Component Status:",
            f"  ValidatorRegistry: {self.validator_registry.get_validator_count()} validators registered",
            f"  ValidationCoordinator: {coordination_stats.get('coordination_mode', 'unknown')} mode",
            ""
        ])
        
        return "\n".join(report_parts)


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