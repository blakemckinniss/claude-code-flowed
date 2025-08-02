#!/usr/bin/env python3
"""Validator Registry - Single Responsibility Component.

Handles validator discovery, initialization, and registration.
Separates validator management concerns from execution orchestration.
"""

import sys
import inspect
from typing import List, Dict, Any, Optional, Type
from ..core.workflow_validator import HiveWorkflowValidator


class ValidatorRegistry:
    """Manages validator registration and initialization with single responsibility."""
    
    def __init__(self, validator_classes: Dict[str, Type[HiveWorkflowValidator]]):
        """Initialize registry with available validator classes.
        
        Args:
            validator_classes: Map of validator names to validator classes
        """
        self.validator_classes = validator_classes
        self.validators: List[HiveWorkflowValidator] = []
        self._validator_map: Dict[str, HiveWorkflowValidator] = {}
    
    def register_validators(self, enabled_validators: List[str]) -> None:
        """Register and initialize enabled validators.
        
        Args:
            enabled_validators: List of validator names to enable
        """
        self.validators.clear()
        self._validator_map.clear()
        
        for validator_name in enabled_validators:
            validator_class = self.validator_classes.get(validator_name)
            if not validator_class:
                print(f"Warning: Unknown validator {validator_name}", file=sys.stderr)
                continue
            
            validator = self._initialize_validator(validator_name, validator_class)
            if validator:
                self.validators.append(validator)
                self._validator_map[validator_name] = validator
        
        # Sort by priority (highest first)
        self.validators.sort(key=lambda v: getattr(v, 'priority', 500), reverse=True)
    
    def _initialize_validator(self, validator_name: str, 
                            validator_class: Type[HiveWorkflowValidator]) -> Optional[HiveWorkflowValidator]:
        """Initialize a single validator with proper priority assignment.
        
        Args:
            validator_name: Name of the validator
            validator_class: Validator class to initialize
            
        Returns:
            Initialized validator instance or None if initialization failed
        """
        priority = self._get_validator_priority(validator_name)
        
        try:
            # Check if validator accepts priority parameter
            init_sig = inspect.signature(validator_class.__init__)
            if 'priority' in init_sig.parameters:
                validator = validator_class(priority=priority)
            else:
                validator = validator_class()
                # Set priority manually if validator has priority attribute
                if hasattr(validator, 'priority'):
                    validator.priority = priority
                else:
                    # Add priority attribute
                    validator.priority = priority
        except Exception:
            # Fallback: try without priority first, then add it
            try:
                validator = validator_class()
                validator.priority = priority
            except Exception:
                # Skip this validator if it can't be initialized
                print(f"Warning: Could not initialize validator {validator_name}", file=sys.stderr)
                return None
        
        # Ensure validator has get_validator_name method
        if not hasattr(validator, 'get_validator_name'):
            def get_validator_name(self=validator, name=validator_name):
                return getattr(self, 'name', name)
            validator.get_validator_name = get_validator_name
        
        return validator
    
    def _get_validator_priority(self, validator_name: str) -> int:
        """Get priority for a validator based on its name and role.
        
        Args:
            validator_name: Name of the validator
            
        Returns:
            Priority score (higher = more important)
        """
        priorities = {
            "zen_hierarchy_validator": 1000,       # Highest - Queen ZEN is supreme
            "safety_validator": 950,               # Very High - Safety first
            "overwrite_protection_validator": 940, # Very High - Prevent data loss
            "mcp_separation_validator": 925,       # Very High - Critical separation
            "mcp_coordination_validator": 900,     # High - MCP optimization critical
            "rogue_system_validator": 890,         # High - Prevent architectural chaos
            "conflicting_architecture_validator": 880, # High - Maintain consistency
            "concurrent_execution_validator": 875, # High - Concurrency enforcement
            "duplication_detection_validator": 860, # High - Prevent code duplication
            "neural_pattern_validator": 850,       # High - Neural learning intelligence
            "github_coordinator_analyzer": 825,    # High - GitHub workflow intelligence
            "github_pr_analyzer": 820,             # High - PR workflow intelligence
            "github_issue_analyzer": 815,          # High - Issue workflow intelligence  
            "github_release_analyzer": 810,        # High - Release workflow intelligence
            "github_repo_analyzer": 805,           # High - Repository health intelligence
            "github_sync_analyzer": 800,           # High - Multi-repo sync intelligence
            "agent_patterns_validator": 775,       # High - Agent recommendations
            "hive_workflow_optimizer": 750,        # High - Workflow intelligence
            "visual_formats_validator": 650,       # Medium - Visual formatting
            "efficiency_optimizer": 600,           # Medium - General efficiency
            "claude_flow_suggester": 500           # Medium - Workflow suggestions
        }
        return priorities.get(validator_name, 500)
    
    def get_validators(self) -> List[HiveWorkflowValidator]:
        """Get all registered validators sorted by priority.
        
        Returns:
            List of validator instances
        """
        return self.validators.copy()
    
    def get_validator_by_name(self, validator_name: str) -> Optional[HiveWorkflowValidator]:
        """Get a specific validator by name.
        
        Args:
            validator_name: Name of the validator to retrieve
            
        Returns:
            Validator instance or None if not found
        """
        return self._validator_map.get(validator_name)
    
    def get_validator_names(self) -> List[str]:
        """Get names of all registered validators.
        
        Returns:
            List of validator names
        """
        return [v.get_validator_name() for v in self.validators]
    
    def get_validator_count(self) -> int:
        """Get total number of registered validators.
        
        Returns:
            Number of registered validators
        """
        return len(self.validators)