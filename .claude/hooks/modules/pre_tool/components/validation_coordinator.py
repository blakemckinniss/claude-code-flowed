#!/usr/bin/env python3
"""Validation Coordinator - Single Responsibility Component.

Orchestrates validation execution across multiple validators using parallel framework.
Separates execution coordination from validator management and result processing.
"""

import sys
import time
from typing import Dict, Any, List, Optional
from ..core.workflow_validator import HiveWorkflowValidator, ValidationResult, WorkflowContextTracker
from ...optimization.parallel_validator import get_parallel_validator, ValidationPriority
from ...optimization.validation_cache import get_validation_cache
from ...optimization.object_pool import get_object_pools


class ValidationCoordinator:
    """Coordinates validation execution with single responsibility."""
    
    def __init__(self):
        """Initialize validation coordinator."""
        self.parallel_validator = get_parallel_validator()
        self.validation_cache = get_validation_cache()
        self.object_pools = get_object_pools()
        self.use_parallel_validation = True
        
    def coordinate_validation(self, 
                            tool_name: str, 
                            tool_input: Dict[str, Any], 
                            validators: List[HiveWorkflowValidator],
                            context_tracker: WorkflowContextTracker) -> List[ValidationResult]:
        """Coordinate validation execution across multiple validators.
        
        Args:
            tool_name: Name of the tool being validated
            tool_input: Tool input parameters
            validators: List of validators to execute
            context_tracker: Context tracking instance
            
        Returns:
            List of validation results
        """
        # Update context tracking
        context_tracker.add_tool_context(tool_name)
        
        # Use parallel validation if enabled and multiple validators
        if self.use_parallel_validation and len(validators) > 1:
            return self._coordinate_parallel_validation(
                tool_name, tool_input, validators, context_tracker
            )
        else:
            return self._coordinate_sequential_validation(
                tool_name, tool_input, validators, context_tracker
            )
    
    def _coordinate_parallel_validation(self, 
                                      tool_name: str, 
                                      tool_input: Dict[str, Any], 
                                      validators: List[HiveWorkflowValidator],
                                      context_tracker: WorkflowContextTracker) -> List[ValidationResult]:
        """Coordinate parallel validation execution with caching and object pooling."""
        try:
            validator_names = [v.get_validator_name() for v in validators]
            
            # Borrow context data object from pool
            context_data = self.object_pools.borrow_object("context_data")
            if context_data is None:
                context_data = self._create_default_context_data()
            
            # Check cache for each validator first
            cached_results = {}
            validators_to_run = []
            
            for validator_name in validator_names:
                cached_result = self.validation_cache.get_cached_validation_result(
                    tool_name, tool_input, validator_name, context_tracker,
                    accept_stale=True  # Accept stale results for performance
                )
                
                if cached_result:
                    cached_results[validator_name] = cached_result
                else:
                    validators_to_run.append(validator_name)
            
            # Run uncached validators in parallel
            parallel_results = {}
            if validators_to_run:
                parallel_results = self.parallel_validator.validate_sync(
                    tool_name, tool_input, validators_to_run
                )
                
                # Cache the new results
                for validator_name, result in parallel_results.items():
                    if result.success and result.data:
                        self.validation_cache.cache_validation_result(
                            tool_name, tool_input, validator_name, 
                            result.data, context_tracker
                        )
            
            # Combine cached and fresh results using pooled validation objects
            all_results = []
            
            # Process cached results
            for cached_result in cached_results.values():
                if isinstance(cached_result, dict):
                    validation_result = self._reconstruct_validation_result_from_pool(cached_result)
                    if validation_result:
                        all_results.append(validation_result)
            
            # Process fresh results
            for result in parallel_results.values():
                if result.success and result.data and isinstance(result.data, dict):
                    validation_result = self._reconstruct_validation_result_from_pool(result.data)
                    if validation_result:
                        all_results.append(validation_result)
            
            # Return context data to pool
            if context_data is not None:
                self.object_pools.return_object("context_data", context_data)
            
            return all_results
            
        except Exception as e:
            print(f"Warning: Parallel validation coordination failed, falling back to sequential: {e}", file=sys.stderr)
            return self._coordinate_sequential_validation(
                tool_name, tool_input, validators, context_tracker
            )
    
    def _coordinate_sequential_validation(self, 
                                        tool_name: str, 
                                        tool_input: Dict[str, Any], 
                                        validators: List[HiveWorkflowValidator],
                                        context_tracker: WorkflowContextTracker) -> List[ValidationResult]:
        """Coordinate sequential validation execution with caching and object pooling."""
        all_results = []
        
        # Borrow analysis data object from pool for consistency
        analysis_data = self.object_pools.borrow_object("analysis_data")
        if analysis_data is None:
            analysis_data = self._create_default_analysis_data()
        
        for validator in validators:
            validator_name = validator.get_validator_name()
            
            try:
                # Check cache first
                cached_result = self.validation_cache.get_cached_validation_result(
                    tool_name, tool_input, validator_name, context_tracker,
                    accept_stale=True
                )
                
                if cached_result:
                    # Use cached result with object pooling
                    if isinstance(cached_result, dict):
                        validation_result = self._reconstruct_validation_result_from_pool(cached_result)
                        if validation_result:
                            all_results.append(validation_result)
                else:
                    # Run validator and cache result
                    result = validator.validate_workflow(tool_name, tool_input, context_tracker)
                    if result:
                        all_results.append(result)
                        
                        # Cache the result
                        self.validation_cache.cache_validation_result(
                            tool_name, tool_input, validator_name,
                            result.__dict__, context_tracker
                        )
                        
            except Exception as e:
                print(f"Warning: Validator {validator_name} failed during coordination: {e}", file=sys.stderr)
        
        # Return analysis data to pool
        if analysis_data is not None:
            self.object_pools.return_object("analysis_data", analysis_data)
        
        return all_results
    
    def register_validators_with_parallel_framework(self, validators: List[HiveWorkflowValidator]) -> None:
        """Register validators with the parallel validation framework.
        
        Args:
            validators: List of validators to register
        """
        try:
            for validator in validators:
                validator_name = validator.get_validator_name()
                priority = self._map_priority_to_parallel(validator.priority)
                
                # Create wrapper function for the validator
                def validator_wrapper(tool_name: str, tool_data: Dict[str, Any], 
                                    validator_instance=validator) -> Dict[str, Any]:
                    # Handle different validator interfaces
                    if hasattr(validator_instance, 'validate_workflow'):
                        result = validator_instance.validate_workflow(
                            tool_name, tool_data, WorkflowContextTracker()
                        )
                        return result.__dict__ if result else {"success": True}
                    elif hasattr(validator_instance, 'validate'):
                        # Handle validators with simpler validate method
                        try:
                            success, message, alternatives = validator_instance.validate(tool_name, tool_data)
                            return {
                                "success": success,
                                "message": message or "",
                                "alternatives": alternatives,
                                "severity": "BLOCK" if not success else "ALLOW"
                            }
                        except Exception as e:
                            return {"success": True, "error": str(e)}
                    else:
                        return {"success": True}
                
                # Register with parallel framework
                self.parallel_validator.register_validator(
                    name=validator_name,
                    validator_func=validator_wrapper,
                    priority=priority,
                    timeout=self._get_validator_timeout(validator_name),
                    can_run_parallel=self._can_run_parallel(validator_name)
                )
                
        except Exception as e:
            print(f"Warning: Failed to register validators with parallel framework: {e}", file=sys.stderr)
            self.use_parallel_validation = False
    
    def _create_default_context_data(self) -> Dict[str, Any]:
        """Create default context data structure."""
        return {
            "tools": [],
            "zen_calls": 0,
            "flow_calls": 0, 
            "patterns": [],
            "state": "disconnected"
        }
    
    def _create_default_analysis_data(self) -> Dict[str, Any]:
        """Create default analysis data structure."""
        return {
            "complexity_score": 0,
            "risk_level": "low",
            "recommendations": [],
            "metadata": {}
        }
    
    def _reconstruct_validation_result_from_pool(self, data: Dict[str, Any]) -> Optional[ValidationResult]:
        """Reconstruct ValidationResult from dict data using object pool."""
        try:
            if not data.get("message"):
                return None
            
            # Try to borrow validation result from pool
            validation_result_data = self.object_pools.borrow_object("validation_result")
            if validation_result_data is None:
                # Fallback to regular creation if pool is unavailable
                return self._reconstruct_validation_result(data)
            
            # Populate pooled object with data
            validation_result_data.update({
                "message": data.get("message", ""),
                "severity": data.get("severity", "ALLOW"),
                "suggested_alternative": data.get("suggested_alternative"),
                "hive_guidance": data.get("hive_guidance"),
                "priority_score": data.get("priority_score", 0),
                "violation_type": data.get("violation_type"), 
                "blocking_reason": data.get("blocking_reason")
            })
            
            # Create ValidationResult with pooled data
            from ..core.workflow_validator import ValidationSeverity
            return ValidationResult(
                message=validation_result_data["message"],
                severity=ValidationSeverity(validation_result_data["severity"]),
                suggested_alternative=validation_result_data["suggested_alternative"],
                hive_guidance=validation_result_data["hive_guidance"],
                priority_score=validation_result_data["priority_score"],
                violation_type=validation_result_data["violation_type"],
                blocking_reason=validation_result_data["blocking_reason"]
            )
        except Exception:
            return None
    
    def _reconstruct_validation_result(self, data: Dict[str, Any]) -> Optional[ValidationResult]:
        """Reconstruct ValidationResult from dict data."""
        try:
            if not data.get("message"):
                return None
                
            from ..core.workflow_validator import ValidationSeverity
            return ValidationResult(
                message=data.get("message", ""),
                severity=ValidationSeverity(data.get("severity", "ALLOW")),
                suggested_alternative=data.get("suggested_alternative"),
                hive_guidance=data.get("hive_guidance"),
                priority_score=data.get("priority_score", 0),
                violation_type=data.get("violation_type"),
                blocking_reason=data.get("blocking_reason")
            )
        except Exception:
            return None
    
    def _map_priority_to_parallel(self, priority: int) -> ValidationPriority:
        """Map validator priority to parallel validation priority."""
        if priority >= 900:
            return ValidationPriority.CRITICAL
        elif priority >= 750:
            return ValidationPriority.HIGH
        elif priority >= 500:
            return ValidationPriority.MEDIUM
        else:
            return ValidationPriority.LOW
    
    def _get_validator_timeout(self, validator_name: str) -> float:
        """Get timeout for a validator."""
        # Safety and critical validators get more time
        if validator_name in ["safety_validator", "rogue_system_validator", "overwrite_protection_validator"]:
            return 10.0
        elif validator_name in ["duplication_detection_validator", "conflicting_architecture_validator"]:
            return 8.0
        else:
            return 5.0
    
    def _can_run_parallel(self, validator_name: str) -> bool:
        """Determine if validator can run in parallel."""
        # Safety-critical validators should run alone for reliability
        sequential_only = [
            "safety_validator",
            "rogue_system_validator", 
            "overwrite_protection_validator"
        ]
        return validator_name not in sequential_only
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination performance statistics.
        
        Returns:
            Dictionary of coordination performance metrics
        """
        stats = {
            "parallel_validation_enabled": self.use_parallel_validation,
            "coordination_mode": "parallel" if self.use_parallel_validation else "sequential"
        }
        
        # Add parallel validation performance stats if available
        if self.use_parallel_validation:
            try:
                perf_stats = self.parallel_validator.get_performance_stats()
                stats["parallel_performance"] = perf_stats
            except Exception as e:
                stats["parallel_performance_error"] = str(e)
        
        # Add validation cache statistics
        try:
            cache_stats = self.validation_cache.get_stats()
            stats["cache_performance"] = cache_stats
        except Exception as e:
            stats["cache_performance_error"] = str(e)
        
        # Add object pool statistics
        try:
            pool_stats = self.object_pools.get_all_stats()
            stats["object_pool_performance"] = {
                name: {
                    "hit_rate": pool_stats_data.get_hit_rate(),
                    "efficiency": pool_stats_data.get_efficiency_score(),
                    "current_size": pool_stats_data.current_size,
                    "created_objects": pool_stats_data.created_objects,
                    "borrowed_objects": pool_stats_data.borrowed_objects,
                    "returned_objects": pool_stats_data.returned_objects
                }
                for name, pool_stats_data in pool_stats.items()
            }
        except Exception as e:
            stats["object_pool_performance_error"] = str(e)
        
        return stats