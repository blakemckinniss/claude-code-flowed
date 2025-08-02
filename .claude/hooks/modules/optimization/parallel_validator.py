"""Parallel validation framework for concurrent hook execution.

This module provides a framework for running multiple validators concurrently,
significantly improving hook execution performance by eliminating sequential validation bottlenecks.
"""

import asyncio
import concurrent.futures
import time
import threading
import sys
from typing import Dict, Any, List, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Path setup handled by centralized resolver when importing this module


class ValidationResult(NamedTuple):
    """Result of a validation operation."""
    validator_name: str
    success: bool
    data: Any
    execution_time: float
    error_message: Optional[str] = None
    should_block: bool = False


class ValidationPriority(Enum):
    """Priority levels for validators."""
    CRITICAL = 1000   # Must complete before any tool execution
    HIGH = 800        # Important but can run in parallel
    MEDIUM = 500      # Standard validation
    LOW = 200         # Optional enhancements


@dataclass
class ValidatorConfig:
    """Configuration for a validator."""
    name: str
    priority: ValidationPriority
    timeout: float = 5.0
    retry_count: int = 0
    can_run_parallel: bool = True
    depends_on: List[str] = field(default_factory=list)


class ParallelValidationFramework:
    """Framework for running validators in parallel."""
    
    def __init__(self, max_workers: int = 8, default_timeout: float = 10.0):
        """Initialize parallel validation framework.
        
        Args:
            max_workers: Maximum number of concurrent validators
            default_timeout: Default timeout for validator execution
        """
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.validators: Dict[str, Callable] = {}
        self.validator_configs: Dict[str, ValidatorConfig] = {}
        self.results_cache: Dict[str, ValidationResult] = {}
        self._lock = threading.RLock()
        
        # Performance metrics
        self.execution_stats = {
            "total_runs": 0,
            "parallel_runs": 0,
            "sequential_runs": 0,
            "total_time_saved": 0.0,
            "average_parallel_speedup": 0.0
        }
    
    def register_validator(self, 
                          name: str,
                          validator_func: Callable,
                          priority: ValidationPriority = ValidationPriority.MEDIUM,
                          timeout: float = 5.0,
                          can_run_parallel: bool = True,
                          depends_on: Optional[List[str]] = None) -> None:
        """Register a validator with the framework.
        
        Args:
            name: Unique validator name
            validator_func: Function to execute validation
            priority: Validation priority level
            timeout: Maximum execution time
            can_run_parallel: Whether validator can run concurrently
            depends_on: List of validator names this depends on
        """
        with self._lock:
            config = ValidatorConfig(
                name=name,
                priority=priority,
                timeout=timeout,
                can_run_parallel=can_run_parallel,
                depends_on=depends_on or []
            )
            
            self.validators[name] = validator_func
            self.validator_configs[name] = config
    
    async def validate_parallel(self, 
                               tool_name: str,
                               tool_data: Dict[str, Any],
                               selected_validators: Optional[List[str]] = None) -> Dict[str, ValidationResult]:
        """Run validators in parallel with dependency resolution.
        
        Args:
            tool_name: Name of the tool being validated
            tool_data: Tool data for validation
            selected_validators: Specific validators to run (None for all)
            
        Returns:
            Dictionary of validation results by validator name
        """
        start_time = time.time()
        
        # Determine which validators to run
        validators_to_run = selected_validators or list(self.validators.keys())
        
        # Group validators by priority and dependencies
        validation_groups = self._create_validation_groups(validators_to_run)
        
        all_results = {}
        
        # Execute validation groups in order
        for group in validation_groups:
            group_results = await self._execute_validation_group(
                group, tool_name, tool_data
            )
            all_results.update(group_results)
            
            # Check for critical failures that should stop execution
            critical_failures = [
                result for result in group_results.values()
                if not result.success and result.should_block
            ]
            
            if critical_failures:
                # Stop execution on critical failures
                break
        
        # Update performance statistics
        total_time = time.time() - start_time
        self._update_performance_stats(total_time, len(validators_to_run))
        
        return all_results
    
    def validate_sync(self, 
                     tool_name: str,
                     tool_data: Dict[str, Any],
                     selected_validators: Optional[List[str]] = None) -> Dict[str, ValidationResult]:
        """Synchronous wrapper for parallel validation.
        
        Args:
            tool_name: Name of the tool being validated
            tool_data: Tool data for validation
            selected_validators: Specific validators to run
            
        Returns:
            Dictionary of validation results
        """
        # Run async validation in a new event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.validate_parallel(tool_name, tool_data, selected_validators)
            )
        finally:
            loop.close()
    
    def _create_validation_groups(self, validator_names: List[str]) -> List[List[str]]:
        """Create groups of validators that can run in parallel.
        
        Args:
            validator_names: Names of validators to group
            
        Returns:
            List of validator groups ordered by priority and dependencies
        """
        # Sort by priority first
        sorted_validators = sorted(
            validator_names,
            key=lambda name: self.validator_configs[name].priority.value,
            reverse=True
        )
        
        groups = []
        remaining = set(sorted_validators)
        completed = set()
        
        while remaining:
            # Find validators that can run in this group
            current_group = []
            
            for validator in list(remaining):
                config = self.validator_configs[validator]
                
                # Check if dependencies are satisfied
                deps_satisfied = all(dep in completed for dep in config.depends_on)
                
                if deps_satisfied:
                    current_group.append(validator)
                    remaining.remove(validator)
            
            if not current_group:
                # Circular dependency or missing validator
                print(f"Warning: Circular dependency detected in validators: {remaining}", 
                      file=sys.stderr)
                current_group = list(remaining)
                remaining.clear()
            
            groups.append(current_group)
            completed.update(current_group)
        
        return groups
    
    async def _execute_validation_group(self, 
                                       validator_names: List[str],
                                       tool_name: str,
                                       tool_data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Execute a group of validators in parallel.
        
        Args:
            validator_names: Names of validators in this group
            tool_name: Tool being validated
            tool_data: Tool data
            
        Returns:
            Dictionary of validation results
        """
        # Separate parallel and sequential validators
        parallel_validators = []
        sequential_validators = []
        
        for name in validator_names:
            config = self.validator_configs[name]
            if config.can_run_parallel:
                parallel_validators.append(name)
            else:
                sequential_validators.append(name)
        
        results = {}
        
        # Run parallel validators concurrently
        if parallel_validators:
            parallel_results = await self._run_parallel_validators(
                parallel_validators, tool_name, tool_data
            )
            results.update(parallel_results)
        
        # Run sequential validators one by one
        for validator_name in sequential_validators:
            result = await self._run_single_validator(
                validator_name, tool_name, tool_data
            )
            results[validator_name] = result
        
        return results
    
    async def _run_parallel_validators(self, 
                                     validator_names: List[str],
                                     tool_name: str,
                                     tool_data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Run validators in parallel using asyncio.
        
        Args:
            validator_names: Names of validators to run
            tool_name: Tool being validated
            tool_data: Tool data
            
        Returns:
            Dictionary of validation results
        """
        # Create tasks for each validator
        tasks = []
        for validator_name in validator_names:
            task = asyncio.create_task(
                self._run_single_validator(validator_name, tool_name, tool_data)
            )
            tasks.append((validator_name, task))
        
        # Wait for all tasks to complete
        results = {}
        for validator_name, task in tasks:
            try:
                result = await task
                results[validator_name] = result
            except Exception as e:
                results[validator_name] = ValidationResult(
                    validator_name=validator_name,
                    success=False,
                    data=None,
                    execution_time=0.0,
                    error_message=str(e),
                    should_block=False
                )
        
        return results
    
    async def _run_single_validator(self, 
                                   validator_name: str,
                                   tool_name: str,
                                   tool_data: Dict[str, Any]) -> ValidationResult:
        """Run a single validator with timeout and error handling.
        
        Args:
            validator_name: Name of validator to run
            tool_name: Tool being validated
            tool_data: Tool data
            
        Returns:
            Validation result
        """
        start_time = time.time()
        config = self.validator_configs[validator_name]
        validator_func = self.validators[validator_name]
        
        try:
            # Run validator with timeout
            result = await asyncio.wait_for(
                self._execute_validator_safely(validator_func, tool_name, tool_data),
                timeout=config.timeout
            )
            
            execution_time = time.time() - start_time
            
            return ValidationResult(
                validator_name=validator_name,
                success=True,
                data=result,
                execution_time=execution_time,
                should_block=getattr(result, 'should_block', False)
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return ValidationResult(
                validator_name=validator_name,
                success=False,
                data=None,
                execution_time=execution_time,
                error_message=f"Validator timeout after {config.timeout}s",
                should_block=False
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ValidationResult(
                validator_name=validator_name,
                success=False,
                data=None,
                execution_time=execution_time,
                error_message=str(e),
                should_block=False
            )
    
    async def _execute_validator_safely(self, 
                                       validator_func: Callable,
                                       tool_name: str,
                                       tool_data: Dict[str, Any]) -> Any:
        """Execute validator function safely in async context.
        
        Args:
            validator_func: Validator function to execute
            tool_name: Tool being validated
            tool_data: Tool data
            
        Returns:
            Validator result
        """
        # Check if validator is async
        if asyncio.iscoroutinefunction(validator_func):
            return await validator_func(tool_name, tool_data)
        else:
            # Run sync validator in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return await loop.run_in_executor(
                    executor, validator_func, tool_name, tool_data
                )
    
    def _update_performance_stats(self, total_time: float, validator_count: int):
        """Update performance statistics.
        
        Args:
            total_time: Total execution time
            validator_count: Number of validators run
        """
        with self._lock:
            self.execution_stats["total_runs"] += 1
            
            if validator_count > 1:
                self.execution_stats["parallel_runs"] += 1
                # Estimate time saved by parallel execution
                estimated_sequential_time = validator_count * 0.5  # Average 0.5s per validator
                time_saved = max(0, estimated_sequential_time - total_time)
                self.execution_stats["total_time_saved"] += time_saved
            else:
                self.execution_stats["sequential_runs"] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        with self._lock:
            stats = self.execution_stats.copy()
            
            if stats["parallel_runs"] > 0:
                stats["average_parallel_speedup"] = (
                    stats["total_time_saved"] / stats["parallel_runs"]
                )
            
            return stats
    
    def clear_cache(self):
        """Clear the results cache."""
        with self._lock:
            self.results_cache.clear()


# Global instance for use across the hook system
_global_parallel_validator: Optional[ParallelValidationFramework] = None


def get_parallel_validator() -> ParallelValidationFramework:
    """Get or create the global parallel validator instance."""
    global _global_parallel_validator
    if _global_parallel_validator is None:
        _global_parallel_validator = ParallelValidationFramework()
    return _global_parallel_validator


def register_hook_validators():
    """Register all existing hook validators with the parallel framework."""
    validator = get_parallel_validator()
    
    # Register existing validators with appropriate priorities
    try:
        # Critical validators that must complete first
        validator.register_validator(
            "rogue_system_validator",
            None,  # Will be set when importing actual validators
            ValidationPriority.CRITICAL,
            timeout=2.0,
            can_run_parallel=False  # Safety-critical, run alone
        )
        
        # High-priority validators that can run in parallel
        validator.register_validator(
            "concurrent_execution_validator",
            None,
            ValidationPriority.HIGH,
            timeout=3.0,
            can_run_parallel=True
        )
        
        validator.register_validator(
            "mcp_separation_validator",
            None,
            ValidationPriority.HIGH,
            timeout=3.0,
            can_run_parallel=True
        )
        
        # Medium-priority validators
        validator.register_validator(
            "agent_patterns_validator",
            None,
            ValidationPriority.MEDIUM,
            timeout=5.0,
            can_run_parallel=True
        )
        
        validator.register_validator(
            "visual_formats_validator",
            None,
            ValidationPriority.MEDIUM,
            timeout=5.0,
            can_run_parallel=True
        )
        
        # Low-priority validators
        validator.register_validator(
            "duplication_detection_validator",
            None,
            ValidationPriority.LOW,
            timeout=10.0,
            can_run_parallel=True
        )
        
        validator.register_validator(
            "overwrite_protection_validator",
            None,
            ValidationPriority.LOW,
            timeout=10.0,
            can_run_parallel=True
        )
        
    except Exception as e:
        print(f"Warning: Failed to register some validators: {e}", file=sys.stderr)