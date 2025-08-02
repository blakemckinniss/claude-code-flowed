"""Parallel execution framework for validators and analyzers.

This module enables concurrent execution of independent validation operations
to significantly reduce hook execution time.
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class ValidationResult:
    """Result from a validation operation."""
    validator_name: str
    success: bool
    severity: str = "info"
    message: str = ""
    metadata: Optional[Dict[str, Any]] = None
    duration: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ParallelValidationManager:
    """Manages parallel execution of validators."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize parallel validation manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = None
        self._stats = {
            "total_validations": 0,
            "parallel_batches": 0,
            "average_speedup": 0.0
        }
    
    async def validate_parallel(self, 
                              tool_name: str, 
                              tool_input: Dict[str, Any],
                              validators: List[Any]) -> List[ValidationResult]:
        """Run all validators in parallel.
        
        Args:
            tool_name: Name of the tool being validated
            tool_input: Input parameters for the tool
            validators: List of validator instances
            
        Returns:
            List of validation results
        """
        if not validators:
            return []
        
        start_time = time.time()
        
        # Create async tasks for each validator
        tasks = []
        for validator in validators:
            task = asyncio.create_task(
                self._run_validator_async(validator, tool_name, tool_input)
            )
            tasks.append(task)
        
        # Wait for all validators to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to ValidationResult
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                valid_results.append(ValidationResult(
                    validator_name=f"validator_{i}",
                    success=False,
                    severity="error",
                    message=f"Validation error: {result!s}"
                ))
            elif isinstance(result, ValidationResult):
                valid_results.append(result)
            elif isinstance(result, dict):
                # Convert dict to ValidationResult
                valid_results.append(ValidationResult(**result))
        
        # Update statistics
        duration = time.time() - start_time
        self._update_stats(len(validators), duration)
        
        return valid_results
    
    async def _run_validator_async(self, validator: Any, tool_name: str, tool_input: Dict[str, Any]) -> ValidationResult:
        """Run a single validator asynchronously."""
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        try:
            # Run validator in thread pool
            result = await loop.run_in_executor(
                self.executor,
                self._run_validator_sync,
                validator, tool_name, tool_input
            )
            
            duration = time.time() - start_time
            
            # Ensure result is a ValidationResult
            if isinstance(result, dict):
                result = ValidationResult(**result, duration=duration)
            elif not isinstance(result, ValidationResult):
                result = ValidationResult(
                    validator_name=getattr(validator, "__class__.__name__", "unknown"),
                    success=True,
                    message=str(result) if result else "Validation passed",
                    duration=duration
                )
            else:
                result.duration = duration
            
            return result
            
        except Exception as e:
            return ValidationResult(
                validator_name=getattr(validator, "__class__.__name__", "unknown"),
                success=False,
                severity="error",
                message=f"Validation error: {e!s}",
                duration=time.time() - start_time
            )
    
    def _run_validator_sync(self, validator: Any, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Run a validator synchronously."""
        # Handle different validator interfaces
        if hasattr(validator, "validate_workflow"):
            return validator.validate_workflow(tool_name, tool_input, None)
        elif hasattr(validator, "validate"):
            return validator.validate(tool_name, tool_input)
        elif hasattr(validator, "analyze"):
            return validator.analyze(tool_name, tool_input)
        elif callable(validator):
            return validator(tool_name, tool_input)
        else:
            raise ValueError(f"Validator {validator} does not have a valid interface")
    
    def _update_stats(self, validator_count: int, duration: float):
        """Update performance statistics."""
        self._stats["total_validations"] += validator_count
        self._stats["parallel_batches"] += 1
        
        # Estimate speedup (assumes linear time for sequential execution)
        estimated_sequential_time = duration * validator_count / self.max_workers
        speedup = estimated_sequential_time / duration if duration > 0 else 1.0
        
        # Update rolling average
        prev_avg = self._stats["average_speedup"]
        batch_count = self._stats["parallel_batches"]
        self._stats["average_speedup"] = (
            (prev_avg * (batch_count - 1) + speedup) / batch_count
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "max_workers": self.max_workers,
            "total_validations": self._stats["total_validations"],
            "parallel_batches": self._stats["parallel_batches"],
            "average_speedup": round(self._stats["average_speedup"], 2)
        }
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class ValidatorPipeline:
    """Pipeline for organizing validators into stages."""
    
    def __init__(self, parallel_manager: ParallelValidationManager):
        """Initialize validator pipeline.
        
        Args:
            parallel_manager: Manager for parallel execution
        """
        self.parallel_manager = parallel_manager
        self.stages = []
    
    def add_stage(self, name: str, validators: List[Any], critical: bool = False):
        """Add a validation stage.
        
        Args:
            name: Stage name
            validators: List of validators for this stage
            critical: If True, stop pipeline on any failure in this stage
        """
        self.stages.append({
            "name": name,
            "validators": validators,
            "critical": critical
        })
    
    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the validation pipeline.
        
        Args:
            tool_name: Tool being validated
            tool_input: Tool input parameters
            
        Returns:
            Pipeline execution results
        """
        results = {
            "stages": {},
            "overall_success": True,
            "critical_failure": False,
            "total_duration": 0.0
        }
        
        start_time = time.time()
        
        for stage in self.stages:
            stage_name = stage["name"]
            
            # Run validators in parallel for this stage
            stage_results = await self.parallel_manager.validate_parallel(
                tool_name, tool_input, stage["validators"]
            )
            
            results["stages"][stage_name] = stage_results
            
            # Check for critical failures
            if stage["critical"]:
                critical_failures = [r for r in stage_results if not r.success and r.severity in ["critical", "error"]]
                if critical_failures:
                    results["critical_failure"] = True
                    results["overall_success"] = False
                    break
            
            # Update overall success
            if any(not r.success for r in stage_results):
                results["overall_success"] = False
        
        results["total_duration"] = time.time() - start_time
        return results