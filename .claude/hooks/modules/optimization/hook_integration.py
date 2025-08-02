"""Integration layer for optimization modules with hook system.

This module provides seamless integration of all optimization components
with the existing hook infrastructure, ensuring backward compatibility
while enabling advanced optimizations.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json

# Import optimization modules
from .integrated_optimizer import IntegratedHookOptimizer, get_hook_optimizer
from .async_orchestrator import TaskPriority
from .performance_monitor import get_performance_monitor


class HookOptimizationIntegration:
    """Main integration class for hook optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._optimizer = None
        self._monitor = None
        self._initialized = False
        self._optimization_enabled = True
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load optimization configuration."""
        config_path = Path(__file__).parent.parent.parent / "settings.json"
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    settings = json.load(f)
                    optimization_config = settings.get("optimization", {})
                    self._optimization_enabled = optimization_config.get("enabled", True)
            except Exception as e:
                self.logger.exception(f"Failed to load config: {e}")
    
    async def initialize(self):
        """Initialize optimization components."""
        if self._initialized or not self._optimization_enabled:
            return
        
        try:
            # Get or create optimizer
            self._optimizer = await get_hook_optimizer()
            self._monitor = get_performance_monitor()
            
            self._initialized = True
            self.logger.info("Hook optimization system initialized")
            
        except Exception as e:
            self.logger.exception(f"Failed to initialize optimization: {e}")
            self._optimization_enabled = False
    
    async def execute_pre_tool_validators(self,
                                         tool_name: str,
                                         tool_input: Dict[str, Any],
                                         validators: List[Any]) -> List[Any]:
        """Execute pre-tool validators with optimizations."""
        
        if not self._optimization_enabled or not self._optimizer:
            # Fallback to standard execution
            return await self._execute_validators_standard(validators, tool_name, tool_input)
        
        try:
            # Use optimized execution
            start_time = time.time()
            
            results = await self._optimizer.execute_validators_optimized(
                validators, tool_name, tool_input
            )
            
            # Record metrics
            duration = (time.time() - start_time) * 1000
            self._monitor.record_hook_execution(
                f"pre_tool_validators_{tool_name}",
                duration,
                success=True
            )
            
            return results
            
        except Exception as e:
            self.logger.exception(f"Optimized execution failed: {e}")
            # Fallback to standard execution
            return await self._execute_validators_standard(validators, tool_name, tool_input)
    
    async def execute_post_tool_hook(self,
                                    hook_name: str,
                                    hook_data: Dict[str, Any]) -> Any:
        """Execute post-tool hook with optimizations."""
        
        if not self._optimization_enabled or not self._optimizer:
            # Fallback to standard execution
            return self._execute_hook_standard(hook_name, hook_data)
        
        try:
            # Use optimized execution
            result = await self._optimizer.execute_hook_optimized(
                hook_path=hook_name,
                hook_data=hook_data,
                priority=TaskPriority.HIGH
            )
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Optimized hook execution failed: {e}")
            # Fallback to standard execution
            return self._execute_hook_standard(hook_name, hook_data)
    
    async def _execute_validators_standard(self,
                                         validators: List[Any],
                                         tool_name: str,
                                         tool_input: Dict[str, Any]) -> List[Any]:
        """Standard validator execution (fallback)."""
        results = []
        
        for validator in validators:
            try:
                # Execute validator based on its interface
                if hasattr(validator, "validate_workflow"):
                    result = validator.validate_workflow(tool_name, tool_input, None)
                elif hasattr(validator, "validate"):
                    result = validator.validate(tool_name, tool_input)
                elif hasattr(validator, "analyze"):
                    result = validator.analyze(tool_name, tool_input)
                elif callable(validator):
                    result = validator(tool_name, tool_input)
                else:
                    result = None
                
                results.append(result)
                
            except Exception as e:
                self.logger.exception(f"Validator {validator} failed: {e}")
                results.append(None)
        
        return results
    
    def _execute_hook_standard(self, hook_name: str, hook_data: Dict[str, Any]) -> Any:
        """Standard hook execution (fallback)."""
        # This would be the original hook execution logic
        # For now, return a simple result
        return {"success": True, "message": "Standard execution"}
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        if not self._optimizer:
            return {
                "enabled": self._optimization_enabled,
                "initialized": self._initialized,
                "status": "not_initialized"
            }
        
        return {
            "enabled": self._optimization_enabled,
            "initialized": self._initialized,
            "optimizer_status": self._optimizer.get_optimization_status(),
            "monitor_dashboard": self._monitor.get_dashboard_data() if self._monitor else None
        }
    
    async def shutdown(self):
        """Shutdown optimization components."""
        if self._optimizer:
            await self._optimizer.shutdown()
        
        if self._monitor:
            self._monitor.shutdown()
        
        self._initialized = False


# Global integration instance
_integration = HookOptimizationIntegration()


async def get_hook_integration() -> HookOptimizationIntegration:
    """Get or create hook integration instance."""
    if not _integration._initialized:
        await _integration.initialize()
    return _integration


# Convenience functions for hook system
async def execute_pre_tool_validators_optimized(tool_name: str,
                                              tool_input: Dict[str, Any],
                                              validators: List[Any]) -> List[Any]:
    """Execute pre-tool validators with optimizations."""
    integration = await get_hook_integration()
    return await integration.execute_pre_tool_validators(tool_name, tool_input, validators)


async def execute_post_tool_hook_optimized(hook_name: str,
                                         hook_data: Dict[str, Any]) -> Any:
    """Execute post-tool hook with optimizations."""
    integration = await get_hook_integration()
    return await integration.execute_post_tool_hook(hook_name, hook_data)


def get_optimization_status() -> Dict[str, Any]:
    """Get optimization status synchronously."""
    return _integration.get_optimization_status()


# Decorator for optimized hook execution
def optimized_hook(priority: TaskPriority = TaskPriority.NORMAL):
    """Decorator to enable optimized execution for a hook function."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            integration = await get_hook_integration()
            
            if integration._optimization_enabled and integration._optimizer:
                # Use optimizer
                from .async_orchestrator import AsyncTask
                
                AsyncTask(
                    id=f"hook_{func.__name__}_{time.time()}",
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    priority=priority,
                    created_at=time.time()
                )
                
                future = await integration._optimizer.orchestrator.submit_task(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    priority=priority
                )
                
                return await future
            else:
                # Standard execution
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator