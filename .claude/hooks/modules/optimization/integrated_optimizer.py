"""Integrated Hook System Optimizer.

Combines all optimization modules into a cohesive system with:
- Unified configuration
- Automatic optimization selection
- Performance-based adaptation
- Seamless fallback mechanisms
"""

import asyncio
import os
import sys
import json
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import logging

# Import all optimization modules
from .async_orchestrator import AsyncOrchestrator, HookOrchestratorAdapter, TaskPriority
from .subprocess_coordinator import SubprocessCoordinator, ProcessConfig
from .intelligent_batcher import IntelligentBatcher, BatchingStrategy, BackpressureController
from .performance_monitor import PerformanceMonitor, get_performance_monitor
from .parallel import ParallelValidationManager
from .circuit_breaker import CircuitBreakerManager, HookCircuitBreaker
from .cache import ValidatorCache, PerformanceMetricsCache
from .pipeline import HookPipeline, AsyncHookPipeline
from .hook_pool import HookExecutionPool, get_global_pool


class OptimizationConfig:
    """Configuration for the integrated optimizer."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load optimization configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)
        
        # Default configuration
        return {
            "async_orchestration": {
                "enabled": True,
                "min_workers": 2,
                "max_workers": 8,
                "enable_shared_memory": True
            },
            "subprocess_coordination": {
                "enabled": True,
                "pool_size": 4,
                "memory_limit_mb": 100,
                "enable_cgroups": False
            },
            "intelligent_batching": {
                "enabled": True,
                "strategy": "adaptive",
                "min_batch_size": 5,
                "max_batch_size": 50,
                "batch_timeout_ms": 25
            },
            "circuit_breakers": {
                "enabled": True,
                "failure_threshold": 5,
                "recovery_timeout": 30.0,
                "half_open_max_calls": 3
            },
            "caching": {
                "enabled": True,
                "validator_cache_ttl": 300,
                "validator_cache_size": 1000,
                "metrics_write_interval": 5.0
            },
            "performance_monitoring": {
                "enabled": True,
                "export_interval": 60,
                "anomaly_detection": True,
                "resource_monitoring": True
            },
            "adaptive_optimization": {
                "enabled": True,
                "optimization_interval": 30,
                "performance_threshold": 0.8
            }
        }
    
    def get_module_config(self, module: str) -> Dict[str, Any]:
        """Get configuration for specific module."""
        return self.config.get(module, {})


class AdaptiveOptimizer:
    """Adapts optimization strategies based on performance."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_history = []
        self.current_profile = "balanced"
        
        # Optimization profiles
        self.profiles = {
            "latency": {
                "async_workers": "high",
                "batch_size": "small",
                "cache_aggressive": True,
                "circuit_breaker_sensitive": True
            },
            "throughput": {
                "async_workers": "max",
                "batch_size": "large",
                "cache_aggressive": False,
                "circuit_breaker_sensitive": False
            },
            "balanced": {
                "async_workers": "medium",
                "batch_size": "adaptive",
                "cache_aggressive": True,
                "circuit_breaker_sensitive": True
            },
            "resource_constrained": {
                "async_workers": "min",
                "batch_size": "small",
                "cache_aggressive": True,
                "circuit_breaker_sensitive": True
            }
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance characteristics."""
        metrics = self.monitor.get_dashboard_data()
        
        # Extract key metrics
        hook_metrics = metrics.get("hook_metrics", {})
        resource_usage = metrics.get("resource_usage", {})
        
        # Calculate performance indicators
        avg_latency = hook_metrics.get("duration", {}).get("mean", 0)
        error_rate = hook_metrics.get("errors", {}).get("count", 0) / max(1, hook_metrics.get("executions", {}).get("count", 1))
        cpu_usage = resource_usage.get("cpu_percent", 0)
        memory_usage = resource_usage.get("memory_percent", 0)
        
        return {
            "avg_latency_ms": avg_latency,
            "error_rate": error_rate,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "recommended_profile": self._recommend_profile(
                avg_latency, error_rate, cpu_usage, memory_usage
            )
        }
    
    def _recommend_profile(self, latency: float, error_rate: float, 
                          cpu: float, memory: float) -> str:
        """Recommend optimization profile based on metrics."""
        
        # High resource usage - constrain
        if cpu > 80 or memory > 80:
            return "resource_constrained"
        
        # High error rate - be conservative
        if error_rate > 0.05:
            return "balanced"
        
        # Low latency requirement
        if latency < 50:
            return "latency"
        
        # Can optimize for throughput
        if cpu < 50 and memory < 50:
            return "throughput"
        
        return "balanced"
    
    def apply_profile(self, profile: str) -> Dict[str, Any]:
        """Apply optimization profile."""
        if profile not in self.profiles:
            profile = "balanced"
        
        self.current_profile = profile
        settings = self.profiles[profile]
        
        self.optimization_history.append({
            "timestamp": time.time(),
            "profile": profile,
            "settings": settings
        })
        
        return settings


class IntegratedHookOptimizer:
    """Main integrated optimizer combining all optimization techniques."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = OptimizationConfig(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Start optimization loop
        self._running = True
        self._optimization_task = None
        
        if self.config.get_module_config("adaptive_optimization").get("enabled", True):
            self._start_optimization_loop()
    
    def _initialize_components(self):
        """Initialize all optimization components."""
        
        # Performance monitoring
        self.monitor = get_performance_monitor()
        
        # Async orchestration
        async_config = self.config.get_module_config("async_orchestration")
        if async_config.get("enabled", True):
            self.orchestrator = None  # Will be created async
            self.orchestrator_adapter = None
        
        # Subprocess coordination
        subprocess_config = self.config.get_module_config("subprocess_coordination")
        if subprocess_config.get("enabled", True):
            self.subprocess_coordinator = SubprocessCoordinator(
                pool_size=subprocess_config.get("pool_size", 4)
            )
        
        # Intelligent batching
        batch_config = self.config.get_module_config("intelligent_batching")
        if batch_config.get("enabled", True):
            self.batcher = IntelligentBatcher(
                strategy=BatchingStrategy[batch_config.get("strategy", "ADAPTIVE").upper()],
                min_batch_size=batch_config.get("min_batch_size", 5),
                max_batch_size=batch_config.get("max_batch_size", 50),
                batch_timeout_ms=batch_config.get("batch_timeout_ms", 25)
            )
            self.backpressure = BackpressureController()
        
        # Circuit breakers
        breaker_config = self.config.get_module_config("circuit_breakers")
        if breaker_config.get("enabled", True):
            self.circuit_breakers = CircuitBreakerManager(
                default_config={
                    "failure_threshold": breaker_config.get("failure_threshold", 5),
                    "recovery_timeout": breaker_config.get("recovery_timeout", 30.0),
                    "half_open_max_calls": breaker_config.get("half_open_max_calls", 3)
                }
            )
        
        # Caching
        cache_config = self.config.get_module_config("caching")
        if cache_config.get("enabled", True):
            self.validator_cache = ValidatorCache(
                ttl=cache_config.get("validator_cache_ttl", 300),
                max_size=cache_config.get("validator_cache_size", 1000)
            )
            self.metrics_cache = PerformanceMetricsCache(
                write_interval=cache_config.get("metrics_write_interval", 5.0)
            )
        
        # Hook execution pool
        self.hook_pool = get_global_pool()
        
        # Parallel validation
        self.parallel_validator = ParallelValidationManager(max_workers=4)
        
        # Pipeline
        self.pipeline = HookPipeline(max_workers=4)
        
        # Adaptive optimizer
        self.adaptive_optimizer = AdaptiveOptimizer(self.monitor)
    
    async def initialize_async_components(self):
        """Initialize async components."""
        async_config = self.config.get_module_config("async_orchestration")
        
        if async_config.get("enabled", True):
            self.orchestrator = AsyncOrchestrator(
                min_workers=async_config.get("min_workers", 2),
                max_workers=async_config.get("max_workers", 8),
                enable_shared_memory=async_config.get("enable_shared_memory", True)
            )
            await self.orchestrator.start()
            
            self.orchestrator_adapter = HookOrchestratorAdapter(self.orchestrator)
    
    def _start_optimization_loop(self):
        """Start the adaptive optimization loop."""
        async def optimization_loop():
            interval = self.config.get_module_config("adaptive_optimization").get(
                "optimization_interval", 30
            )
            
            while self._running:
                try:
                    await asyncio.sleep(interval)
                    
                    # Analyze performance
                    analysis = self.adaptive_optimizer.analyze_performance()
                    
                    # Apply optimizations if needed
                    if analysis["recommended_profile"] != self.adaptive_optimizer.current_profile:
                        settings = self.adaptive_optimizer.apply_profile(
                            analysis["recommended_profile"]
                        )
                        await self._apply_optimization_settings(settings)
                        
                        self.logger.info(
                            f"Applied optimization profile: {analysis['recommended_profile']}"
                        )
                
                except Exception as e:
                    self.logger.exception(f"Optimization loop error: {e}")
        
        self._optimization_task = asyncio.create_task(optimization_loop())
    
    async def _apply_optimization_settings(self, settings: Dict[str, Any]):
        """Apply optimization settings from profile."""
        
        # Adjust async workers
        if hasattr(self, 'orchestrator') and self.orchestrator:
            worker_setting = settings.get("async_workers", "medium")
            if worker_setting == "high":
                # Scale up workers
                pass  # Orchestrator handles this automatically
            elif worker_setting == "min":
                # Scale down workers
                pass  # Orchestrator handles this automatically
        
        # Adjust batch size
        if hasattr(self, 'batcher'):
            batch_setting = settings.get("batch_size", "adaptive")
            if batch_setting == "small":
                self.batcher.adaptive_sizer.current_size = self.batcher.min_batch_size
            elif batch_setting == "large":
                self.batcher.adaptive_sizer.current_size = self.batcher.max_batch_size
        
        # Adjust circuit breaker sensitivity
        if hasattr(self, 'circuit_breakers') and settings.get("circuit_breaker_sensitive"):
            # Make circuit breakers more sensitive
            for breaker in self.circuit_breakers.breakers.values():
                breaker.failure_threshold = 3
    
    async def execute_hook_optimized(self,
                                   hook_path: str,
                                   hook_data: Dict[str, Any],
                                   priority: TaskPriority = TaskPriority.NORMAL) -> Dict[str, Any]:
        """Execute hook with all optimizations."""
        
        start_time = time.time()
        
        # Check cache first
        if hasattr(self, 'validator_cache'):
            cached = self.validator_cache.get_validation_result("hook", hook_data)
            if cached:
                self.monitor.metrics.increment("cache.hits", labels={"type": "validator"})
                return cached
        
        # Check circuit breaker
        if hasattr(self, 'circuit_breakers'):
            breaker = self.circuit_breakers.get_breaker(hook_path)
            
            async def execute_with_breaker():
                # Try subprocess pool first
                if hasattr(self, 'subprocess_coordinator'):
                    return self.subprocess_coordinator.execute_hook(
                        hook_path, hook_data
                    )
                else:
                    # Fallback to hook pool
                    return self.hook_pool.execute_hook(
                        "hook", hook_path, hook_data
                    )
            
            try:
                result = await breaker.execute_with_breaker(execute_with_breaker)
            except Exception as e:
                result = {"success": False, "error": str(e)}
        else:
            # Direct execution
            result = self.subprocess_coordinator.execute_hook(
                hook_path, hook_data
            )
        
        # Record metrics
        duration = (time.time() - start_time) * 1000
        self.monitor.record_hook_execution(
            hook_path,
            duration,
            result.get("success", False),
            result.get("error")
        )
        
        # Cache successful results
        if result.get("success") and hasattr(self, 'validator_cache'):
            self.validator_cache.store_result("hook", hook_data, result)
        
        # Record in metrics cache
        if hasattr(self, 'metrics_cache'):
            self.metrics_cache.record_metric({
                "operation_type": "hook_execution",
                "duration": duration / 1000,
                "success": result.get("success", False),
                "hook_path": hook_path
            })
        
        return result
    
    async def execute_validators_optimized(self,
                                         validators: List[Any],
                                         tool_name: str,
                                         tool_input: Dict[str, Any]) -> List[Any]:
        """Execute validators with optimizations."""
        
        # Use orchestrator if available
        if hasattr(self, 'orchestrator_adapter') and self.orchestrator_adapter:
            return await self.orchestrator_adapter.execute_validators_parallel(
                validators, tool_name, tool_input
            )
        
        # Fallback to parallel validation manager
        return await self.parallel_validator.validate_parallel(
            tool_name, tool_input, validators
        )
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        status = {
            "current_profile": self.adaptive_optimizer.current_profile,
            "performance_analysis": self.adaptive_optimizer.analyze_performance()
        }
        
        # Add component stats
        if hasattr(self, 'orchestrator') and self.orchestrator:
            status["orchestrator"] = self.orchestrator.get_stats()
        
        if hasattr(self, 'subprocess_coordinator'):
            status["subprocess"] = self.subprocess_coordinator.get_performance_stats()
        
        if hasattr(self, 'batcher'):
            status["batcher"] = self.batcher.get_stats()
        
        if hasattr(self, 'circuit_breakers'):
            status["circuit_breakers"] = self.circuit_breakers.get_all_states()
        
        if hasattr(self, 'validator_cache'):
            status["cache"] = self.validator_cache.get_stats()
        
        return status
    
    async def shutdown(self):
        """Shutdown the optimizer."""
        self._running = False
        
        if self._optimization_task:
            self._optimization_task.cancel()
        
        # Shutdown components
        if hasattr(self, 'orchestrator') and self.orchestrator:
            await self.orchestrator.shutdown()
        
        if hasattr(self, 'subprocess_coordinator'):
            self.subprocess_coordinator.shutdown()
        
        if hasattr(self, 'batcher'):
            self.batcher.stop()
        
        if hasattr(self, 'monitor'):
            self.monitor.shutdown()
        
        # Save any pending metrics
        if hasattr(self, 'metrics_cache'):
            self.metrics_cache.shutdown()


# Global optimizer instance
_global_optimizer: Optional[IntegratedHookOptimizer] = None


async def get_hook_optimizer() -> IntegratedHookOptimizer:
    """Get or create global hook optimizer."""
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = IntegratedHookOptimizer()
        await _global_optimizer.initialize_async_components()
    
    return _global_optimizer


# Example usage
async def optimized_hook_execution_example():
    """Example of using the integrated optimizer."""
    
    # Get optimizer
    optimizer = await get_hook_optimizer()
    
    # Execute hook with optimizations
    await optimizer.execute_hook_optimized(
        hook_path="/path/to/hook.py",
        hook_data={"tool": "test", "input": {}},
        priority=TaskPriority.HIGH
    )
    
    # Get optimization status
    status = optimizer.get_optimization_status()
    print(f"Optimization Status: {json.dumps(status, indent=2)}")
    
    # Shutdown when done
    await optimizer.shutdown()