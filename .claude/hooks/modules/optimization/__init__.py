"""Optimization modules for Claude Code hooks.

This package provides high-performance infrastructure for hook execution,
including pooling, caching, and parallel processing capabilities.
"""

from .hook_pool import HookExecutionPool
from .cache import ValidatorCache, PerformanceMetricsCache
from .async_db import AsyncDatabaseManager
from .parallel import ParallelValidationManager
from .circuit_breaker import HookCircuitBreaker
from .memory_pool import ContextTracker, BoundedPatternStorage
from .pipeline import HookPipeline

__all__ = [
    'HookExecutionPool',
    'ValidatorCache',
    'PerformanceMetricsCache',
    'AsyncDatabaseManager',
    'ParallelValidationManager',
    'HookCircuitBreaker',
    'ContextTracker',
    'BoundedPatternStorage',
    'HookPipeline'
]