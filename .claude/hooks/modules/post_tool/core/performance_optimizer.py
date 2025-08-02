"""Performance Optimization Engine for Tool Analysis.

This module provides high-performance execution strategies, caching,
and resource management to achieve <100ms analysis targets across
all tool analyzers in the universal feedback system.
"""

import asyncio
import time
import threading
import weakref
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import hashlib
import pickle
import logging

from .tool_analyzer_base import ToolContext, FeedbackResult, ToolAnalyzer


@dataclass
class PerformanceMetrics:
    """Performance tracking for analyzers and operations."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def average_duration(self) -> float:
        """Get average execution duration."""
        return self.total_duration / max(1, self.total_executions)
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        return (self.successful_executions / max(1, self.total_executions)) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total_requests = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(1, total_requests)) * 100
    
    def update_execution(self, duration: float, success: bool, cache_hit: bool = False):
        """Update metrics with execution result."""
        self.total_executions += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1


class HighPerformanceCache:
    """High-performance cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._ttl: Dict[str, float] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._metrics = PerformanceMetrics()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            current_time = time.time()
            
            # Check if key exists and is not expired
            if key in self._cache:
                if current_time - self._ttl.get(key, 0) < self.default_ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._access_times[key] = current_time
                    self._metrics.cache_hits += 1
                    return self._cache[key]
                else:
                    # Expired, remove
                    self._remove_key(key)
            
            self._metrics.cache_misses += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache."""
        with self._lock:
            current_time = time.time()
            
            # Remove expired items first
            self._cleanup_expired()
            
            # Add/update item
            self._cache[key] = value
            self._ttl[key] = current_time
            self._access_times[key] = current_time
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Evict if over size limit
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                self._remove_key(oldest_key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all data structures."""
        self._cache.pop(key, None)
        self._ttl.pop(key, None)
        self._access_times.pop(key, None)
    
    def _cleanup_expired(self) -> None:
        """Remove expired items."""
        current_time = time.time()
        expired_keys = [
            key for key, creation_time in self._ttl.items()
            if current_time - creation_time > self.default_ttl
        ]
        for key in expired_keys:
            self._remove_key(key)
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._ttl.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": self._metrics.cache_hit_rate,
                "total_hits": self._metrics.cache_hits,
                "total_misses": self._metrics.cache_misses
            }


class AsyncExecutionPool:
    """High-performance async execution pool for analyzer coordination."""
    
    def __init__(self, max_workers: int = 4, timeout: float = 5.0):
        """Initialize execution pool.
        
        Args:
            max_workers: Maximum concurrent workers
            timeout: Default timeout for operations
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.metrics = PerformanceMetrics()
        self._shutdown = False
    
    async def execute_analyzers(
        self, 
        analyzers: List[ToolAnalyzer], 
        context: ToolContext,
        timeout_override: Optional[float] = None
    ) -> List[Optional[FeedbackResult]]:
        """Execute analyzers concurrently with performance optimization."""
        if not analyzers or self._shutdown:
            return []
        
        start_time = time.time()
        timeout = timeout_override or self.timeout
        
        # Create tasks with semaphore protection
        tasks = [
            self._execute_with_semaphore(analyzer, context, timeout)
            for analyzer in analyzers
        ]
        
        try:
            # Execute with global timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout * 2
            )
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.warning(f"Analyzer {i} failed: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            # Update metrics
            duration = time.time() - start_time
            success = not any(isinstance(r, Exception) for r in results)
            self.metrics.update_execution(duration, success)
            
            return processed_results
        
        except asyncio.TimeoutError:
            logging.warning(f"Analyzer pool timeout after {timeout * 2}s")
            self.metrics.update_execution(time.time() - start_time, False)
            return [None] * len(analyzers)
    
    async def _execute_with_semaphore(
        self, 
        analyzer: ToolAnalyzer, 
        context: ToolContext,
        timeout: float
    ) -> Optional[FeedbackResult]:
        """Execute single analyzer with semaphore protection."""
        async with self.semaphore:
            try:
                return await asyncio.wait_for(
                    analyzer.analyze_tool(context),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logging.warning(f"Analyzer {analyzer.get_analyzer_name()} timed out")
                return None
            except Exception as e:
                logging.warning(f"Analyzer {analyzer.get_analyzer_name()} failed: {e}")
                return None
    
    def shutdown(self):
        """Shutdown execution pool."""
        self._shutdown = True
        self.thread_pool.shutdown(wait=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution pool statistics."""
        return {
            "max_workers": self.max_workers,
            "timeout": self.timeout,
            "metrics": {
                "total_executions": self.metrics.total_executions,
                "success_rate": self.metrics.success_rate,
                "average_duration": self.metrics.average_duration
            }
        }


class CacheKeyGenerator:
    """Generates optimized cache keys for tool contexts."""
    
    @staticmethod
    def generate_key(context: ToolContext, analyzer_name: str) -> str:
        """Generate cache key for tool context and analyzer."""
        # Create deterministic hash from relevant context data
        key_data = {
            "tool_name": context.tool_name,
            "analyzer": analyzer_name,
            # Only include inputs that affect analysis results
            "input_hash": CacheKeyGenerator._hash_dict(context.tool_input),
            "success": context.success
        }
        
        # Create stable string representation
        key_str = f"{key_data['tool_name']}:{key_data['analyzer']}:{key_data['input_hash']}:{key_data['success']}"
        
        # Return SHA-256 hash for consistent key length
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    @staticmethod
    def _hash_dict(data: Dict[str, Any]) -> str:
        """Create stable hash from dictionary."""
        try:
            # Sort keys for deterministic ordering
            sorted_items = sorted(data.items())
            # Use pickle for complex objects, fallback to str
            serialized = pickle.dumps(sorted_items, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(serialized).hexdigest()
        except Exception:
            # Fallback to string representation
            return hashlib.md5(str(sorted(data.items())).encode()).hexdigest()


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(
        self, 
        cache_size: int = 2000,
        max_workers: int = 4,
        cache_ttl: float = 300.0,
        execution_timeout: float = 5.0
    ):
        """Initialize performance optimizer.
        
        Args:
            cache_size: Maximum cache size
            max_workers: Maximum concurrent workers
            cache_ttl: Cache time-to-live in seconds
            execution_timeout: Execution timeout in seconds
        """
        self.cache = HighPerformanceCache(cache_size, cache_ttl)
        self.execution_pool = AsyncExecutionPool(max_workers, execution_timeout)
        self.key_generator = CacheKeyGenerator()
        
        # Performance tracking
        self.global_metrics = PerformanceMetrics()
        self.analyzer_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        
        # Optimization thresholds
        self.slow_analyzer_threshold = 0.5  # seconds
        self.cache_warmup_enabled = True
        
    async def execute_with_optimization(
        self, 
        analyzers: List[ToolAnalyzer], 
        context: ToolContext
    ) -> List[Optional[FeedbackResult]]:
        """Execute analyzers with full performance optimization."""
        start_time = time.time()
        
        # Sort analyzers by priority and performance history
        optimized_analyzers = self._optimize_analyzer_order(analyzers, context)
        
        # Check cache for each analyzer
        cached_results = {}
        analyzers_to_execute = []
        
        for analyzer in optimized_analyzers:
            cache_key = self.key_generator.generate_key(context, analyzer.get_analyzer_name())
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                cached_results[analyzer.get_analyzer_name()] = cached_result
                # Update analyzer metrics for cache hit
                self.analyzer_metrics[analyzer.get_analyzer_name()].update_execution(0.001, True, True)
            else:
                analyzers_to_execute.append(analyzer)
        
        # Execute remaining analyzers
        execution_results = []
        if analyzers_to_execute:
            execution_results = await self.execution_pool.execute_analyzers(
                analyzers_to_execute, context
            )
        
        # Combine cached and executed results
        final_results = []
        exec_index = 0
        
        for analyzer in optimized_analyzers:
            analyzer_name = analyzer.get_analyzer_name()
            
            if analyzer_name in cached_results:
                final_results.append(cached_results[analyzer_name])
            else:
                if exec_index < len(execution_results):
                    result = execution_results[exec_index]
                    final_results.append(result)
                    
                    # Cache successful results
                    if result is not None:
                        cache_key = self.key_generator.generate_key(context, analyzer_name)
                        self.cache.put(cache_key, result)
                    
                    exec_index += 1
                else:
                    final_results.append(None)
        
        # Update global metrics
        total_duration = time.time() - start_time
        self.global_metrics.update_execution(
            total_duration, 
            any(r is not None for r in final_results)
        )
        
        return final_results
    
    def _optimize_analyzer_order(
        self, 
        analyzers: List[ToolAnalyzer], 
        context: ToolContext
    ) -> List[ToolAnalyzer]:
        """Optimize analyzer execution order based on performance history."""
        def analyzer_score(analyzer: ToolAnalyzer) -> Tuple[int, float]:
            """Calculate analyzer score for ordering."""
            name = analyzer.get_analyzer_name()
            metrics = self.analyzer_metrics.get(name, PerformanceMetrics())
            
            # Primary sort: priority (higher first)
            priority = analyzer.get_priority()
            
            # Secondary sort: average execution time (faster first)
            avg_duration = metrics.average_duration if metrics.total_executions > 0 else 0.1
            
            return (priority, -avg_duration)  # Negative for ascending duration sort
        
        return sorted(analyzers, key=analyzer_score, reverse=True)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "global_metrics": {
                "total_executions": self.global_metrics.total_executions,
                "success_rate": self.global_metrics.success_rate,
                "average_duration": self.global_metrics.average_duration,
                "cache_hit_rate": self.global_metrics.cache_hit_rate
            },
            "cache_stats": self.cache.get_stats(),
            "execution_pool_stats": self.execution_pool.get_stats(),
            "analyzer_performance": {
                name: {
                    "executions": metrics.total_executions,
                    "success_rate": metrics.success_rate,
                    "avg_duration": metrics.average_duration,
                    "cache_hit_rate": metrics.cache_hit_rate
                }
                for name, metrics in self.analyzer_metrics.items()
                if metrics.total_executions > 0
            },
            "slow_analyzers": [
                name for name, metrics in self.analyzer_metrics.items()
                if metrics.average_duration > self.slow_analyzer_threshold
            ]
        }
    
    def optimize_cache_settings(self) -> None:
        """Automatically optimize cache settings based on usage patterns."""
        stats = self.cache.get_stats()
        
        # Increase cache size if hit rate is low and we're at capacity
        if stats["hit_rate"] < 70 and stats["size"] >= stats["max_size"] * 0.9:
            new_size = min(stats["max_size"] * 2, 5000)  # Cap at 5000
            self.cache.max_size = new_size
            logging.info(f"Increased cache size to {new_size} due to low hit rate")
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.cache.clear()
    
    def shutdown(self) -> None:
        """Shutdown optimizer and cleanup resources."""
        self.execution_pool.shutdown()
        self.clear_caches()


class ResourceMonitor:
    """Monitor system resources and adjust performance settings."""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        """Initialize resource monitor."""
        self.optimizer = optimizer
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Resource thresholds
        self.cpu_threshold = 80.0  # Percent
        self.memory_threshold = 85.0  # Percent
        self.response_time_threshold = 0.1  # Seconds
    
    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_performance()
                time.sleep(interval)
            except Exception as e:
                logging.warning(f"Resource monitoring error: {e}")
                time.sleep(interval)
    
    def _check_performance(self) -> None:
        """Check performance and adjust settings."""
        # Get current performance metrics
        report = self.optimizer.get_performance_report()
        global_metrics = report["global_metrics"]
        
        # Adjust based on response times
        if global_metrics["average_duration"] > self.response_time_threshold:
            # Reduce concurrent workers to improve response time
            current_workers = self.optimizer.execution_pool.max_workers
            if current_workers > 1:
                new_workers = max(1, current_workers - 1)
                self.optimizer.execution_pool.max_workers = new_workers
                logging.info(f"Reduced workers to {new_workers} due to slow response times")
        
        # Optimize cache settings
        self.optimizer.optimize_cache_settings()


# Global performance optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None
_optimizer_lock = threading.Lock()


def get_global_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer."""
    global _global_optimizer
    
    if _global_optimizer is None:
        with _optimizer_lock:
            if _global_optimizer is None:
                _global_optimizer = PerformanceOptimizer()
    
    return _global_optimizer


def shutdown_global_optimizer() -> None:
    """Shutdown global optimizer."""
    global _global_optimizer
    
    if _global_optimizer is not None:
        _global_optimizer.shutdown()
        _global_optimizer = None