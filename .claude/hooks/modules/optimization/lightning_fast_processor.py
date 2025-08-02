"""Lightning-Fast Hook Processor for Sub-100ms Performance.

This module implements a high-performance processing pipeline optimized for:
- Sub-100ms stderr feedback generation
- Zero-blocking execution
- Async-first architecture with circuit breakers
- Intelligent caching with LRU eviction
- Memory-efficient pattern storage
"""

import asyncio
import time
import functools
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import json
import hashlib
import weakref
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from contextlib import contextmanager


@dataclass
class ProcessingResult:
    """Result of processing with timing metadata."""
    success: bool
    result: Any
    duration_ms: float
    cached: bool = False
    circuit_breaker_triggered: bool = False
    warnings: List[str] = field(default_factory=list)


class LightningCache:
    """Ultra-fast LRU cache with intelligent eviction."""
    
    def __init__(self, maxsize: int = 1000, ttl_seconds: float = 300):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with TTL check."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL
            if time.time() - self._timestamps[key] > self.ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                return None
            
            # Move to end (mark as recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._hits += 1
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache with LRU eviction."""
        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = value
            self._timestamps[key] = time.time()
            
            # Evict if necessary
            while len(self._cache) > self.maxsize:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
    
    def clear_expired(self) -> int:
        """Clear expired entries, return count cleared."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self._timestamps.items()
                if current_time - timestamp > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
                del self._timestamps[key]
            
            return len(expired_keys)
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "ttl_seconds": self.ttl_seconds
        }


class FastCircuitBreaker:
    """Lightning-fast circuit breaker with minimal overhead."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_seconds: float = 30,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = 0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, fallback: Optional[Callable] = None) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            # Check if we should attempt the call
            if self._state == "OPEN":
                if time.time() - self._last_failure_time > self.timeout_seconds:
                    self._state = "HALF_OPEN"
                else:
                    # Circuit is open, use fallback
                    if fallback:
                        return fallback()
                    else:
                        raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func()
            
            with self._lock:
                # Success - reset if we were in HALF_OPEN
                if self._state == "HALF_OPEN":
                    self._state = "CLOSED"
                    self._failure_count = 0
            
            return result
            
        except self.expected_exception:
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if self._failure_count >= self.failure_threshold:
                    self._state = "OPEN"
            
            if fallback:
                return fallback()
            else:
                raise
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = "CLOSED"
            self._failure_count = 0
            self._last_failure_time = 0


class AsyncTaskPool:
    """High-performance async task pool with bounded execution."""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = threading.Semaphore(queue_size)
        self._active_tasks = 0
        self._lock = threading.Lock()
    
    def submit_bounded(self, func: Callable, *args, timeout: float = 5.0, **kwargs) -> Any:
        """Submit task with timeout and bounded execution."""
        if not self._semaphore.acquire(blocking=False):
            raise queue.Full("Task pool is at capacity")
        
        try:
            with self._lock:
                self._active_tasks += 1
            
            future = self._executor.submit(func, *args, **kwargs)
            
            try:
                result = future.result(timeout=timeout)
                return result
            except Exception:
                future.cancel()
                raise
        finally:
            with self._lock:
                self._active_tasks -= 1
            self._semaphore.release()
    
    def submit_parallel(self, tasks: List[Tuple[Callable, tuple, dict]], 
                       timeout: float = 5.0) -> List[Any]:
        """Submit multiple tasks in parallel."""
        futures = []
        results = []
        
        for func, args, kwargs in tasks:
            if not self._semaphore.acquire(blocking=False):
                # If we can't acquire, run remaining tasks synchronously
                results.append(func(*args, **kwargs))
                continue
            
            future = self._executor.submit(func, *args, **kwargs)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures, timeout=timeout):
            try:
                results.append(future.result())
            except Exception as e:
                results.append(e)
            finally:
                self._semaphore.release()
        
        return results
    
    @property
    def active_tasks(self) -> int:
        """Get number of active tasks."""
        return self._active_tasks
    
    def shutdown(self):
        """Shutdown the task pool."""
        self._executor.shutdown(wait=True)


class MemoryEfficientStorage:
    """Memory-efficient pattern storage with automatic cleanup."""
    
    def __init__(self, max_patterns: int = 500, cleanup_interval: int = 100):
        self.max_patterns = max_patterns
        self.cleanup_interval = cleanup_interval
        self._patterns = {}
        self._access_times = {}
        self._operation_count = 0
        self._lock = threading.RLock()
    
    def store_pattern(self, key: str, pattern: Dict[str, Any]):
        """Store pattern with automatic cleanup."""
        with self._lock:
            self._patterns[key] = pattern
            self._access_times[key] = time.time()
            self._operation_count += 1
            
            # Periodic cleanup
            if self._operation_count % self.cleanup_interval == 0:
                self._cleanup_old_patterns()
    
    def get_pattern(self, key: str) -> Optional[Dict[str, Any]]:
        """Get pattern and update access time."""
        with self._lock:
            if key in self._patterns:
                self._access_times[key] = time.time()
                return self._patterns[key]
            return None
    
    def _cleanup_old_patterns(self):
        """Remove least recently used patterns."""
        if len(self._patterns) <= self.max_patterns:
            return
        
        # Sort by access time and remove oldest
        sorted_patterns = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        to_remove = len(self._patterns) - self.max_patterns + 10  # Remove extra for efficiency
        
        for key, _ in sorted_patterns[:to_remove]:
            del self._patterns[key]
            del self._access_times[key]
    
    def size(self) -> int:
        """Get current storage size."""
        return len(self._patterns)


class LightningFastProcessor:
    """Main ultra-fast processing engine for hooks."""
    
    def __init__(self):
        # Performance components
        self.cache = LightningCache(maxsize=1000, ttl_seconds=300)
        self.circuit_breaker = FastCircuitBreaker(failure_threshold=3, timeout_seconds=15)
        self.task_pool = AsyncTaskPool(max_workers=4, queue_size=50)
        self.pattern_storage = MemoryEfficientStorage(max_patterns=500)
        
        # Performance tracking
        self._start_time = time.time()
        self._processed_count = 0
        self._total_processing_time = 0.0
        self._cache_hits = 0
        self._circuit_breaker_trips = 0
        
        # Cleanup timer
        self._cleanup_timer = None
        self._start_cleanup_timer()
    
    def _start_cleanup_timer(self):
        """Start periodic cleanup timer."""
        def cleanup():
            try:
                expired = self.cache.clear_expired()
                if expired > 0:
                    print(f"Cleaned up {expired} expired cache entries", flush=True)
            except Exception:
                pass
            
            # Schedule next cleanup
            self._cleanup_timer = threading.Timer(60.0, cleanup)  # Every minute
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()
        
        cleanup()
    
    def process_tool_analysis(self, 
                            tool_name: str, 
                            tool_input: Dict[str, Any], 
                            tool_response: Dict[str, Any]) -> ProcessingResult:
        """Main processing entry point optimized for speed."""
        start_time = time.perf_counter()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(tool_name, tool_input, tool_response)
            
            # Check cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                return ProcessingResult(
                    success=True,
                    result=cached_result,
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                    cached=True
                )
            
            # Fast-path analysis for common cases
            result = self._fast_path_analysis(tool_name, tool_input, tool_response)
            
            if result is None:
                # Fallback to comprehensive analysis with circuit breaker
                result = self.circuit_breaker.call(
                    lambda: self._comprehensive_analysis(tool_name, tool_input, tool_response),
                    fallback=lambda: self._minimal_analysis(tool_name, tool_input, tool_response)
                )
                
                circuit_breaker_triggered = self.circuit_breaker.state != "CLOSED"
                if circuit_breaker_triggered:
                    self._circuit_breaker_trips += 1
            else:
                circuit_breaker_triggered = False
            
            # Cache the result
            self.cache.put(cache_key, result)
            
            # Update stats
            self._processed_count += 1
            processing_time = (time.perf_counter() - start_time) * 1000
            self._total_processing_time += processing_time
            
            return ProcessingResult(
                success=True,
                result=result,
                duration_ms=processing_time,
                cached=False,
                circuit_breaker_triggered=circuit_breaker_triggered
            )
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return ProcessingResult(
                success=False,
                result={"error": str(e)},
                duration_ms=processing_time,
                warnings=[f"Processing failed: {e}"]
            )
    
    def _generate_cache_key(self, tool_name: str, tool_input: Dict[str, Any], 
                           tool_response: Dict[str, Any]) -> str:
        """Generate cache key for the analysis."""
        # Create a deterministic hash based on inputs
        key_data = {
            "tool": tool_name,
            "input_hash": hashlib.md5(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:8],
            "success": tool_response.get("success", True),
            "has_error": "error" in tool_response
        }
        
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
    
    def _fast_path_analysis(self, tool_name: str, tool_input: Dict[str, Any], 
                           tool_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fast-path analysis for common cases."""
        # Skip analysis for simple successful operations
        if (tool_response.get("success", True) and 
            tool_name in {"LS", "Read", "Glob"} and
            "error" not in tool_response):
            return {"needs_guidance": False, "fast_path": True}
        
        # Quick check for hook file violations
        if tool_name in {"Write", "Edit", "MultiEdit"}:
            file_path = tool_input.get("file_path") or tool_input.get("path", "")
            if ".claude/hooks" in file_path and file_path.endswith(".py"):
                # Check for sys.path manipulation
                content = ""
                if tool_name == "Write":
                    content = tool_input.get("content", "")
                elif tool_name in ["Edit", "MultiEdit"]:
                    if tool_name == "Edit":
                        content = tool_input.get("new_string", "")
                    else:
                        edits = tool_input.get("edits", [])
                        content = "\n".join(edit.get("new_string", "") for edit in edits)
                
                if "sys.path" in content and "path_resolver.py" not in file_path:
                    return {
                        "needs_guidance": True,
                        "guidance_type": "hook_violation",
                        "severity": "high",
                        "fast_path": True
                    }
        
        return None  # Continue to comprehensive analysis
    
    def _comprehensive_analysis(self, tool_name: str, tool_input: Dict[str, Any], 
                               tool_response: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis with parallel execution."""
        analyzers = [
            ("workflow_pattern", self._analyze_workflow_pattern),
            ("resource_usage", self._analyze_resource_usage),
            ("error_patterns", self._analyze_error_patterns)
        ]
        
        # Run analyzers in parallel with timeout
        tasks = [
            (analyzer_func, (tool_name, tool_input, tool_response), {})
            for _, analyzer_func in analyzers
        ]
        
        try:
            results = self.task_pool.submit_parallel(tasks, timeout=3.0)
            
            analysis_results = dict(zip([name for name, _ in analyzers], results))
            
            # Determine if guidance is needed
            needs_guidance = any(
                result.get("needs_guidance", False) if isinstance(result, dict) else False
                for result in results
            )
            
            return {
                "needs_guidance": needs_guidance,
                "analysis": analysis_results,
                "comprehensive": True
            }
            
        except Exception:
            # Fallback to minimal analysis
            return self._minimal_analysis(tool_name, tool_input, tool_response)
    
    def _minimal_analysis(self, tool_name: str, tool_input: Dict[str, Any], 
                         tool_response: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal analysis as fallback."""
        return {
            "needs_guidance": not tool_response.get("success", True),
            "analysis": "minimal_fallback",
            "minimal": True
        }
    
    def _analyze_workflow_pattern(self, tool_name: str, tool_input: Dict[str, Any], 
                                 tool_response: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow patterns."""
        # Simple pattern detection
        if tool_name == "Task":
            return {"needs_guidance": True, "pattern": "agent_spawn"}
        
        if "mcp__" in tool_name:
            return {"needs_guidance": False, "pattern": "mcp_coordination"}
        
        return {"needs_guidance": False, "pattern": "standard"}
    
    def _analyze_resource_usage(self, tool_name: str, tool_input: Dict[str, Any], 
                               tool_response: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        # Check for resource-intensive operations
        if tool_name in {"Bash", "WebSearch", "Task"}:
            return {"needs_guidance": False, "resource_intensive": True}
        
        return {"needs_guidance": False, "resource_intensive": False}
    
    def _analyze_error_patterns(self, tool_name: str, tool_input: Dict[str, Any], 
                              tool_response: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns."""
        if not tool_response.get("success", True):
            error_msg = tool_response.get("error", "").lower()
            
            if "timeout" in error_msg:
                return {"needs_guidance": True, "error_type": "timeout"}
            elif "memory" in error_msg:
                return {"needs_guidance": True, "error_type": "memory"}
            else:
                return {"needs_guidance": True, "error_type": "general"}
        
        return {"needs_guidance": False, "error_type": None}
    
    def generate_guidance_message(self, result: Dict[str, Any]) -> str:
        """Generate user-friendly guidance message."""
        if not result.get("needs_guidance", False):
            return ""
        
        if result.get("fast_path") and result.get("guidance_type") == "hook_violation":
            return """
⚠️ HOOK FILE VIOLATION DETECTED
Hook files should use centralized path management:
  from modules.utils.path_resolver import setup_hook_paths
  setup_hook_paths()

See: .claude/hooks/PATH_MANAGEMENT.md
"""
        
        analysis = result.get("analysis", {})
        
        if isinstance(analysis, dict):
            workflow = analysis.get("workflow_pattern", {})
            if workflow.get("pattern") == "agent_spawn":
                return """
💡 OPTIMIZATION OPPORTUNITY
Consider using ZEN coordination for complex tasks:
  - mcp__zen__planner for task breakdown
  - mcp__claude-flow__swarm_init for parallel execution
"""
        
        return "💡 Consider optimizing this operation for better performance."
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        uptime = time.time() - self._start_time
        avg_processing_time = (
            self._total_processing_time / self._processed_count 
            if self._processed_count > 0 else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "total_processed": self._processed_count,
            "average_processing_time_ms": avg_processing_time,
            "cache_stats": self.cache.stats(),
            "cache_hits": self._cache_hits,
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_trips": self._circuit_breaker_trips,
            "active_tasks": self.task_pool.active_tasks,
            "pattern_storage_size": self.pattern_storage.size(),
            "performance_target_met": avg_processing_time < 100.0  # Sub-100ms target
        }
    
    def shutdown(self):
        """Shutdown the processor."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        self.task_pool.shutdown()


# Global processor instance
_global_processor: Optional[LightningFastProcessor] = None


def get_lightning_processor() -> LightningFastProcessor:
    """Get or create global lightning processor."""
    global _global_processor
    if _global_processor is None:
        _global_processor = LightningFastProcessor()
    return _global_processor


# Convenient wrapper functions
def process_hook_fast(tool_name: str, tool_input: Dict[str, Any], 
                     tool_response: Dict[str, Any]) -> Tuple[str, float]:
    """Process hook with lightning speed and return guidance message and timing."""
    processor = get_lightning_processor()
    result = processor.process_tool_analysis(tool_name, tool_input, tool_response)
    
    guidance_message = processor.generate_guidance_message(result.result)
    
    return guidance_message, result.duration_ms


@contextmanager
def performance_timer(operation_name: str = "operation"):
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        print(f"⚡ {operation_name} completed in {duration_ms:.2f}ms", flush=True)
