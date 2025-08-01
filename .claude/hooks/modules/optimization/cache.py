"""Caching system for hook validation results and performance metrics.

This module provides intelligent caching to reduce redundant validations
and improve hook execution performance.
"""

import time
import hashlib
import json
import sys
import threading

# Path setup handled by centralized resolver when importing this module
from typing import Dict, Any, Optional, Set
from collections import OrderedDict


class ValidatorCache:
    """Cache validation results to avoid redundant processing."""
    
    def __init__(self, ttl: int = 300, max_size: int = 1000):
        """Initialize cache with TTL and size limits.
        
        Args:
            ttl: Time-to-live in seconds (default: 5 minutes)
            max_size: Maximum number of cached entries
        """
        self.ttl = ttl
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _generate_cache_key(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Generate a cache key from tool name and input."""
        # Create deterministic string representation
        input_str = json.dumps(tool_input, sort_keys=True)
        combined = f"{tool_name}:{input_str}"
        
        # Generate hash for consistent key length
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get_validation_result(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached validation result if available and valid."""
        cache_key = self._generate_cache_key(tool_name, tool_input)
        
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check if entry is still valid
                if time.time() - entry["timestamp"] < self.ttl:
                    # Move to end (LRU behavior)
                    self.cache.move_to_end(cache_key)
                    self._stats["hits"] += 1
                    return entry["result"]
                else:
                    # Expired entry
                    del self.cache[cache_key]
            
            self._stats["misses"] += 1
            return None
    
    def store_result(self, tool_name: str, tool_input: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Store validation result in cache."""
        cache_key = self._generate_cache_key(tool_name, tool_input)
        
        with self.lock:
            # Evict oldest entries if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
                self._stats["evictions"] += 1
            
            # Store new entry
            self.cache[cache_key] = {
                "timestamp": time.time(),
                "result": result
            }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl,
                "max_size": self.max_size
            }


class PerformanceMetricsCache:
    """In-memory cache for performance metrics with write-behind persistence."""
    
    def __init__(self, write_interval: float = 5.0, batch_size: int = 100):
        """Initialize metrics cache.
        
        Args:
            write_interval: Seconds between batch writes
            batch_size: Maximum metrics to accumulate before forcing write
        """
        self.write_interval = write_interval
        self.batch_size = batch_size
        self.cache: Dict[str, Any] = {}
        self.dirty_keys: Set[str] = set()
        self.pending_metrics: list = []
        self.lock = threading.RLock()
        self._writer_thread = None
        self._running = False
        self._start_background_writer()
    
    def _start_background_writer(self):
        """Start the background writer thread."""
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._background_writer,
            daemon=True
        )
        self._writer_thread.start()
    
    def _background_writer(self):
        """Periodically write dirty metrics to persistent storage."""
        while self._running:
            time.sleep(self.write_interval)
            self._flush_metrics()
    
    def record_metric(self, metric_data: Dict[str, Any]) -> None:
        """Record a performance metric."""
        with self.lock:
            # Generate unique key
            timestamp = metric_data.get("timestamp", time.time())
            operation = metric_data.get("operation_type", "unknown")
            key = f"{operation}_{timestamp}"
            
            # Store in cache
            self.cache[key] = metric_data
            self.dirty_keys.add(key)
            self.pending_metrics.append(metric_data)
            
            # Force flush if batch size reached
            if len(self.pending_metrics) >= self.batch_size:
                self._flush_metrics()
    
    def _flush_metrics(self) -> None:
        """Flush pending metrics to persistent storage."""
        with self.lock:
            if not self.pending_metrics:
                return
            
            # Get metrics to write
            metrics_to_write = self.pending_metrics.copy()
            self.pending_metrics.clear()
            self.dirty_keys.clear()
        
        # Write metrics outside lock
        # This would normally write to database
        # For now, we'll just log the count
        if metrics_to_write:
            print(f"Flushing {len(metrics_to_write)} metrics to storage", file=sys.stderr)
    
    def get_recent_metrics(self, seconds: int = 300) -> list:
        """Get metrics from the last N seconds."""
        cutoff_time = time.time() - seconds
        
        with self.lock:
            recent_metrics = [
                metric for metric in self.cache.values()
                if metric.get("timestamp", 0) > cutoff_time
            ]
        
        return sorted(recent_metrics, key=lambda m: m.get("timestamp", 0))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "cached_metrics": len(self.cache),
                "pending_writes": len(self.pending_metrics),
                "write_interval": self.write_interval,
                "batch_size": self.batch_size
            }
    
    def shutdown(self):
        """Shutdown the cache and flush remaining metrics."""
        self._running = False
        self._flush_metrics()
        if self._writer_thread:
            self._writer_thread.join(timeout=5)