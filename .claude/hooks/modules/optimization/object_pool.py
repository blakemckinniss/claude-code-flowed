"""Object pooling system for memory efficiency in hook operations.

This module provides thread-safe object pools to reduce memory allocation
overhead and improve performance by reusing validation objects and data structures.
"""

import threading
import time
import sys
from typing import Dict, Any, List, Optional, Type, Callable, Generic, TypeVar
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from pathlib import Path
import gc

# Path setup handled by centralized resolver when importing this module

T = TypeVar('T')


class PoolPolicy(Enum):
    """Pool management policies."""
    LRU = "lru"           # Least Recently Used
    LIFO = "lifo"         # Last In, First Out (stack-like)
    FIFO = "fifo"         # First In, First Out (queue-like)


@dataclass
class PoolStats:
    """Statistics for object pool performance."""
    created_objects: int = 0
    borrowed_objects: int = 0
    returned_objects: int = 0
    destroyed_objects: int = 0
    current_size: int = 0
    max_size_reached: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_borrow_time: float = 0.0
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total) if total > 0 else 0.0
    
    def get_efficiency_score(self) -> float:
        """Calculate pool efficiency score (0-1)."""
        if self.created_objects == 0 or self.borrowed_objects == 0:
            return 0.0
        reuse_rate = (self.borrowed_objects - self.created_objects) / self.borrowed_objects
        return max(0.0, reuse_rate)


@dataclass
class PooledObject:
    """Wrapper for pooled objects with metadata."""
    obj: Any
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    in_use: bool = False
    
    def touch(self) -> None:
        """Update last used timestamp and increment use count."""
        self.last_used = time.time()
        self.use_count += 1


class ObjectPool(Generic[T]):
    """Thread-safe object pool for efficient memory management."""
    
    def __init__(self,
                 factory: Callable[[], T],
                 reset_func: Optional[Callable[[T], None]] = None,
                 max_size: int = 50,
                 min_size: int = 5,
                 policy: PoolPolicy = PoolPolicy.LRU,
                 max_idle_time: float = 300.0,  # 5 minutes
                 cleanup_interval: float = 60.0):  # 1 minute
        """Initialize object pool.
        
        Args:
            factory: Function to create new objects
            reset_func: Function to reset objects before reuse
            max_size: Maximum number of objects in pool
            min_size: Minimum number of objects to maintain
            policy: Pool management policy
            max_idle_time: Maximum time object can remain idle (seconds)
            cleanup_interval: How often to run cleanup (seconds)
        """
        self.factory = factory
        self.reset_func = reset_func
        self.max_size = max_size
        self.min_size = min_size
        self.policy = policy
        self.max_idle_time = max_idle_time
        self.cleanup_interval = cleanup_interval
        
        # Pool storage based on policy
        if policy == PoolPolicy.LRU:
            self._pool: Dict[int, PooledObject] = {}
        elif policy == PoolPolicy.LIFO:
            self._pool: deque = deque()
        else:  # FIFO
            self._pool: deque = deque()
        
        # Thread safety
        self._lock = threading.RLock()
        self._stats = PoolStats()
        
        # Active objects tracking
        self._active_objects: Dict[int, PooledObject] = {}
        
        # Cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True
        )
        self._cleanup_thread.start()
        
        # Pre-populate with minimum objects
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Pre-populate pool with minimum objects."""
        with self._lock:
            for _ in range(self.min_size):
                obj = self._create_object()
                self._add_to_pool(obj)
    
    def _create_object(self) -> PooledObject:
        """Create a new pooled object."""
        try:
            raw_obj = self.factory()
            pooled_obj = PooledObject(obj=raw_obj)
            self._stats.created_objects += 1
            return pooled_obj
        except Exception as e:
            print(f"Warning: Failed to create pooled object: {e}", file=sys.stderr)
            raise
    
    def _add_to_pool(self, pooled_obj: PooledObject) -> None:
        """Add object to pool based on policy."""
        pooled_obj.in_use = False
        
        if self.policy == PoolPolicy.LRU:
            self._pool[id(pooled_obj)] = pooled_obj
        else:  # LIFO or FIFO
            self._pool.append(pooled_obj)
        
        self._stats.current_size = len(self._pool)
        self._stats.max_size_reached = max(self._stats.max_size_reached, self._stats.current_size)
    
    def _remove_from_pool(self) -> Optional[PooledObject]:
        """Remove object from pool based on policy."""
        if not self._pool:
            return None
        
        if self.policy == PoolPolicy.LRU:
            # Find least recently used
            if self._pool:
                lru_key = min(self._pool.keys(), key=lambda k: self._pool[k].last_used)
                return self._pool.pop(lru_key)
        elif self.policy == PoolPolicy.LIFO:
            return self._pool.pop()  # Remove from end (stack)
        else:  # FIFO
            return self._pool.popleft()  # Remove from front (queue)
        
        return None
    
    def borrow(self) -> T:
        """Borrow an object from the pool."""
        start_time = time.time()
        
        with self._lock:
            # Try to get from pool
            pooled_obj = self._remove_from_pool()
            
            if pooled_obj is None:
                # Pool is empty, create new object
                if len(self._active_objects) >= self.max_size:
                    # Pool at capacity, force cleanup
                    self._force_cleanup()
                    pooled_obj = self._remove_from_pool()
                
                if pooled_obj is None:
                    # Still no object, create new one
                    pooled_obj = self._create_object()
                    self._stats.cache_misses += 1
                else:
                    self._stats.cache_hits += 1
            else:
                self._stats.cache_hits += 1
            
            # Reset object if reset function provided
            if self.reset_func:
                try:
                    self.reset_func(pooled_obj.obj)
                except Exception as e:
                    print(f"Warning: Failed to reset pooled object: {e}", file=sys.stderr)
                    # Create new object if reset fails
                    pooled_obj = self._create_object()
            
            # Mark as in use and track
            pooled_obj.in_use = True
            pooled_obj.touch()
            self._active_objects[id(pooled_obj)] = pooled_obj
            
            # Update stats
            self._stats.borrowed_objects += 1
            borrow_time = time.time() - start_time
            self._update_average_borrow_time(borrow_time)
            
            return pooled_obj.obj
    
    def return_object(self, obj: T) -> None:
        """Return an object to the pool."""
        with self._lock:
            # Find the pooled object wrapper
            pooled_obj = None
            for obj_id, active_obj in self._active_objects.items():
                if active_obj.obj is obj:
                    pooled_obj = active_obj
                    del self._active_objects[obj_id]
                    break
            
            if pooled_obj is None:
                print("Warning: Attempted to return unknown object to pool", file=sys.stderr)
                return
            
            # Check if pool has space
            if len(self._pool) < self.max_size:
                self._add_to_pool(pooled_obj)
                self._stats.returned_objects += 1
            else:
                # Pool is full, destroy object
                self._destroy_object(pooled_obj)
    
    def _destroy_object(self, pooled_obj: PooledObject) -> None:
        """Destroy a pooled object."""
        try:
            # Call cleanup if object has cleanup method
            if hasattr(pooled_obj.obj, 'cleanup'):
                pooled_obj.obj.cleanup()
            del pooled_obj.obj
            self._stats.destroyed_objects += 1
        except Exception as e:
            print(f"Warning: Error destroying pooled object: {e}", file=sys.stderr)
    
    def _update_average_borrow_time(self, borrow_time: float) -> None:
        """Update average borrow time."""
        if self._stats.borrowed_objects == 1:
            self._stats.average_borrow_time = borrow_time
        else:
            # Running average
            self._stats.average_borrow_time = (
                (self._stats.average_borrow_time * (self._stats.borrowed_objects - 1) + borrow_time)
                / self._stats.borrowed_objects
            )
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of idle objects."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_idle_objects()
            except Exception as e:
                print(f"Warning: Pool cleanup error: {e}", file=sys.stderr)
    
    def _cleanup_idle_objects(self) -> None:
        """Remove idle objects that exceed max idle time."""
        with self._lock:
            current_time = time.time()
            
            if self.policy == PoolPolicy.LRU:
                # Remove idle objects from LRU pool
                idle_keys = []
                for obj_id, pooled_obj in self._pool.items():
                    if (current_time - pooled_obj.last_used) > self.max_idle_time:
                        idle_keys.append(obj_id)
                
                # Don't remove below minimum size
                can_remove = max(0, len(self._pool) - self.min_size)
                for obj_id in idle_keys[:can_remove]:
                    pooled_obj = self._pool.pop(obj_id)
                    self._destroy_object(pooled_obj)
            
            else:  # LIFO or FIFO
                # Remove idle objects from deque
                objects_to_remove = []
                for pooled_obj in self._pool:
                    if (current_time - pooled_obj.last_used) > self.max_idle_time:
                        objects_to_remove.append(pooled_obj)
                
                # Don't remove below minimum size
                can_remove = max(0, len(self._pool) - self.min_size)
                for pooled_obj in objects_to_remove[:can_remove]:
                    self._pool.remove(pooled_obj)
                    self._destroy_object(pooled_obj)
            
            self._stats.current_size = len(self._pool)
    
    def _force_cleanup(self) -> None:
        """Force cleanup when pool reaches capacity."""
        with self._lock:
            # Cleanup active objects that might be leaked
            leaked_objects = []
            for obj_id, pooled_obj in list(self._active_objects.items()):
                if not pooled_obj.in_use:
                    leaked_objects.append(obj_id)
            
            # Return leaked objects to pool
            for obj_id in leaked_objects:
                pooled_obj = self._active_objects.pop(obj_id)
                if len(self._pool) < self.max_size:
                    self._add_to_pool(pooled_obj)
                else:
                    self._destroy_object(pooled_obj)
    
    def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        with self._lock:
            stats_copy = PoolStats()
            stats_copy.created_objects = self._stats.created_objects
            stats_copy.borrowed_objects = self._stats.borrowed_objects
            stats_copy.returned_objects = self._stats.returned_objects
            stats_copy.destroyed_objects = self._stats.destroyed_objects
            stats_copy.current_size = len(self._pool)
            stats_copy.max_size_reached = self._stats.max_size_reached
            stats_copy.cache_hits = self._stats.cache_hits
            stats_copy.cache_misses = self._stats.cache_misses
            stats_copy.average_borrow_time = self._stats.average_borrow_time
            return stats_copy
    
    def clear(self) -> None:
        """Clear all objects from pool."""
        with self._lock:
            # Destroy all pooled objects
            if self.policy == PoolPolicy.LRU:
                for pooled_obj in self._pool.values():
                    self._destroy_object(pooled_obj)
                self._pool.clear()
            else:
                while self._pool:
                    pooled_obj = self._pool.pop()
                    self._destroy_object(pooled_obj)
            
            # Clear active objects (they're still in use)
            # Just clear the tracking dict
            self._active_objects.clear()
            
            self._stats.current_size = 0
    
    def resize(self, new_max_size: int, new_min_size: Optional[int] = None) -> None:
        """Resize the pool."""
        with self._lock:
            self.max_size = new_max_size
            if new_min_size is not None:
                self.min_size = new_min_size
            
            # Adjust current pool size if needed
            current_size = len(self._pool)
            if current_size > new_max_size:
                # Remove excess objects
                excess = current_size - new_max_size
                for _ in range(excess):
                    pooled_obj = self._remove_from_pool()
                    if pooled_obj:
                        self._destroy_object(pooled_obj)
            elif current_size < self.min_size:
                # Add more objects
                needed = self.min_size - current_size
                for _ in range(needed):
                    obj = self._create_object()
                    self._add_to_pool(obj)


class ValidationObjectPools:
    """Manager for all validation object pools."""
    
    def __init__(self):
        """Initialize validation object pools."""
        self._pools: Dict[str, ObjectPool] = {}
        self._lock = threading.RLock()
        
        # Initialize common pools
        self._initialize_common_pools()
    
    def _initialize_common_pools(self) -> None:
        """Initialize commonly used object pools."""
        
        # ValidationResult objects pool
        def create_validation_result_dict():
            return {
                "message": "",
                "severity": "ALLOW",
                "suggested_alternative": None,
                "hive_guidance": None,
                "priority_score": 0,
                "violation_type": None,
                "blocking_reason": None
            }
        
        def reset_validation_result(obj):
            obj.clear()
            obj.update({
                "message": "",
                "severity": "ALLOW", 
                "suggested_alternative": None,
                "hive_guidance": None,
                "priority_score": 0,
                "violation_type": None,
                "blocking_reason": None
            })
        
        self._pools["validation_result"] = ObjectPool(
            factory=create_validation_result_dict,
            reset_func=reset_validation_result,
            max_size=100,
            min_size=10,
            policy=PoolPolicy.LRU
        )
        
        # Context tracker data pool
        def create_context_data():
            return {
                "tools": [],
                "zen_calls": 0,
                "flow_calls": 0,
                "patterns": [],
                "state": "disconnected"
            }
        
        def reset_context_data(obj):
            obj["tools"].clear()
            obj["zen_calls"] = 0
            obj["flow_calls"] = 0
            obj["patterns"].clear()
            obj["state"] = "disconnected"
        
        self._pools["context_data"] = ObjectPool(
            factory=create_context_data,
            reset_func=reset_context_data,
            max_size=50,
            min_size=5,
            policy=PoolPolicy.FIFO
        )
        
        # Tool data analysis objects
        def create_analysis_data():
            return {
                "complexity_score": 0,
                "risk_level": "low",
                "recommendations": [],
                "metadata": {}
            }
        
        def reset_analysis_data(obj):
            obj["complexity_score"] = 0
            obj["risk_level"] = "low"
            obj["recommendations"].clear()
            obj["metadata"].clear()
        
        self._pools["analysis_data"] = ObjectPool(
            factory=create_analysis_data,
            reset_func=reset_analysis_data,
            max_size=75,
            min_size=8,
            policy=PoolPolicy.LRU
        )
    
    def get_pool(self, pool_name: str) -> Optional[ObjectPool]:
        """Get an object pool by name."""
        with self._lock:
            return self._pools.get(pool_name)
    
    def register_pool(self, name: str, pool: ObjectPool) -> None:
        """Register a new object pool."""
        with self._lock:
            self._pools[name] = pool
    
    def borrow_object(self, pool_name: str):
        """Borrow an object from specified pool."""
        pool = self.get_pool(pool_name)
        if pool:
            return pool.borrow()
        return None
    
    def return_object(self, pool_name: str, obj) -> None:
        """Return an object to specified pool."""
        pool = self.get_pool(pool_name)
        if pool:
            pool.return_object(obj)
    
    def get_all_stats(self) -> Dict[str, PoolStats]:
        """Get statistics for all pools."""
        with self._lock:
            return {name: pool.get_stats() for name, pool in self._pools.items()}
    
    def cleanup_all(self) -> None:
        """Cleanup all pools."""
        with self._lock:
            for pool in self._pools.values():
                pool.clear()


# Global pools instance
_global_pools: Optional[ValidationObjectPools] = None


def get_object_pools() -> ValidationObjectPools:
    """Get or create the global object pools instance."""
    global _global_pools
    if _global_pools is None:
        _global_pools = ValidationObjectPools()
    return _global_pools


def clear_all_pools() -> None:
    """Clear all object pools."""
    global _global_pools
    if _global_pools:
        _global_pools.cleanup_all()


def get_pool_stats() -> Dict[str, PoolStats]:
    """Get statistics for all object pools."""
    pools = get_object_pools()
    return pools.get_all_stats()