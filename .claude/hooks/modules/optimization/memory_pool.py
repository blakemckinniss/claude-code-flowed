"""Memory management with object pooling for hook optimization.

This module provides object pooling and bounded collections to reduce memory
allocation overhead and prevent memory leaks in long-running sessions.
"""

import time
import threading
from typing import Dict, Any, Optional, TypeVar, Generic, Type, List, cast
from collections import OrderedDict, deque
from abc import ABC, abstractmethod
import weakref


T = TypeVar('T')


class Poolable(ABC):
    """Base class for poolable objects."""
    
    @abstractmethod
    def reset(self):
        """Reset object to initial state for reuse."""
        pass


class ObjectPool(Generic[T]):
    """Generic object pool for reusing expensive objects."""
    
    def __init__(self,
                 factory: Type[T],
                 max_size: int = 10,
                 pre_create: int = 0):
        """Initialize object pool.
        
        Args:
            factory: Class or factory function to create objects
            max_size: Maximum pool size
            pre_create: Number of objects to pre-create
        """
        self.factory = factory
        self.max_size = max_size
        self._pool: deque = deque()
        self._in_use: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.RLock()
        self._stats = {
            "created": 0,
            "reused": 0,
            "peak_usage": 0
        }
        
        # Pre-create objects
        for _ in range(min(pre_create, max_size)):
            obj = self._create_object()
            self._pool.append(obj)
    
    def _create_object(self) -> T:
        """Create a new object."""
        obj = self.factory()
        self._stats["created"] += 1
        return obj
    
    def acquire(self) -> T:
        """Get an object from the pool."""
        with self._lock:
            # Try to get from pool
            if self._pool:
                obj = self._pool.popleft()
                self._stats["reused"] += 1
            else:
                # Create new object
                obj = self._create_object()
            
            # Track usage
            self._in_use.add(obj)
            current_usage = len(self._in_use)
            if current_usage > self._stats["peak_usage"]:
                self._stats["peak_usage"] = current_usage
            
            # Reset if poolable
            if isinstance(obj, Poolable):
                obj.reset()
            
            return cast('T', obj)
    
    def release(self, obj: T):
        """Return an object to the pool."""
        with self._lock:
            # Remove from in-use tracking
            self._in_use.discard(obj)
            
            # Return to pool if space available
            if len(self._pool) < self.max_size:
                self._pool.append(obj)
    
    def clear(self):
        """Clear the pool."""
        with self._lock:
            self._pool.clear()
            # Note: in_use objects will be garbage collected when released
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "in_use": len(self._in_use),
                "total_created": self._stats["created"],
                "total_reused": self._stats["reused"],
                "peak_usage": self._stats["peak_usage"],
                "efficiency": self._stats["reused"] / max(1, self._stats["created"] + self._stats["reused"])
            }


class ContextTracker(Poolable):
    """Poolable context tracker for validation state."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset to initial state."""
        self.tool_history: List[str] = []
        self.validation_results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.timestamp = time.time()
    
    def add_tool_operation(self, tool_name: str, result: Any):
        """Track a tool operation."""
        self.tool_history.append(tool_name)
        self.validation_results[f"{tool_name}_{len(self.tool_history)}"] = result
    
    def get_recent_operations(self, count: int = 10) -> List[str]:
        """Get recent tool operations."""
        return self.tool_history[-count:]
    
    def should_recycle(self) -> bool:
        """Check if this tracker should be recycled."""
        # Recycle if too old or too many operations
        age = time.time() - self.timestamp
        return age > 300 or len(self.tool_history) > 100


class BoundedPatternStorage:
    """Bounded storage for neural patterns with LRU eviction."""
    
    def __init__(self, max_patterns: int = 1000):
        """Initialize bounded pattern storage.
        
        Args:
            max_patterns: Maximum number of patterns to store
        """
        self.max_patterns = max_patterns
        self.patterns: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.access_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._stats = {
            "stores": 0,
            "retrievals": 0,
            "evictions": 0
        }
    
    def store_pattern(self, key: str, data: Dict[str, Any]):
        """Store a pattern with LRU eviction."""
        with self._lock:
            # Check if we need to evict
            if key not in self.patterns and len(self.patterns) >= self.max_patterns:
                # Evict least recently used
                evicted_key, _ = self.patterns.popitem(last=False)
                self.access_counts.pop(evicted_key, None)
                self._stats["evictions"] += 1
            
            # Store pattern
            self.patterns[key] = {
                "data": data,
                "timestamp": time.time(),
                "access_count": 0
            }
            self.access_counts[key] = 0
            
            # Move to end (most recent)
            self.patterns.move_to_end(key)
            self._stats["stores"] += 1
    
    def add_pattern(self, key: str, data: Dict[str, Any]):
        """Alias for store_pattern to maintain compatibility."""
        self.store_pattern(key, data)
    
    def get_pattern(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a pattern."""
        with self._lock:
            if key in self.patterns:
                # Update access tracking
                self.patterns.move_to_end(key)
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.patterns[key]["access_count"] += 1
                self._stats["retrievals"] += 1
                
                return self.patterns[key]["data"]
            return None
    
    def get_recent_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recently accessed patterns."""
        with self._lock:
            recent = []
            for key in reversed(self.patterns):
                if len(recent) >= limit:
                    break
                pattern_info = self.patterns[key].copy()
                pattern_info["key"] = key
                recent.append(pattern_info)
            return recent
    
    def get_popular_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed patterns."""
        with self._lock:
            # Sort by access count
            sorted_keys = sorted(
                self.access_counts.keys(),
                key=lambda k: self.access_counts[k],
                reverse=True
            )[:limit]
            
            popular = []
            for key in sorted_keys:
                if key in self.patterns:
                    pattern_info = self.patterns[key].copy()
                    pattern_info["key"] = key
                    popular.append(pattern_info)
            return popular
    
    def cleanup_old_patterns(self, max_age_seconds: float = 3600):
        """Remove patterns older than specified age."""
        with self._lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, pattern in self.patterns.items():
                age = current_time - pattern["timestamp"]
                if age > max_age_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.patterns.pop(key, None)
                self.access_counts.pop(key, None)
                self._stats["evictions"] += 1
    
    def get_all_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored patterns as a dictionary."""
        with self._lock:
            result = {}
            for key, pattern_info in self.patterns.items():
                result[key] = pattern_info["data"]
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total_accesses = sum(self.access_counts.values())
            
            return {
                "pattern_count": len(self.patterns),
                "total_stores": self._stats["stores"],
                "total_retrievals": self._stats["retrievals"],
                "total_evictions": self._stats["evictions"],
                "total_accesses": total_accesses,
                "max_patterns": self.max_patterns,
                "utilization": len(self.patterns) / self.max_patterns
            }


# Global instances for memory management
_context_pool: Optional[ObjectPool[ContextTracker]] = None
_pattern_storage: Optional[BoundedPatternStorage] = None


def get_context_pool() -> ObjectPool[ContextTracker]:
    """Get or create the global context tracker pool."""
    global _context_pool
    if _context_pool is None:
        _context_pool = ObjectPool(
            factory=ContextTracker,
            max_size=10,
            pre_create=3
        )
    return _context_pool


def get_pattern_storage() -> BoundedPatternStorage:
    """Get or create the global pattern storage."""
    global _pattern_storage
    if _pattern_storage is None:
        _pattern_storage = BoundedPatternStorage(max_patterns=1000)
    return _pattern_storage