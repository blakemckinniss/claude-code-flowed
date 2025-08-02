"""Validation result caching with TTL for performance optimization.

This module provides intelligent caching for validation results to avoid redundant
operations and significantly improve hook execution performance.
"""

import time
import threading
import hashlib
import json
import sys
from typing import Dict, Any, Optional, NamedTuple, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Path setup handled by centralized resolver when importing this module


class CacheEntryStatus(Enum):
    """Status of cache entries."""
    FRESH = "fresh"           # Within TTL, use immediately
    STALE = "stale"          # Expired but usable as fallback
    EXPIRED = "expired"       # Too old, should be removed


@dataclass
class CacheEntry:
    """A cache entry with TTL and metadata."""
    key: str
    value: Any
    created_at: float
    ttl_seconds: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    validator_name: str = ""
    tool_name: str = ""
    
    def is_fresh(self) -> bool:
        """Check if entry is within TTL."""
        return (time.time() - self.created_at) < self.ttl_seconds
    
    def is_expired(self, max_stale_time: float = 3600) -> bool:
        """Check if entry is too old to be useful."""
        return (time.time() - self.created_at) > (self.ttl_seconds + max_stale_time)
    
    def get_status(self, max_stale_time: float = 3600) -> CacheEntryStatus:
        """Get the current status of this cache entry."""
        if self.is_fresh():
            return CacheEntryStatus.FRESH
        elif self.is_expired(max_stale_time):
            return CacheEntryStatus.EXPIRED
        else:
            return CacheEntryStatus.STALE
    
    def touch(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = time.time()


class ValidationResultCache:
    """Intelligent caching system for validation results with TTL."""
    
    def __init__(self, 
                 default_ttl: float = 300,  # 5 minutes
                 max_entries: int = 1000,
                 cleanup_interval: float = 60,  # 1 minute
                 max_stale_time: float = 3600):  # 1 hour
        """Initialize validation cache.
        
        Args:
            default_ttl: Default TTL in seconds
            max_entries: Maximum number of cache entries
            cleanup_interval: How often to run cleanup (seconds)
            max_stale_time: Maximum time to keep stale entries (seconds)
        """
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval
        self.max_stale_time = max_stale_time
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stale_hits": 0,
            "evictions": 0,
            "cleanups": 0,
            "total_entries_created": 0
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def get(self, 
            key: str, 
            accept_stale: bool = False,
            touch_entry: bool = True) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            accept_stale: Whether to return stale entries
            touch_entry: Whether to update access tracking
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                self._stats["misses"] += 1
                return None
            
            status = entry.get_status(self.max_stale_time)
            
            if status == CacheEntryStatus.FRESH:
                if touch_entry:
                    entry.touch()
                self._stats["hits"] += 1
                return entry.value
            
            elif status == CacheEntryStatus.STALE and accept_stale:
                if touch_entry:
                    entry.touch()
                self._stats["stale_hits"] += 1
                return entry.value
            
            else:
                # Entry is expired or stale not accepted
                if status == CacheEntryStatus.EXPIRED:
                    del self._cache[key]
                self._stats["misses"] += 1
                return None
    
    def put(self, 
            key: str, 
            value: Any, 
            ttl: Optional[float] = None,
            validator_name: str = "",
            tool_name: str = "") -> None:
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            validator_name: Name of validator that produced this result
            tool_name: Tool name this validation was for
        """
        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_entries:
                self._evict_entries()
            
            ttl = ttl or self.default_ttl
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl,
                validator_name=validator_name,
                tool_name=tool_name
            )
            
            self._cache[key] = entry
            self._stats["total_entries_created"] += 1
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def get_cache_key(self, 
                     tool_name: str, 
                     tool_data: Dict[str, Any], 
                     validator_name: str,
                     context_signature: Optional[str] = None) -> str:
        """Generate cache key for validation result.
        
        Args:
            tool_name: Name of the tool
            tool_data: Tool input data
            validator_name: Name of validator
            context_signature: Optional context signature
            
        Returns:
            Cache key string
        """
        # Create deterministic hash of tool data
        tool_data_str = json.dumps(tool_data, sort_keys=True, default=str)
        data_hash = hashlib.md5(tool_data_str.encode()).hexdigest()[:16]
        
        # Include context signature if available
        context_part = f":{context_signature}" if context_signature else ""
        
        return f"{validator_name}:{tool_name}:{data_hash}{context_part}"
    
    def get_context_signature(self, context_tracker) -> str:
        """Generate signature for current validation context.
        
        Args:
            context_tracker: WorkflowContextTracker instance
            
        Returns:
            Context signature string
        """
        try:
            # Create signature based on current context state
            signature_parts = [
                str(context_tracker.get_coordination_state()),
                str(context_tracker.get_tools_since_zen()),
                str(context_tracker.get_tools_since_flow()),
                str(context_tracker.get_recent_pattern() or "")
            ]
            
            signature_str = "|".join(signature_parts)
            return hashlib.md5(signature_str.encode()).hexdigest()[:8]
            
        except Exception:
            return "default"
    
    def _evict_entries(self) -> None:
        """Evict least recently used entries to make space."""
        if not self._cache:
            return
        
        # Calculate number of entries to evict (25% of max)
        evict_count = max(1, self.max_entries // 4)
        
        # Sort by last accessed time (oldest first)
        entries_by_access = sorted(
            self._cache.items(),
            key=lambda item: item[1].last_accessed
        )
        
        # Evict oldest entries
        for i in range(min(evict_count, len(entries_by_access))):
            key, _ = entries_by_access[i]
            del self._cache[key]
            self._stats["evictions"] += 1
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired entries."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_entries()
            except Exception as e:
                print(f"Warning: Cache cleanup error: {e}", file=sys.stderr)
    
    def _cleanup_expired_entries(self) -> None:
        """Remove expired entries from cache."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.get_status(self.max_stale_time) == CacheEntryStatus.EXPIRED:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                self._stats["cleanups"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests) if total_requests > 0 else 0
            stale_rate = (self._stats["stale_hits"] / total_requests) if total_requests > 0 else 0
            
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "stale_hits": self._stats["stale_hits"],
                "hit_rate": round(hit_rate, 3),
                "stale_rate": round(stale_rate, 3),
                "evictions": self._stats["evictions"],
                "cleanups": self._stats["cleanups"],
                "total_entries_created": self._stats["total_entries_created"],
                "default_ttl": self.default_ttl,
                "max_stale_time": self.max_stale_time
            }
    
    def get_entry_details(self) -> List[Dict[str, Any]]:
        """Get details of all cache entries for debugging.
        
        Returns:
            List of entry details
        """
        with self._lock:
            entries = []
            for key, entry in self._cache.items():
                age = time.time() - entry.created_at
                status = entry.get_status(self.max_stale_time)
                
                entries.append({
                    "key": key,
                    "validator": entry.validator_name,
                    "tool": entry.tool_name,
                    "age_seconds": round(age, 2),
                    "ttl_seconds": entry.ttl_seconds,
                    "status": status.value,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed
                })
            
            # Sort by age (newest first)
            entries.sort(key=lambda e: e["age_seconds"])
            
            return entries


class SmartValidationCache(ValidationResultCache):
    """Enhanced cache with intelligent TTL and validator-specific optimizations."""
    
    def __init__(self):
        """Initialize smart cache with optimized settings."""
        super().__init__(
            default_ttl=600,      # 10 minutes default
            max_entries=2000,     # Larger cache
            cleanup_interval=120, # Cleanup every 2 minutes
            max_stale_time=7200   # Keep stale entries for 2 hours
        )
        
        # Validator-specific TTL settings
        self.validator_ttls = {
            # Fast-changing validators need shorter TTL
            "concurrent_execution_validator": 60,      # 1 minute
            "mcp_separation_validator": 120,           # 2 minutes
            "agent_patterns_validator": 300,           # 5 minutes
            
            # Stable validators can cache longer
            "rogue_system_validator": 1800,            # 30 minutes
            "safety_validator": 1800,                  # 30 minutes
            "overwrite_protection_validator": 3600,    # 1 hour
            
            # Structural validators cache very long
            "duplication_detection_validator": 7200,   # 2 hours
            "conflicting_architecture_validator": 7200 # 2 hours
        }
    
    def get_ttl_for_validator(self, validator_name: str) -> float:
        """Get appropriate TTL for a validator.
        
        Args:
            validator_name: Name of the validator
            
        Returns:
            TTL in seconds
        """
        return self.validator_ttls.get(validator_name, self.default_ttl)
    
    def cache_validation_result(self, 
                               tool_name: str,
                               tool_data: Dict[str, Any],
                               validator_name: str,
                               result: Any,
                               context_tracker) -> None:
        """Cache a validation result with smart TTL.
        
        Args:
            tool_name: Name of the tool
            tool_data: Tool input data
            validator_name: Name of validator
            result: Validation result to cache
            context_tracker: Context tracker for signature
        """
        # Get context signature
        context_sig = self.get_context_signature(context_tracker)
        
        # Generate cache key
        cache_key = self.get_cache_key(
            tool_name, tool_data, validator_name, context_sig
        )
        
        # Get appropriate TTL
        ttl = self.get_ttl_for_validator(validator_name)
        
        # Store in cache
        self.put(
            key=cache_key,
            value=result,
            ttl=ttl,
            validator_name=validator_name,
            tool_name=tool_name
        )
    
    def get_cached_validation_result(self, 
                                   tool_name: str,
                                   tool_data: Dict[str, Any],
                                   validator_name: str,
                                   context_tracker,
                                   accept_stale: bool = True) -> Optional[Any]:
        """Get cached validation result.
        
        Args:
            tool_name: Name of the tool
            tool_data: Tool input data
            validator_name: Name of validator
            context_tracker: Context tracker for signature
            accept_stale: Whether to accept stale results
            
        Returns:
            Cached validation result or None
        """
        # Get context signature
        context_sig = self.get_context_signature(context_tracker)
        
        # Generate cache key
        cache_key = self.get_cache_key(
            tool_name, tool_data, validator_name, context_sig
        )
        
        # Try to get from cache
        return self.get(cache_key, accept_stale=accept_stale)


# Global cache instance
_global_validation_cache: Optional[SmartValidationCache] = None


def get_validation_cache() -> SmartValidationCache:
    """Get or create the global validation cache instance."""
    global _global_validation_cache
    if _global_validation_cache is None:
        _global_validation_cache = SmartValidationCache()
    return _global_validation_cache


def clear_validation_cache() -> None:
    """Clear the global validation cache."""
    global _global_validation_cache
    if _global_validation_cache:
        _global_validation_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get validation cache statistics."""
    cache = get_validation_cache()
    return cache.get_stats()