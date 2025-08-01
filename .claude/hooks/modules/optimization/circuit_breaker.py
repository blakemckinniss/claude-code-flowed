"""Circuit breaker pattern for hook resilience.

This module implements the circuit breaker pattern to prevent cascading failures
and provide graceful degradation when hooks experience repeated failures.
"""

import time
import threading
from typing import Dict, Any, Callable, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    state_changes: list = field(default_factory=list)


class HookCircuitBreaker:
    """Circuit breaker for hook execution."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 half_open_max_calls: int = 3):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Max calls to test in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.half_open_calls = 0
        self._lock = threading.RLock()
        
        # Fallback function when circuit is open
        self._fallback_func = None
    
    def set_fallback(self, fallback_func: Callable[..., Any]):
        """Set fallback function for when circuit is open."""
        self._fallback_func = fallback_func
    
    async def execute_with_breaker(self, 
                                   func: Callable[..., Any],
                                   *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func or fallback
        """
        with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats.rejected_calls += 1
                    return self._get_fallback_result(*args, **kwargs)
            
            elif self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    # Too many test calls, reject
                    self.stats.rejected_calls += 1
                    return self._get_fallback_result(*args, **kwargs)
                self.half_open_calls += 1
        
        # Execute the function
        try:
            self.stats.total_calls += 1
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            with self._lock:
                self._on_success()
            
            return result
            
        except Exception as e:
            with self._lock:
                self._on_failure()
            
            if self.state == CircuitState.OPEN:
                return self._get_fallback_result(*args, **kwargs)
            else:
                raise
    
    def execute_sync(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Synchronous version of execute_with_breaker."""
        with self._lock:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.stats.rejected_calls += 1
                    return self._get_fallback_result(*args, **kwargs)
            
            elif self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    self.stats.rejected_calls += 1
                    return self._get_fallback_result(*args, **kwargs)
                self.half_open_calls += 1
        
        # Execute the function
        try:
            self.stats.total_calls += 1
            result = func(*args, **kwargs)
            
            with self._lock:
                self._on_success()
            
            return result
            
        except Exception as e:
            with self._lock:
                self._on_failure()
            
            if self.state == CircuitState.OPEN:
                return self._get_fallback_result(*args, **kwargs)
            else:
                raise
    
    def _on_success(self):
        """Handle successful execution."""
        self.stats.successful_calls += 1
        self.stats.consecutive_failures = 0
        
        if self.state == CircuitState.HALF_OPEN:
            # Success in half-open state, close the circuit
            self._transition_to_closed()
    
    def _on_failure(self):
        """Handle failed execution."""
        self.stats.failed_calls += 1
        self.stats.consecutive_failures += 1
        self.stats.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state, reopen the circuit
            self._transition_to_open()
        elif (self.state == CircuitState.CLOSED and 
              self.stats.consecutive_failures >= self.failure_threshold):
            # Too many failures, open the circuit
            self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (self.stats.last_failure_time and 
                time.time() - self.stats.last_failure_time > self.recovery_timeout)
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
        self._record_state_change("OPEN", "Too many failures")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
        self.stats.consecutive_failures = 0
        self._record_state_change("CLOSED", "Service recovered")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self._record_state_change("HALF_OPEN", "Testing recovery")
    
    def _record_state_change(self, new_state: str, reason: str):
        """Record state change for monitoring."""
        self.stats.state_changes.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "new_state": new_state,
            "reason": reason,
            "stats": {
                "total_calls": self.stats.total_calls,
                "failed_calls": self.stats.failed_calls,
                "consecutive_failures": self.stats.consecutive_failures
            }
        })
    
    def _get_fallback_result(self, *args, **kwargs) -> Any:
        """Get fallback result when circuit is open."""
        if self._fallback_func:
            try:
                return self._fallback_func(*args, **kwargs)
            except Exception:
                pass
        
        # Default fallback response
        return {
            "success": True,
            "circuit_open": True,
            "message": "Hook execution skipped due to circuit breaker",
            "exit_code": 0
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state and stats."""
        with self._lock:
            success_rate = (
                self.stats.successful_calls / self.stats.total_calls 
                if self.stats.total_calls > 0 else 0.0
            )
            
            return {
                "state": self.state.value,
                "stats": {
                    "total_calls": self.stats.total_calls,
                    "successful_calls": self.stats.successful_calls,
                    "failed_calls": self.stats.failed_calls,
                    "rejected_calls": self.stats.rejected_calls,
                    "success_rate": round(success_rate, 3),
                    "consecutive_failures": self.stats.consecutive_failures
                },
                "config": {
                    "failure_threshold": self.failure_threshold,
                    "recovery_timeout": self.recovery_timeout,
                    "half_open_max_calls": self.half_open_max_calls
                },
                "recent_state_changes": self.stats.state_changes[-5:]  # Last 5 changes
            }
    
    def reset(self):
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.stats = CircuitStats()
            self.half_open_calls = 0
            self._record_state_change("CLOSED", "Manual reset")


# Fix import for async support
import asyncio


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different hooks."""
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """Initialize circuit breaker manager."""
        self.breakers: Dict[str, HookCircuitBreaker] = {}
        self.default_config = default_config or {
            "failure_threshold": 5,
            "recovery_timeout": 30.0,
            "half_open_max_calls": 3
        }
        self._lock = threading.Lock()
    
    def get_breaker(self, hook_name: str) -> HookCircuitBreaker:
        """Get or create circuit breaker for a hook."""
        with self._lock:
            if hook_name not in self.breakers:
                self.breakers[hook_name] = HookCircuitBreaker(**self.default_config)
            return self.breakers[hook_name]
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_state() 
                for name, breaker in self.breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self.breakers.values():
                breaker.reset()