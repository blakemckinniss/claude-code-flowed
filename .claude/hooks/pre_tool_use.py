#!/usr/bin/env python3
"""Optimized PreToolUse hook handler with performance enhancements.

Integrates all optimization modules for high-performance validation:
- Hook execution pooling to eliminate cold starts
- Parallel validator execution
- Smart caching for validation results
- Asynchronous database operations
- Circuit breaker for resilience
- Memory pooling for efficiency
"""

import json
import sys
from pathlib import Path
import time
import asyncio
from typing import Any, Dict, Optional

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Now we can import cleanly without explicit sys.path.insert calls

# Import optimization modules
try:
    from modules.optimization import (
        AsyncDatabaseManager,
        BoundedPatternStorage,
        ContextTracker,
        HookCircuitBreaker,
        HookExecutionPool,
        HookPipeline,
        ParallelValidationManager,
        PerformanceMetricsCache,
        ValidatorCache,
    )
    # Import new integrated optimization system
    from modules.optimization.hook_integration import (
        get_hook_integration,
        execute_pre_tool_validators_optimized,
        get_optimization_status
    )
    OPTIMIZATION_AVAILABLE = True
    NEW_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Optimization modules not available: {e}", file=sys.stderr)
    OPTIMIZATION_AVAILABLE = False
    NEW_OPTIMIZER_AVAILABLE = False
    # Set to None for type checking
    HookExecutionPool = None
    ValidatorCache = None
    PerformanceMetricsCache = None
    AsyncDatabaseManager = None
    ParallelValidationManager = None
    HookCircuitBreaker = None
    ContextTracker = None
    BoundedPatternStorage = None
    HookPipeline = None
    get_hook_integration = None
    execute_pre_tool_validators_optimized = None
    get_optimization_status = None

# Import memory integration
try:
    from modules.memory.hook_memory_integration import get_hook_memory_integration
    MEMORY_INTEGRATION = True
except ImportError:
    get_hook_memory_integration = None
    MEMORY_INTEGRATION = False

# Global optimization instances - initialized once
_hook_pool: Optional[Any] = None
_validator_cache: Optional[Any] = None
_metrics_cache: Optional[Any] = None
_async_db: Optional[Any] = None
_parallel_validator: Optional[Any] = None
_circuit_breaker: Optional[Any] = None
_context_tracker: Optional[Any] = None
_pattern_storage: Optional[Any] = None
_pipeline: Optional[Any] = None

def initialize_optimization():
    """Initialize all optimization components."""
    global _hook_pool, _validator_cache, _metrics_cache, _async_db
    global _parallel_validator, _circuit_breaker, _context_tracker
    global _pattern_storage, _pipeline

    try:
        # Initialize execution pool
        if HookExecutionPool:
            _hook_pool = HookExecutionPool(pool_size=6)

        # Initialize caches
        if ValidatorCache:
            _validator_cache = ValidatorCache(ttl=300, max_size=1000)
        if PerformanceMetricsCache:
            _metrics_cache = PerformanceMetricsCache(
                write_interval=5.0,
                batch_size=100
            )

        # Initialize async database
        if AsyncDatabaseManager:
            _async_db = AsyncDatabaseManager(
                db_path=Path("/home/devcontainers/flowed/.claude/hooks/db/hooks.db"),
                batch_size=100,
                batch_timeout=5.0
            )

        # Initialize parallel validator
        if ParallelValidationManager:
            _parallel_validator = ParallelValidationManager(max_workers=4)

        # Initialize circuit breaker
        if HookCircuitBreaker:
            _circuit_breaker = HookCircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30.0,
                half_open_max_calls=3
            )

        # Initialize memory management
        if ContextTracker:
            _context_tracker = ContextTracker()
        if BoundedPatternStorage:
            _pattern_storage = BoundedPatternStorage(max_patterns=500)

        # Initialize pipeline
        _pipeline = create_optimized_validation_pipeline()

        print("⚡ Optimization infrastructure initialized", file=sys.stderr)

    except Exception as e:
        print(f"Warning: Failed to initialize optimization: {e}", file=sys.stderr)
        # Reset global optimization flag if initialization completely fails


def create_optimized_validation_pipeline() -> Optional[Any]:
    """Create an optimized validation pipeline."""
    if not OPTIMIZATION_AVAILABLE or not HookPipeline:
        return None

    pipeline = HookPipeline(max_workers=6)

    # Stage 1: Parse and cache check
    def parse_and_check_cache(data, context):
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})

        # Check cache first using the correct method
        if _validator_cache and hasattr(_validator_cache, "get_validation_result"):
            cached_result = _validator_cache.get_validation_result(tool_name, tool_input)

            if cached_result is not None:
                context["cached"] = True
                return cached_result

        context["cached"] = False
        context["cache_key"] = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
        return data

    # Stage 2: Parallel validation
    def parallel_validation(data, context):
        if context.get("cached"):
            return data

        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})

        # Define validators
        validators = []

        if tool_name == "Bash":
            validators.append(("command_safety", validate_bash_safety))
            validators.append(("command_resources", check_command_resources))

        elif tool_name in ["Write", "Edit", "MultiEdit"]:
            validators.append(("file_permissions", check_file_permissions))
            validators.append(("file_pattern", check_file_patterns))

        # Run validators (using sequential fallback for async compatibility)
        if validators:
            validation_dict = {}
            for name, validator in validators:
                try:
                    if callable(validator):
                        validator(tool_name, tool_input)
                        validation_dict[name] = {
                            "success": True,
                            "error": "",
                            "severity": "info",
                            "validator_name": name
                        }
                    else:
                        validation_dict[name] = {
                            "success": False,
                            "error": "Validator not callable",
                            "severity": "error",
                            "validator_name": name
                        }
                except Exception as ve:
                    validation_dict[name] = {
                        "success": False,
                        "error": str(ve),
                        "severity": "error",
                        "validator_name": name
                    }
            data["validation_results"] = validation_dict

        return data

    # Stage 3: Circuit breaker check
    def circuit_breaker_check(data, context):
        if context.get("cached"):
            return data

        data.get("tool_name", "")

        def validation_func():
            # Check if any validation failed
            results = data.get("validation_results", {})
            for result in results.values():
                if not result.get("success", True):
                    raise ValueError(f"Validation failed: {result.get('error')}")
            return data

        if _circuit_breaker:
            try:
                result = asyncio.run(_circuit_breaker.execute_with_breaker(validation_func))
                return result
            except Exception as e:
                # Circuit breaker tripped
                data["circuit_breaker_error"] = str(e)
                return data
        else:
            return validation_func()

    # Stage 4: Context tracking
    def track_context(data, context):
        if context.get("cached"):
            return data

        tool_name = data.get("tool_name", "")
        data.get("tool_input", {})

        if _context_tracker:
            _context_tracker.add_tool_operation(tool_name, {
                "type": "pre_tool_validation",
                "timestamp": time.time(),
                "status": "success" if not data.get("circuit_breaker_error") else "failed"
            })

        return data

    # Stage 5: Cache result
    def cache_result(data, context):
        if context.get("cached"):
            return data

        cache_key = context.get("cache_key")
        if cache_key and _validator_cache and hasattr(_validator_cache, "cache_validation_result"):
            _validator_cache.cache_validation_result(cache_key, data)

        return data

    # Add stages to pipeline
    pipeline.add_stage("parse_cache", parse_and_check_cache)
    pipeline.add_stage("validation", parallel_validation)
    pipeline.add_stage("circuit_breaker", circuit_breaker_check)
    pipeline.add_stage("context_tracking", track_context)
    pipeline.add_stage("cache_result", cache_result)

    return pipeline


def main(data: str) -> str:
    """Main hook entry point with optimized processing."""
    try:
        input_data = json.loads(data)
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        
        # Initialize memory integration
        memory_integration = None
        if MEMORY_INTEGRATION and get_hook_memory_integration:
            try:
                memory_integration = get_hook_memory_integration()
                # Capture pre-tool memory asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        memory_integration.capture_pre_tool_memory(tool_name, tool_input)
                    )
                finally:
                    loop.close()
            except Exception as e:
                print(f"Warning: Memory capture failed: {e}", file=sys.stderr)
        
        # Try new integrated optimizer first
        if NEW_OPTIMIZER_AVAILABLE and execute_pre_tool_validators_optimized:
            print("🚀 Using new integrated optimizer for validation...", file=sys.stderr)
            try:
                # Define validators based on tool type
                validators = []
                if tool_name == "Bash":
                    validators.extend([validate_bash_safety, check_command_resources])
                elif tool_name in ["Write", "Edit", "MultiEdit"]:
                    validators.extend([check_file_permissions, check_file_patterns])
                
                if validators:
                    # Run async optimized validation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(
                            execute_pre_tool_validators_optimized(tool_name, tool_input, validators)
                        )
                        # Process results
                        validation_results = {}
                        for i, result in enumerate(results):
                            validator_name = validators[i].__name__ if i < len(validators) else f"validator_{i}"
                            if result is not None and isinstance(result, Exception):
                                validation_results[validator_name] = {
                                    "success": False,
                                    "error": str(result),
                                    "severity": "error"
                                }
                            else:
                                validation_results[validator_name] = {
                                    "success": True,
                                    "error": "",
                                    "severity": "info"
                                }
                        input_data["validation_results"] = validation_results
                    finally:
                        loop.close()
                
                return json.dumps(input_data)
                
            except Exception as e:
                print(f"New optimizer failed, falling back to pipeline: {e}", file=sys.stderr)
        
        # Fall back to original pipeline optimization
        if not _pipeline and OPTIMIZATION_AVAILABLE:
            initialize_optimization()
        
        # Use optimized pipeline if available
        if _pipeline and OPTIMIZATION_AVAILABLE:
            print("⚡ Running optimized validation pipeline...", file=sys.stderr)
            try:
                result = _pipeline.execute(input_data, {})
                if result and hasattr(result, 'data'):
                    return json.dumps(result.data)
                elif result and hasattr(result, 'to_dict'):
                    return json.dumps(result.to_dict())
                elif result:
                    return json.dumps(result)
                else:
                    return json.dumps(input_data)
            except Exception as e:
                print(f"Pipeline failed, falling back to legacy: {e}", file=sys.stderr)
                return fallback_validation(input_data)
        else:
            return fallback_validation(input_data)
            
    except Exception as e:
        # According to Claude Hooks documentation, for PreToolUse hooks:
        # - Exit code 2: Blocks the tool call and shows stderr to Claude
        # - stderr is fed back to Claude for processing
        print(f"Pre-tool validation failed: {e}", file=sys.stderr)
        print("Hook validation error - tool execution blocked for safety", file=sys.stderr)
        sys.exit(2)  # Block tool execution and provide feedback to Claude


def fallback_validation(data: Dict[str, Any]) -> str:
    """Fallback validation logic."""
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    
    validation_results = {}
    
    # Basic validation based on tool type
    if tool_name == "Bash":
        try:
            validate_bash_safety(tool_name, tool_input)
            validation_results["command_safety"] = {"success": True, "error": ""}
        except Exception as e:
            validation_results["command_safety"] = {"success": False, "error": str(e)}
    
    elif tool_name in ["Write", "Edit", "MultiEdit"]:
        try:
            check_file_permissions(tool_name, tool_input)
            validation_results["file_permissions"] = {"success": True, "error": ""}
        except Exception as e:
            validation_results["file_permissions"] = {"success": False, "error": str(e)}
    
    data["validation_results"] = validation_results
    return json.dumps(data)


def validate_bash_safety(tool_name: str, tool_input: Dict[str, Any]) -> None:
    """Validate bash command safety."""
    command = tool_input.get("command", "")
    
    # Basic safety checks
    dangerous_patterns = [
        "rm -rf /", "dd if=", ":(){ :|:& };:", "mkfs", "format",
        "curl", "wget", "ssh", "ftp", "telnet"
    ]
    
    for pattern in dangerous_patterns:
        if pattern in command.lower():
            raise ValueError(f"Potentially dangerous command pattern: {pattern}")


def check_command_resources(tool_name: str, tool_input: Dict[str, Any]) -> None:
    """Check if command might consume excessive resources."""
    command = tool_input.get("command", "")
    
    # Check for resource-intensive operations
    intensive_commands = ["find /", "grep -r", "tar -", "zip -r", "unzip"]
    
    for cmd in intensive_commands:
        if cmd in command:
            print(f"Warning: Command may be resource intensive: {cmd}", file=sys.stderr)


def check_file_permissions(tool_name: str, tool_input: Dict[str, Any]) -> None:
    """Check file operation permissions."""
    path = tool_input.get("path", "")
    
    # Basic path validation
    if path.startswith("/"):
        if not path.startswith("/home/devcontainers/flowed"):
            raise ValueError(f"File operations outside project directory not allowed: {path}")


def check_file_patterns(tool_name: str, tool_input: Dict[str, Any]) -> None:
    """Check for problematic file patterns."""
    path = tool_input.get("path", "")
    
    # Check for system files
    system_paths = ["/etc/", "/usr/", "/var/", "/sys/", "/proc/"]
    
    for sys_path in system_paths:
        if path.startswith(sys_path):
            raise ValueError(f"System file access not allowed: {path}")


def cleanup():
    """Cleanup resources on exit."""
    global _hook_pool, _async_db

    try:
        if _hook_pool and hasattr(_hook_pool, "shutdown"):
            _hook_pool.shutdown()

        if _async_db and hasattr(_async_db, "close"):
            _async_db.close()

    except Exception as e:
        print(f"Error during cleanup: {e}", file=sys.stderr)


if __name__ == "__main__":
    import atexit
    atexit.register(cleanup)
    
    input_data = sys.stdin.read()
    result = main(input_data)
    print(result)
