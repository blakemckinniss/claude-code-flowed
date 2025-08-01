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
from typing import Any, Dict, Optional

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Now we can import cleanly without explicit sys.path.insert calls
from modules.utils.process_manager import managed_subprocess_run

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
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Optimization modules not available: {e}", file=sys.stderr)
    OPTIMIZATION_AVAILABLE = False
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

# Import existing validation modules
try:
    from modules.pre_tool import (
        DebugValidationReporter,
        GuidanceOutputHandler,
        PreToolAnalysisManager,
    )
    ENHANCED_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced validation modules not available: {e}", file=sys.stderr)
    ENHANCED_VALIDATION_AVAILABLE = False
    # Set to None for type checking
    PreToolAnalysisManager = None
    GuidanceOutputHandler = None
    DebugValidationReporter = None


# Global instances for persistent optimization
_hook_pool: Optional[Any] = None
_validator_cache: Optional[Any] = None
_metrics_cache: Optional[Any] = None
_async_db: Optional[Any] = None
_parallel_validator: Optional[Any] = None
_circuit_breaker: Optional[Any] = None
_context_tracker: Optional[Any] = None
_pattern_storage: Optional[Any] = None
_pipeline: Optional[Any] = None


def initialize_optimization_infrastructure():
    """Initialize all optimization components."""
    global _hook_pool, _validator_cache, _metrics_cache, _async_db
    global _parallel_validator, _circuit_breaker, _context_tracker
    global _pattern_storage, _pipeline

    if not OPTIMIZATION_AVAILABLE:
        print("Warning: Optimization modules not available - skipping initialization", file=sys.stderr)
        return

    try:
        # Initialize hook execution pool
        if HookExecutionPool:
            _hook_pool = HookExecutionPool(pool_size=4)

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
        global OPTIMIZATION_AVAILABLE
        OPTIMIZATION_AVAILABLE = False


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

        # Run validators in parallel
        if validators and _parallel_validator and hasattr(_parallel_validator, "validate_parallel"):
            results = _parallel_validator.validate_parallel(
                tool_input, validators
            )
            data["validation_results"] = results

        return data

    # Stage 3: Circuit breaker check
    def circuit_breaker_check(data, context):
        if context.get("cached"):
            return data

        tool_name = data.get("tool_name", "")

        def validation_func():
            # Check if any validation failed
            results = data.get("validation_results", {})
            for result in results.values():
                if not result.get("success", True):
                    raise ValueError(f"Validation failed: {result.get('error')}")
            return data

        def fallback_func():
            # Fallback to basic validation
            print("⚠️ Circuit breaker open - using fallback validation", file=sys.stderr)
            return data

        if _circuit_breaker and hasattr(_circuit_breaker, "execute_sync"):
            return _circuit_breaker.execute_sync(validation_func)
        return validation_func()

    # Stage 4: Update metrics and cache
    def update_metrics_and_cache(data, context):
        if not context.get("cached"):
            # Cache the result
            tool_name = data.get("tool_name", "")
            tool_input = data.get("tool_input", {})
            if _validator_cache and hasattr(_validator_cache, "store_result"):
                _validator_cache.store_result(tool_name, tool_input, data)

            # Update metrics
            if _metrics_cache and hasattr(_metrics_cache, "record_metric"):
                _metrics_cache.record_metric({
                    "operation_type": "pre_tool_validation",
                    "tool": data.get("tool_name"),
                    "cached": False,
                    "duration": context.get("duration", 0),
                    "timestamp": time.time()
                })

        return data

    # Add stages to pipeline
    if hasattr(pipeline, "add_stage"):
        pipeline.add_stage("parse_cache", parse_and_check_cache, timeout=1.0)
        pipeline.add_stage("validate", parallel_validation, timeout=5.0)
        pipeline.add_stage("circuit_break", circuit_breaker_check, timeout=2.0)
        pipeline.add_stage("metrics", update_metrics_and_cache, timeout=1.0)

    return pipeline


# Validator functions
def validate_bash_safety(command: str) -> Dict[str, Any]:
    """Validate bash command safety."""
    dangerous_patterns = [
        "rm -rf /",
        "dd if=/dev/zero",
        "fork bomb",
        ":(){ :|:& };:",
        "> /dev/sda",
        "chmod -R 777 /"
    ]

    for pattern in dangerous_patterns:
        if pattern in command:
            return {
                "success": False,
                "error": f"Dangerous command pattern detected: {pattern}"
            }

    return {"success": True}


def check_command_resources(command: str) -> Dict[str, Any]:
    """Check if command might consume excessive resources."""
    resource_intensive = [
        "find /",
        "grep -r",
        "du -h /",
        "ps aux"
    ]

    for pattern in resource_intensive:
        if pattern in command and "limit" not in command:
            return {
                "success": True,
                "warning": f"Resource intensive command: {pattern}"
            }

    return {"success": True}


def check_file_permissions(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """Check file permissions before edit."""
    file_path = file_info.get("file_path") or file_info.get("path", "")

    if file_path.startswith("/etc/") or file_path.startswith("/sys/"):
        return {
            "success": False,
            "error": f"Cannot edit system file: {file_path}"
        }

    return {"success": True}


def check_file_patterns(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """Check file patterns for known issues."""
    file_path = file_info.get("file_path") or file_info.get("path", "")

    # Check for pattern matches
    if _pattern_storage and hasattr(_pattern_storage, "get_all_patterns"):
        patterns = _pattern_storage.get_all_patterns()
        for pattern_id, pattern_data in patterns.items():
            # Check if this pattern is relevant to the file type/path
            pattern_type = pattern_data.get("type", "")
            if pattern_type and pattern_type in file_path:
                return {
                    "success": True,
                    "info": f"Known pattern for type: {pattern_type}"
                }

    return {"success": True}


def run_optimized_validation(input_data: Dict[str, Any]) -> Optional[str]:
    """Run validation through optimized pipeline."""
    if not _pipeline or not hasattr(_pipeline, "execute"):
        return None

    try:
        start_time = time.time()

        # Execute pipeline
        result = _pipeline.execute(input_data)

        # Record performance metrics
        if _metrics_cache and hasattr(_metrics_cache, "record_metric"):
            _metrics_cache.record_metric({
                "operation_type": "pipeline_execution",
                "duration": getattr(result, "total_duration", 0),
                "stages_completed": len(getattr(result, "stages_completed", [])),
                "success": getattr(result, "success", True)
            })

        # Check for errors
        if not getattr(result, "success", True):
            errors = getattr(result, "errors", {})
            if errors:
                return f"Validation failed: {', '.join(errors.values())}"

        # Check for validation failures
        results_data = getattr(result, "results", {})
        validation_results = results_data.get("validate", {}).get("validation_results", {})
        for validator_name, validator_result in validation_results.items():
            if not validator_result.get("success", True):
                return validator_result.get("error", "Validation failed")

        return None

    except Exception as e:
        print(f"Error in optimized validation: {e}", file=sys.stderr)
        return str(e)


def main():
    """Main hook handler with optimized performance."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize optimization infrastructure on first run
    if OPTIMIZATION_AVAILABLE and _hook_pool is None:
        initialize_optimization_infrastructure()

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Try optimized validation first
    if OPTIMIZATION_AVAILABLE and _pipeline:
        print("⚡ Running optimized validation pipeline...", file=sys.stderr)
        error = run_optimized_validation(input_data)

        if error:
            print(f"🚨 Optimized Validation Failed:\n{error}", file=sys.stderr)
            sys.exit(2)
        else:
            print("✅ Optimized validation completed successfully", file=sys.stderr)
            sys.exit(0)

    # Fall back to enhanced validation
    if ENHANCED_VALIDATION_AVAILABLE and PreToolAnalysisManager:
        try:
            analysis_manager = PreToolAnalysisManager()
            if DebugValidationReporter:
                debug_reporter = DebugValidationReporter(analysis_manager)
                debug_reporter.log_debug_info(tool_name, tool_input)

            print("👑 Queen ZEN's Hive Intelligence: Analyzing tool usage...", file=sys.stderr)

            guidance_info = analysis_manager.validate_tool_usage(tool_name, tool_input)

            if guidance_info and GuidanceOutputHandler:
                GuidanceOutputHandler.handle_validation_guidance(guidance_info)
            else:
                print("✅ Hive Intelligence: Tool usage is optimal - proceeding", file=sys.stderr)
                if GuidanceOutputHandler:
                    GuidanceOutputHandler.handle_no_guidance()

        except Exception as e:
            print(f"Warning: Enhanced validation failed: {e}", file=sys.stderr)

    # Legacy validation as final fallback
    print("🔄 Using legacy validation system...", file=sys.stderr)

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if command:
            try:
                result = managed_subprocess_run(
                    ["npx", "claude-flow@alpha", "hooks", "pre-command",
                     "--command", command,
                     "--validate-safety", "true"],
                    check=False, capture_output=True,
                    text=True,
                    timeout=10,
                    max_memory_mb=50,  # 50MB memory limit
                    tags={"hook": "pre-command", "type": "validation"}
                )
                if result.returncode != 0:
                    print(f"🚨 Bash Command Validation Failed:\n{result.stderr}", file=sys.stderr)
                    sys.exit(2)
            except Exception as e:
                print(f"Error validating command: {e}", file=sys.stderr)
                sys.exit(2)

    print("✅ Validation completed successfully", file=sys.stderr)


def cleanup():
    """Cleanup resources on exit."""
    global _hook_pool, _async_db

    try:
        if _hook_pool and hasattr(_hook_pool, "shutdown"):
            _hook_pool.shutdown()

        if _async_db and hasattr(_async_db, "close"):
            _async_db.close()

    except Exception as e:
        print(f"Warning: Error during cleanup: {e}", file=sys.stderr)


# Register cleanup handler
import atexit

atexit.register(cleanup)


if __name__ == "__main__":
    main()
