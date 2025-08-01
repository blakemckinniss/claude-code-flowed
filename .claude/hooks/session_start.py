#!/usr/bin/env python3
"""Optimized Session Start Hook with performance enhancements.

Integrates optimization modules for fast session initialization:
- Pre-warmed hook execution pool
- Cached session context
- Asynchronous session data loading
- Parallel validator initialization
- Performance metrics initialization
"""

import sys
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Define hooks directory for local usage
hooks_dir = Path(__file__).parent

# Import optimization modules
try:
    from modules.optimization import (
        HookExecutionPool,
        ValidatorCache,
        PerformanceMetricsCache,
        AsyncDatabaseManager,
        ParallelValidationManager,
        HookCircuitBreaker,
        ContextTracker,
        BoundedPatternStorage,
        HookPipeline
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Optimization modules not available: {e}", file=sys.stderr)
    OPTIMIZATION_AVAILABLE = False

# Import existing modules
try:
    from modules.pre_tool.analyzers.neural_pattern_validator import NeuralPatternStorage
    from modules.pre_tool.manager import PreToolAnalysisManager
    ENHANCED_FEATURES = True
except ImportError:
    NeuralPatternStorage = None
    PreToolAnalysisManager = None
    ENHANCED_FEATURES = False


# Global optimization infrastructure initialization
_initialization_complete = False
_initialization_lock = threading.Lock()
_session_cache = None
_metrics_cache = None
_async_db = None


def initialize_optimization_infrastructure():
    """Initialize global optimization infrastructure once."""
    global _initialization_complete, _session_cache, _metrics_cache, _async_db
    
    with _initialization_lock:
        if _initialization_complete:
            return
        
        if not OPTIMIZATION_AVAILABLE:
            _initialization_complete = True
            return
        
        try:
            # Initialize session cache
            _session_cache = ValidatorCache(ttl=600, max_size=100)
            
            # Initialize metrics cache
            _metrics_cache = PerformanceMetricsCache(
                write_interval=5.0,
                batch_size=100
            )
            
            # Initialize async database
            _async_db = AsyncDatabaseManager(
                db_path=Path(hooks_dir / "db" / "session_data.db"),
                batch_size=50,
                batch_timeout=5.0
            )
            
            # Pre-warm hook execution pools in background
            def warm_pools():
                try:
                    pool = HookExecutionPool(pool_size=4)
                    # Keep reference to prevent garbage collection
                    setattr(initialize_optimization_infrastructure, '_pool', pool)
                except Exception:
                    pass
            
            warm_thread = threading.Thread(target=warm_pools, daemon=True)
            warm_thread.start()
            
            _initialization_complete = True
            print("⚡ Session optimization infrastructure initialized", file=sys.stderr)
            
        except Exception as e:
            print(f"Warning: Failed to initialize optimization: {e}", file=sys.stderr)
            _initialization_complete = True


def get_cached_session_context() -> Optional[Dict[str, Any]]:
    """Get cached session context if available."""
    if not _session_cache:
        return None
    
    try:
        # Check for cached context
        cache_key = "session_context_enhanced"
        cached = _session_cache.get(cache_key)
        
        if cached and time.time() - cached.get('timestamp', 0) < 300:  # 5 min cache
            return cached
        
        return None
    except Exception:
        return None


def cache_session_context(context: Dict[str, Any]):
    """Cache session context for fast startup."""
    if not _session_cache:
        return
    
    try:
        context['timestamp'] = time.time()
        _session_cache.set("session_context_enhanced", context)
    except Exception:
        pass


def parallel_session_initialization() -> Dict[str, Any]:
    """Initialize session components in parallel."""
    session_info = {
        "neural_patterns_loaded": 0,
        "github_context_detected": False,
        "pre_tool_validators": 0,
        "session_tracking_active": False,
        "optimization_active": OPTIMIZATION_AVAILABLE,
        "initialization_time": 0
    }
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        
        # Submit parallel tasks
        futures[executor.submit(initialize_session_tracking)] = "tracking"
        futures[executor.submit(load_neural_patterns)] = "neural"
        futures[executor.submit(detect_github_context)] = "github"
        futures[executor.submit(count_active_validators)] = "validators"
        
        # Collect results
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                result = future.result(timeout=2.0)
                
                if task_name == "tracking":
                    session_info["session_tracking_active"] = result
                elif task_name == "neural":
                    session_info["neural_patterns_loaded"] = result
                elif task_name == "github":
                    session_info["github_context_detected"] = result
                elif task_name == "validators":
                    session_info["pre_tool_validators"] = result
                    
            except Exception as e:
                print(f"Warning: {task_name} initialization failed: {e}", file=sys.stderr)
    
    session_info["initialization_time"] = time.time() - start_time
    
    # Record metrics
    if _metrics_cache:
        _metrics_cache.record_metric({
            "operation_type": "session_start",
            "duration": session_info["initialization_time"],
            "components_loaded": sum(1 for v in session_info.values() if v),
            "optimization_active": OPTIMIZATION_AVAILABLE
        })
    
    return session_info


def initialize_session_tracking() -> bool:
    """Initialize session tracking directory."""
    try:
        session_dir = hooks_dir / ".session"
        session_dir.mkdir(exist_ok=True)
        return True
    except Exception:
        return False


def load_neural_patterns() -> int:
    """Load neural patterns if available."""
    if not ENHANCED_FEATURES or NeuralPatternStorage is None:
        return 0
    
    try:
        neural_storage = NeuralPatternStorage()
        recent_patterns = neural_storage.get_recent_patterns(limit=10)
        return len(recent_patterns) if recent_patterns else 0
    except Exception:
        return 0


def count_active_validators() -> int:
    """Count active validators if available."""
    if not ENHANCED_FEATURES or PreToolAnalysisManager is None:
        return 0
    
    try:
        manager = PreToolAnalysisManager()
        return len(manager.validators)
    except Exception:
        return 0


def detect_github_context() -> bool:
    """Detect if we're in a GitHub repository context."""
    try:
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:
            if (current_dir / ".git").exists():
                git_config = current_dir / ".git" / "config"
                if git_config.exists():
                    with open(git_config, 'r') as f:
                        return "github.com" in f.read()
                break
            current_dir = current_dir.parent
        return False
    except Exception:
        return False


def store_session_async(session_info: Dict[str, Any]):
    """Store session information asynchronously."""
    if _async_db:
        try:
            session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            _async_db.write({
                "session_id": session_id,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "session_data": json.dumps(session_info),
                "optimization_active": OPTIMIZATION_AVAILABLE
            })
        except Exception:
            pass
    else:
        # Fallback to sync storage
        store_session_start_sync(session_info)


def store_session_start_sync(session_info: Dict[str, Any]):
    """Store session start information synchronously (fallback)."""
    try:
        session_db_path = hooks_dir / ".session" / "session_state.db"
        session_db_path.parent.mkdir(exist_ok=True)
        
        session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        with sqlite3.connect(session_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_states (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    session_data TEXT
                )
            """)
            
            conn.execute("""
                INSERT OR REPLACE INTO session_states (session_id, start_time, session_data)
                VALUES (?, ?, ?)
            """, (
                session_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(session_info)
            ))
    except Exception:
        pass


def get_optimized_context_message(session_info: Dict[str, Any]) -> str:
    """Get optimized context message with performance stats."""
    base_message = """🚀 MCP ZEN Orchestration Context Loaded

⚡ CRITICAL CONCURRENT EXECUTION RULES:
1. GOLDEN RULE: 1 MESSAGE = ALL RELATED OPERATIONS
2. ALWAYS batch TodoWrite operations (5-10+ todos in ONE call)
3. ALWAYS spawn Task agents concurrently in ONE message
4. ALWAYS batch file operations (Read/Write/Edit) together
5. ALWAYS group bash commands in ONE message

🎯 Available MCP Tools for Coordination:
• mcp__claude-flow__swarm_init - Initialize swarm topology (mesh/hierarchical/ring/star)
• mcp__claude-flow__agent_spawn - Create specialized agents (54 types available)
• mcp__claude-flow__task_orchestrate - Break down complex tasks
• mcp__claude-flow__memory_usage - Persistent memory across sessions
• mcp__claude-flow__neural_train - Improve coordination patterns

🤖 Key Agent Categories (54 Total):
• Core: coder, reviewer, tester, planner, researcher
• Swarm: hierarchical-coordinator, mesh-coordinator, adaptive-coordinator
• Consensus: byzantine-coordinator, raft-manager, gossip-coordinator
• GitHub: pr-manager, code-review-swarm, issue-tracker, release-manager
• SPARC: sparc-coord, sparc-coder, specification, architecture
• Specialized: backend-dev, mobile-dev, ml-developer, api-docs

⚠️ REMEMBER:
• MCP tools = Coordination only
• Claude Code = All actual execution
• Never split related operations across messages
• Always include coordination hooks in Task agent instructions"""

    # Add performance-optimized features
    enhanced_parts = [
        "",
        "🧠 ENHANCED HIVE INTELLIGENCE ACTIVE:",
        f"• Neural patterns from previous sessions: {session_info.get('neural_patterns_loaded', 0)}",
        f"• GitHub repository context: {'Detected' if session_info.get('github_context_detected') else 'None'}",
        f"• Pre-tool validators active: {session_info.get('pre_tool_validators', 7)}",
        f"• Advanced session tracking: {'Enabled' if session_info.get('session_tracking_active') else 'Basic'}"
    ]
    
    if session_info.get('optimization_active'):
        enhanced_parts.extend([
            "",
            "⚡ PERFORMANCE OPTIMIZATIONS ACTIVE:",
            f"• Session initialization: {session_info.get('initialization_time', 0):.2f}s",
            "• Hook execution pooling: Enabled",
            "• Parallel validation: Enabled",
            "• Smart caching: Active",
            "• Async operations: Running"
        ])
    
    return base_message + "\n".join(enhanced_parts)


def main():
    """Main hook handler with optimized session initialization."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize optimization infrastructure
    if OPTIMIZATION_AVAILABLE:
        initialize_optimization_infrastructure()
    
    # Check for cached session context first
    cached_context = get_cached_session_context()
    
    if cached_context:
        session_info = cached_context
        print("⚡ Using cached session context for fast startup", file=sys.stderr)
    else:
        # Initialize session components in parallel
        session_info = parallel_session_initialization()
        
        # Cache the context for next time
        cache_session_context(session_info)
    
    # Store session information asynchronously
    store_session_async(session_info)
    
    # Create the JSON output with optimized context
    context_message = get_optimized_context_message(session_info)
    
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context_message
        }
    }
    
    # Output the JSON response
    print(json.dumps(output))
    sys.exit(0)


def cleanup():
    """Cleanup optimization resources."""
    global _async_db
    
    if _async_db:
        _async_db.shutdown()
    
    if _metrics_cache:
        # Metrics cache flushes automatically
        pass


# Register cleanup
import atexit
atexit.register(cleanup)


if __name__ == "__main__":
    main()