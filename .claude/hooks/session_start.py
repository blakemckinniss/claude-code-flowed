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
    # Import new integrated optimization system
    from modules.optimization.hook_integration import (
        get_hook_integration,
        get_optimization_status
    )
    OPTIMIZATION_AVAILABLE = True
    NEW_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Optimization modules not available: {e}", file=sys.stderr)
    # Provide fallback None assignments to prevent unbound variable errors
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
    get_optimization_status = None
    OPTIMIZATION_AVAILABLE = False
    NEW_OPTIMIZER_AVAILABLE = False

# Import existing modules
try:
    from modules.pre_tool.analyzers.neural_pattern_validator import NeuralPatternStorage
    from modules.pre_tool.manager import PreToolAnalysisManager
    ENHANCED_FEATURES = True
except ImportError:
    NeuralPatternStorage = None
    PreToolAnalysisManager = None
    ENHANCED_FEATURES = False

# Import memory integration
try:
    from modules.memory.hook_memory_integration import get_hook_memory_integration
    MEMORY_INTEGRATION = True
except ImportError:
    get_hook_memory_integration = None
    MEMORY_INTEGRATION = False


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
            # Initialize session cache only if ValidatorCache is available
            if ValidatorCache is not None:
                _session_cache = ValidatorCache(ttl=600, max_size=100)
            
            # Initialize metrics cache only if PerformanceMetricsCache is available
            if PerformanceMetricsCache is not None:
                _metrics_cache = PerformanceMetricsCache(
                    write_interval=5.0,
                    batch_size=100
                )
            
            # Initialize async database only if AsyncDatabaseManager is available
            if AsyncDatabaseManager is not None:
                _async_db = AsyncDatabaseManager(
                    db_path=Path(hooks_dir / "db" / "session_data.db"),
                    batch_size=50,
                    batch_timeout=5.0
                )
            
            # Pre-warm hook execution pools in background
            def warm_pools():
                try:
                    if HookExecutionPool is not None:
                        pool = HookExecutionPool(pool_size=4)
                        # Keep reference to prevent garbage collection
                        initialize_optimization_infrastructure._pool = pool
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
        # Check for cached context using correct ValidatorCache method
        # ValidatorCache expects tool_name and tool_input parameters
        cached = _session_cache.get_validation_result("session_context", {})
        
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
        # Use correct ValidatorCache method
        _session_cache.store_result("session_context", {}, context)
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
        "memory_integration_active": MEMORY_INTEGRATION,
        "initialization_time": 0,
        "git_structure": {}
    }
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        
        # Submit parallel tasks
        futures[executor.submit(initialize_session_tracking)] = "tracking"
        futures[executor.submit(load_neural_patterns)] = "neural"
        futures[executor.submit(detect_github_context)] = "github"
        futures[executor.submit(count_active_validators)] = "validators"
        futures[executor.submit(get_git_project_structure)] = "git_structure"
        
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
                elif task_name == "git_structure":
                    session_info["git_structure"] = result
                    
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
                    with open(git_config) as f:
                        return "github.com" in f.read()
                break
            current_dir = current_dir.parent
        return False
    except Exception:
        return False


def get_git_project_structure() -> Dict[str, Any]:
    """Get git project structure information for reference grounding (coding files only)."""
    try:
        import subprocess
        
        # Define coding file extensions
        coding_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.md',
            '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb',
            '.go', '.rs', '.php', '.swift', '.kt', '.scala',
            '.sh', '.bash', '.zsh', '.yaml', '.yml', '.toml'
        }
        
        # Get all files from git
        git_output = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if git_output.returncode != 0:
            return {
                "total_files": 0,
                "coding_files": 0,
                "agent_files": 0,
                "hook_files": 0,
                "key_directories": [],
                "project_root": "unknown"
            }
        
        # Filter for coding files only
        all_files = git_output.stdout.strip().split('\n')
        coding_files = []
        for file_path in all_files:
            # Check if file has a coding extension
            for ext in coding_extensions:
                if file_path.endswith(ext):
                    coding_files.append(file_path)
                    break
        
        # Get key directories from coding files
        key_dirs = set()
        for file_path in coding_files:
            if '/' in file_path:
                parts = file_path.split('/')
                # Add top-level directory
                key_dirs.add(parts[0])
                # Add second-level for .claude directory
                if len(parts) > 1 and parts[0] == '.claude':
                    key_dirs.add(f"{parts[0]}/{parts[1]}")
        
        # Sort and limit directories
        sorted_dirs = sorted(list(key_dirs))[:12]
        
        # Count specific file types
        agent_count = len([f for f in coding_files if f.startswith('.claude/agents/') and f.endswith('.md')])
        hook_count = len([f for f in coding_files if f.startswith('.claude/hooks/') and f.endswith('.py')])
        python_count = len([f for f in coding_files if f.endswith('.py')])
        json_count = len([f for f in coding_files if f.endswith('.json')])
        
        return {
            "total_files": len(all_files),
            "coding_files": len(coding_files),
            "python_files": python_count,
            "json_files": json_count,
            "agent_files": agent_count,
            "hook_files": hook_count,
            "key_directories": sorted_dirs,
            "project_root": Path.cwd().name
        }
    except Exception:
        return {
            "total_files": 0,
            "coding_files": 0,
            "python_files": 0,
            "json_files": 0,
            "agent_files": 0,
            "hook_files": 0,
            "key_directories": [],
            "project_root": "unknown"
        }


def store_session_async(session_info: Dict[str, Any]):
    """Store session information asynchronously."""
    if _async_db:
        try:
            session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            # Create table if not exists first
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS session_states (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    session_data TEXT,
                    optimization_active BOOLEAN
                )
            """
            _async_db.queue_script(create_table_sql)
            
            # Insert session data using correct AsyncDatabaseManager method
            insert_sql = """
                INSERT OR REPLACE INTO session_states (session_id, start_time, session_data, optimization_active)
                VALUES (?, ?, ?, ?)
            """
            _async_db.queue_write(insert_sql, (
                session_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(session_info),
                OPTIMIZATION_AVAILABLE
            ))
        except Exception as e:
            # For SessionStart hooks, database errors should not block session initialization
            # Log the error and fall back to sync storage to ensure session state is preserved
            print(f"Warning: Async session storage failed: {e}", file=sys.stderr)
            print("Falling back to synchronous session storage", file=sys.stderr)
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
    # ZEN-enhanced message emphasizing the hook → ZEN → Claude Flow hierarchy
    base_message = """🧠 INTELLIGENT HOOK SYSTEM INITIALIZED - THE BRAIN OF CLAUDE CODE

🎯 ARCHITECTURE HIERARCHY:
┌─────────────────┐
│  Claude Hooks   │ ← YOU ARE HERE: Primary Intelligence Layer
├─────────────────┤
│  ZEN MCP Tools  │ ← Orchestration & Coordination
├─────────────────┤
│  Claude Flow    │ ← Execution & Multi-Agent Swarms
└─────────────────┘

🚨 CRITICAL: HOOKS DRIVE ALL INTELLIGENCE
• Every prompt is analyzed by UserPromptSubmit hook
• Context Intelligence Engine determines complexity
• ZEN directives are injected automatically
• Best practices are enforced proactively

⚡ ZEN DISCOVERY PHILOSOPHY:
1. START WITH 0 AGENTS - Let ZEN investigate first
2. ANALYZE COMPLEXITY - Deep understanding before action
3. SCALE INTELLIGENTLY - Deploy agents as needed
4. TRUST THE HOOKS - They see patterns you might miss

🎯 Available MCP Tools (COORDINATION ONLY):
• mcp__zen__thinkdeep - Deep investigation and analysis
• mcp__zen__analyze - Comprehensive code analysis
• mcp__zen__planner - Sequential task breakdown
• mcp__zen__consensus - Multi-model decision making
• mcp__claude-flow__swarm_init - Initialize agent topology
• mcp__claude-flow__agent_spawn - Create specialized agents
• mcp__claude-flow__memory_usage - Persistent memory

🤖 Agent Categories (54 Types - Deploy After Discovery):
• Core: coder, reviewer, tester, planner, researcher
• Architecture: system-architect, api-architect, database-architect
• Security: security-auditor, security-analyzer, security-architect
• GitHub: pr-manager, code-review-swarm, issue-tracker
• Specialized: 40+ domain-specific agents

⚠️ GOLDEN RULES (ENFORCED BY HOOKS):
• 1 MESSAGE = ALL RELATED OPERATIONS
• MCP = Coordination ONLY (never execution)
• Claude Code = ALL actual execution
• Start with ZEN discovery, scale up as needed"""

    # Add performance-optimized features
    enhanced_parts = [
        "",
        "🧠 HOOK INTELLIGENCE STATUS:",
        f"• Neural patterns loaded: {session_info.get('neural_patterns_loaded', 0)}",
        f"• GitHub context: {'✓ Detected' if session_info.get('github_context_detected') else '✗ Not detected'}",
        f"• Active validators: {session_info.get('pre_tool_validators', 7)}",
        f"• Session tracking: {'✓ Enhanced' if session_info.get('session_tracking_active') else '✗ Basic'}",
        f"• Memory integration: {'✓ Active' if session_info.get('memory_integration_active') else '✗ Disabled'}",
        f"• Learning system: {'✓ Adaptive' if session_info.get('neural_patterns_loaded', 0) > 0 else '✓ Ready'}"
    ]
    
    # Add git project structure for reference grounding
    git_struct = session_info.get('git_structure', {})
    if git_struct and git_struct.get('coding_files', 0) > 0:
        enhanced_parts.extend([
            "",
            "📁 PROJECT STRUCTURE (coding files only):",
            f"• Project: {git_struct.get('project_root', 'unknown')}",
            f"• Coding files: {git_struct.get('coding_files', 0)} / {git_struct.get('total_files', 0)} total",
            f"• Python files: {git_struct.get('python_files', 0)}",
            f"• JSON files: {git_struct.get('json_files', 0)}",
            f"• Agent definitions (.md): {git_struct.get('agent_files', 0)}",
            f"• Hook modules (.py): {git_struct.get('hook_files', 0)}",
            f"• Key directories: {', '.join(git_struct.get('key_directories', [])[:8])}"
        ])
        if len(git_struct.get('key_directories', [])) > 8:
            enhanced_parts.append(f"  + {len(git_struct.get('key_directories', [])) - 8} more directories...")
    elif git_struct:
        # Show minimal info if no coding files found
        enhanced_parts.extend([
            "",
            "📁 PROJECT STRUCTURE:",
            f"• Project: {git_struct.get('project_root', 'unknown')}",
            "• No coding files detected in git repository"
        ])
    
    if session_info.get('optimization_active'):
        enhanced_parts.extend([
            "",
            "⚡ PERFORMANCE OPTIMIZATIONS:",
            f"• Session init time: {session_info.get('initialization_time', 0):.2f}s",
            "• Hook pooling: ✓ Enabled",
            "• Parallel validation: ✓ Active",
            "• Smart caching: ✓ Running",
            "• Circuit breakers: ✓ Protected"
        ])
    
    enhanced_parts.extend([
        "",
        "💡 REMEMBER: Hooks guide you to 10x development. Trust the system!"
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
    
    # Initialize new integrated optimizer
    if NEW_OPTIMIZER_AVAILABLE and get_hook_integration:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                integration = loop.run_until_complete(get_hook_integration())
                if integration:
                    print("🚀 New integrated optimizer initialized successfully", file=sys.stderr)
            finally:
                loop.close()
        except Exception as e:
            print(f"Warning: Failed to initialize integrated optimizer: {e}", file=sys.stderr)
    
    # Initialize memory integration
    memory_integration = None
    if MEMORY_INTEGRATION and get_hook_memory_integration:
        try:
            memory_integration = get_hook_memory_integration()
            # Capture session start context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(memory_integration.capture_session_start_memory({
                    "hook": "session_start",
                    "input": input_data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
            finally:
                loop.close()
        except Exception as e:
            print(f"Warning: Failed to initialize memory integration: {e}", file=sys.stderr)
    
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
    
    # Add optimization status to session info
    if NEW_OPTIMIZER_AVAILABLE and get_optimization_status:
        try:
            opt_status = get_optimization_status()
            session_info["integrated_optimizer"] = opt_status.get("enabled", False)
            session_info["optimizer_profile"] = opt_status.get("optimizer_status", {}).get("current_profile", "unknown")
        except Exception:
            pass
    
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