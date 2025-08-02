#!/usr/bin/env python3
"""
Centralized Subprocess Manager for Claude Hooks

Prevents runaway processes, orphaned subprocesses, and zombie processes
by providing centralized process lifecycle management with monitoring,
resource limits, and automatic cleanup.

Enhanced with Claude-Flow integration for intelligent command suggestions.
"""

import psutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import logging
import json
import os

# Configure logging for the process manager
logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Information about a managed process."""
    pid: int
    command: List[str]
    start_time: float
    timeout: Optional[float]
    max_memory_mb: Optional[int]
    process: Union[subprocess.Popen, psutil.Process]
    cleanup_callbacks: List[Callable] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)


class ProcessManager:
    """
    Centralized manager for all subprocess operations in Claude hooks.
    
    Features:
    - Process tracking and monitoring
    - Resource limits enforcement
    - Automatic cleanup and timeout handling
    - Orphan process prevention
    - Thread-safe operations
    - Graceful shutdown handling
    """
    
    def __init__(self, max_processes: int = 50, cleanup_interval: float = 30.0):
        self.max_processes = max_processes
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe process registry
        self._processes: Dict[int, ProcessInfo] = {}
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring = True
        self._monitor_thread = None
        self._shutdown_event = threading.Event()
        
        # Process counter for unique IDs
        self._process_counter = 0
        
        # Claude-Flow integration
        self.claude_flow = ClaudeFlowIntegration()
        
        # Start monitoring thread
        self._start_monitoring()
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.shutdown_all)
    
    def _start_monitoring(self):
        """Start the background monitoring thread."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="ProcessManager-Monitor",
                daemon=True
            )
            self._monitor_thread.start()
    
    def _monitoring_loop(self):
        """Background loop for process monitoring and cleanup."""
        while self._monitoring and not self._shutdown_event.is_set():
            try:
                self._cleanup_finished_processes()
                self._check_process_limits()
                self._enforce_timeouts()
                
                # Wait for cleanup interval or shutdown
                self._shutdown_event.wait(self.cleanup_interval)
                
            except Exception as e:
                logger.exception(f"Error in process monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retry
    
    def _cleanup_finished_processes(self):
        """Remove finished processes from registry."""
        with self._lock:
            finished_pids = []
            
            for pid, info in self._processes.items():
                try:
                    if isinstance(info.process, subprocess.Popen):
                        # subprocess.Popen
                        if info.process.poll() is not None:
                            finished_pids.append(pid)
                    elif isinstance(info.process, psutil.Process):
                        # psutil.Process
                        if not info.process.is_running():
                            finished_pids.append(pid)
                    else:
                        # Unknown process type, check via psutil
                        process = psutil.Process(pid)
                        if not process.is_running():
                            finished_pids.append(pid)
                            
                except (psutil.NoSuchProcess, AttributeError, ProcessLookupError):
                    finished_pids.append(pid)
            
            for pid in finished_pids:
                self._remove_process(pid)
    
    def _check_process_limits(self):
        """Check and enforce process resource limits."""
        with self._lock:
            for pid, info in list(self._processes.items()):
                try:
                    if info.max_memory_mb:
                        process = psutil.Process(pid)
                        memory_mb = process.memory_info().rss / (1024 * 1024)
                        
                        if memory_mb > info.max_memory_mb:
                            logger.warning(
                                f"Process {pid} exceeded memory limit "
                                f"({memory_mb:.1f}MB > {info.max_memory_mb}MB). Terminating."
                            )
                            self._terminate_process(pid, reason="memory_limit")
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self._remove_process(pid)
    
    def _enforce_timeouts(self):
        """Enforce process timeouts."""
        current_time = time.time()
        
        with self._lock:
            for pid, info in list(self._processes.items()):
                if info.timeout:
                    elapsed = current_time - info.start_time
                    if elapsed > info.timeout:
                        logger.warning(
                            f"Process {pid} timed out after {elapsed:.1f}s. Terminating."
                        )
                        self._terminate_process(pid, reason="timeout")
    
    def _terminate_process(self, pid: int, reason: str = "manual"):
        """Safely terminate a process."""
        with self._lock:
            info = self._processes.get(pid)
            if not info:
                return
            
            try:
                # Run cleanup callbacks first
                for callback in info.cleanup_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.exception(f"Cleanup callback failed for PID {pid}: {e}")
                
                # Terminate the process
                if hasattr(info.process, 'terminate'):
                    # subprocess.Popen
                    info.process.terminate()
                    try:
                        info.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        info.process.kill()
                else:
                    # psutil.Process
                    process = psutil.Process(pid)
                    process.terminate()
                    
                    # Wait for graceful termination
                    try:
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        process.kill()
                
                logger.info(f"Terminated process {pid} (reason: {reason})")
                
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass  # Process already dead
            except Exception as e:
                logger.exception(f"Error terminating process {pid}: {e}")
            finally:
                self._remove_process(pid)
    
    def _remove_process(self, pid: int):
        """Remove process from registry."""
        with self._lock:
            if pid in self._processes:
                del self._processes[pid]
    
    def run(
        self,
        command: List[str],
        timeout: Optional[float] = None,
        max_memory_mb: Optional[int] = None,
        cleanup_callback: Optional[Callable] = None,
        tags: Optional[Dict[str, Any]] = None,
        suggest_claude_flow: bool = True,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Run a command with process management and claude-flow suggestions.
        
        Args:
            command: Command to execute
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            cleanup_callback: Function to call before termination
            tags: Additional metadata for the process
            suggest_claude_flow: Whether to analyze for claude-flow optimizations
            **kwargs: Additional arguments for subprocess.run()
        
        Returns:
            subprocess.CompletedProcess result
        
        Raises:
            ProcessManagerError: If limits are exceeded or startup fails
        """
        if len(self._processes) >= self.max_processes:
            raise ProcessManagerError(f"Maximum process limit ({self.max_processes}) reached")
        
        # Check for claude-flow optimization suggestions
        if suggest_claude_flow and self.claude_flow:
            optimization = self.claude_flow.analyze_command_for_optimization(command)
            if optimization:
                logger.info("🚀 Claude-Flow Optimization Available:")
                for suggestion in optimization['suggestions']:
                    logger.info(f"  • {suggestion['suggestion']}")
                    logger.info(f"    Benefit: {suggestion['benefit']}")
        
        # Set default timeout if not specified
        if timeout is None:
            timeout = kwargs.get('timeout', 60.0)
        kwargs['timeout'] = timeout
        
        # Extract check parameter to avoid duplicate keyword argument
        check_param = kwargs.pop('check', False)
        
        try:
            # Start the process
            result = subprocess.run(command, check=check_param, **kwargs)
            return result
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out: {' '.join(command)}")
            raise
        except Exception as e:
            logger.exception(f"Command failed: {' '.join(command)}: {e}")
            raise
    
    def popen(
        self,
        command: List[str],
        timeout: Optional[float] = None,
        max_memory_mb: Optional[int] = None,
        cleanup_callback: Optional[Callable] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> subprocess.Popen:
        """
        Start a managed subprocess.Popen process.
        
        Args:
            command: Command to execute
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            cleanup_callback: Function to call before termination
            tags: Additional metadata for the process
            **kwargs: Additional arguments for subprocess.Popen()
        
        Returns:
            subprocess.Popen instance
        
        Raises:
            ProcessManagerError: If limits are exceeded or startup fails
        """
        if len(self._processes) >= self.max_processes:
            raise ProcessManagerError(f"Maximum process limit ({self.max_processes}) reached")
        
        try:
            # Create the process
            process = subprocess.Popen(command, **kwargs)
            
            # Register with manager
            self._register_process(
                process=process,
                command=command,
                timeout=timeout,
                max_memory_mb=max_memory_mb,
                cleanup_callback=cleanup_callback,
                tags=tags or {}
            )
            
            return process
            
        except Exception as e:
            logger.exception(f"Failed to start process: {' '.join(command)}: {e}")
            raise ProcessManagerError(f"Failed to start process: {e}")
    
    def _register_process(
        self,
        process: subprocess.Popen,
        command: List[str],
        timeout: Optional[float] = None,
        max_memory_mb: Optional[int] = None,
        cleanup_callback: Optional[Callable] = None,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Register a process with the manager."""
        with self._lock:
            pid = process.pid
            
            callbacks = []
            if cleanup_callback:
                callbacks.append(cleanup_callback)
            
            info = ProcessInfo(
                pid=pid,
                command=command,
                start_time=time.time(),
                timeout=timeout,
                max_memory_mb=max_memory_mb,
                process=process,
                cleanup_callbacks=callbacks,
                tags=tags or {}
            )
            
            self._processes[pid] = info
            logger.debug(f"Registered process {pid}: {' '.join(command)}")
    
    @contextmanager
    def managed_process(
        self,
        command: List[str],
        timeout: Optional[float] = None,
        max_memory_mb: Optional[int] = None,
        cleanup_callback: Optional[Callable] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Context manager for subprocess.Popen with automatic cleanup.
        
        Usage:
            with manager.managed_process(['command']) as process:
                # Use process
                process.communicate()
        """
        process = None
        try:
            process = self.popen(
                command=command,
                timeout=timeout,
                max_memory_mb=max_memory_mb,
                cleanup_callback=cleanup_callback,
                tags=tags,
                **kwargs
            )
            yield process
        finally:
            if process and process.pid in self._processes:
                self.terminate_process(process.pid)
    
    def terminate_process(self, pid: int):
        """Manually terminate a managed process."""
        self._terminate_process(pid, reason="manual")
    
    def get_process_info(self, pid: int) -> Optional[ProcessInfo]:
        """Get information about a managed process."""
        with self._lock:
            return self._processes.get(pid)
    
    def list_processes(self, tags: Optional[Dict[str, Any]] = None) -> List[ProcessInfo]:
        """List all managed processes, optionally filtered by tags."""
        with self._lock:
            processes = list(self._processes.values())
            
            if tags:
                filtered = []
                for proc in processes:
                    if all(proc.tags.get(k) == v for k, v in tags.items()):
                        filtered.append(proc)
                return filtered
            
            return processes
    
    def shutdown_all(self):
        """Shutdown all managed processes and stop monitoring."""
        logger.info("Shutting down process manager...")
        
        # Stop monitoring
        self._monitoring = False
        self._shutdown_event.set()
        
        # Terminate all processes
        with self._lock:
            pids = list(self._processes.keys())
        
        for pid in pids:
            self._terminate_process(pid, reason="shutdown")
        
        # Wait for monitor thread to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logger.info("Process manager shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get process manager statistics."""
        with self._lock:
            return {
                "total_processes": len(self._processes),
                "max_processes": self.max_processes,
                "monitoring_active": self._monitoring,
                "processes": [
                    {
                        "pid": info.pid,
                        "command": " ".join(info.command),
                        "uptime": time.time() - info.start_time,
                        "timeout": info.timeout,
                        "max_memory_mb": info.max_memory_mb,
                        "tags": info.tags
                    }
                    for info in self._processes.values()
                ]
            }
    
    def run_claude_flow(
        self,
        context: Dict[str, Any],
        timeout: Optional[float] = None,
        **kwargs
    ) -> Optional[subprocess.CompletedProcess]:
        """
        Run suggested claude-flow command based on context.
        
        Args:
            context: Task context for command suggestion
            timeout: Maximum execution time
            **kwargs: Additional subprocess arguments
        
        Returns:
            CompletedProcess result or None if no suggestion
        """
        suggestion = self.claude_flow.suggest_command(context)
        if suggestion:
            command, description = suggestion
            logger.info(f"🐝 Running Claude-Flow: {description}")
            logger.info(f"  Command: {' '.join(command)}")
            
            return self.run(
                command=command,
                timeout=timeout or 300.0,  # Default 5 min for claude-flow
                suggest_claude_flow=False,  # Don't suggest on claude-flow itself
                tags={'claude_flow': True, 'context': context},
                **kwargs
            )
        
        return None
    
    def run_claude_flow_async(
        self,
        context: Dict[str, Any],
        callback: Optional[Callable[[subprocess.CompletedProcess], None]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Optional[threading.Thread]:
        """
        Run claude-flow command asynchronously with callback.
        
        Args:
            context: Task context for command suggestion
            callback: Function to call with result
            timeout: Maximum execution time
            **kwargs: Additional subprocess arguments
        
        Returns:
            Thread running the command or None
        """
        suggestion = self.claude_flow.suggest_command(context)
        if suggestion:
            def run_async():
                try:
                    result = self.run_claude_flow(context, timeout, **kwargs)
                    if callback and result:
                        callback(result)
                except Exception as e:
                    logger.exception(f"Async claude-flow execution failed: {e}")
            
            thread = threading.Thread(target=run_async, daemon=True)
            thread.start()
            return thread
        
        return None


class ProcessManagerError(Exception):
    """Exception raised by ProcessManager."""
    pass


class ClaudeFlowIntegration:
    """
    Claude-Flow command integration for intelligent workflow suggestions.
    
    Provides context-aware claude-flow command recommendations based on
    the current task and detected patterns.
    """
    
    def __init__(self):
        self.command_patterns = {
            # Swarm patterns for different task types
            "api_development": {
                "command": ["npx", "claude-flow", "swarm", "{task}", "--strategy", "development", "--max-agents", "5", "--parallel"],
                "description": "Deploy API development swarm with parallel execution"
            },
            "code_review": {
                "command": ["npx", "claude-flow", "swarm", "{task}", "--strategy", "analysis", "--read-only", "--max-agents", "3"],
                "description": "Deploy code review swarm in read-only mode"
            },
            "security_audit": {
                "command": ["npx", "claude-flow", "swarm", "{task}", "--strategy", "analysis", "--analysis", "--monitor"],
                "description": "Deploy security audit swarm with monitoring"
            },
            "performance_optimization": {
                "command": ["npx", "claude-flow", "swarm", "{task}", "--strategy", "optimization", "--parallel", "--ui"],
                "description": "Deploy performance optimization swarm with UI"
            },
            "testing": {
                "command": ["npx", "claude-flow", "swarm", "{task}", "--strategy", "testing", "--max-agents", "4"],
                "description": "Deploy testing swarm for comprehensive coverage"
            },
            
            # SPARC patterns
            "sparc_tdd": {
                "command": ["npx", "claude-flow", "sparc", "tdd", "{feature}"],
                "description": "Run complete TDD workflow using SPARC methodology"
            },
            "sparc_dev": {
                "command": ["npx", "claude-flow", "sparc", "run", "dev", "{task}"],
                "description": "Execute SPARC development mode"
            },
            
            # Memory operations
            "memory_store": {
                "command": ["npx", "claude-flow", "memory", "store", "{key}", "{value}", "--namespace", "{namespace}"],
                "description": "Store data in persistent memory"
            },
            "memory_query": {
                "command": ["npx", "claude-flow", "memory", "query", "{pattern}"],
                "description": "Search memory by pattern"
            },
            
            # Agent management
            "agent_spawn": {
                "command": ["npx", "claude-flow", "agent", "spawn", "{type}", "--name", "{name}"],
                "description": "Spawn individual agent"
            },
            
            # Hive mind operations
            "hive_mind_spawn": {
                "command": ["npx", "claude-flow", "hive-mind", "spawn", "{objective}"],
                "description": "Create intelligent swarm with hive mind"
            }
        }
    
    def suggest_command(self, context: Dict[str, Any]) -> Optional[Tuple[List[str], str]]:
        """
        Suggest appropriate claude-flow command based on context.
        
        Args:
            context: Dictionary containing task information
                - task_type: Type of task being performed
                - description: Task description
                - parameters: Additional parameters
        
        Returns:
            Tuple of (command_list, description) or None
        """
        task_type = context.get('task_type', '').lower()
        description = context.get('description', '')
        parameters = context.get('parameters', {})
        
        # Pattern matching for task types
        if 'api' in task_type or 'rest' in description.lower():
            pattern = self.command_patterns['api_development']
        elif 'review' in task_type or 'audit' in task_type:
            if 'security' in description.lower():
                pattern = self.command_patterns['security_audit']
            else:
                pattern = self.command_patterns['code_review']
        elif 'performance' in task_type or 'optimize' in task_type:
            pattern = self.command_patterns['performance_optimization']
        elif 'test' in task_type:
            pattern = self.command_patterns['testing']
        elif 'sparc' in task_type:
            if 'tdd' in description.lower():
                pattern = self.command_patterns['sparc_tdd']
            else:
                pattern = self.command_patterns['sparc_dev']
        elif 'memory' in task_type:
            if 'store' in description.lower():
                pattern = self.command_patterns['memory_store']
            else:
                pattern = self.command_patterns['memory_query']
        elif 'hive' in task_type:
            pattern = self.command_patterns['hive_mind_spawn']
        else:
            return None
        
        # Build command with parameters
        command = []
        for part in pattern['command']:
            if part.startswith('{') and part.endswith('}'):
                key = part[1:-1]
                value = parameters.get(key, description)
                command.append(value)
            else:
                command.append(part)
        
        return (command, pattern['description'])
    
    def analyze_command_for_optimization(self, command: List[str]) -> Optional[Dict[str, Any]]:
        """
        Analyze a command and suggest claude-flow optimizations.
        
        Args:
            command: Command list to analyze
        
        Returns:
            Dictionary with optimization suggestions or None
        """
        cmd_str = ' '.join(command).lower()
        
        suggestions = []
        
        # Check for common patterns that could benefit from claude-flow
        if 'npm test' in cmd_str or 'pytest' in cmd_str:
            suggestions.append({
                'pattern': 'testing',
                'suggestion': 'Consider using: npx claude-flow swarm "Run comprehensive tests" --strategy testing',
                'benefit': 'Parallel test execution with intelligent agent coordination'
            })
        
        if 'eslint' in cmd_str or 'prettier' in cmd_str or 'lint' in cmd_str:
            suggestions.append({
                'pattern': 'code_quality',
                'suggestion': 'Consider using: npx claude-flow swarm "Code quality analysis" --strategy analysis --read-only',
                'benefit': 'Comprehensive code quality analysis with multiple perspectives'
            })
        
        if 'build' in cmd_str and 'claude-flow' not in cmd_str:
            suggestions.append({
                'pattern': 'build',
                'suggestion': 'Consider using: npx claude-flow sparc run dev "Build optimization"',
                'benefit': 'SPARC methodology for systematic build process improvement'
            })
        
        if suggestions:
            return {
                'original_command': command,
                'suggestions': suggestions,
                'recommendation': 'Claude-Flow can enhance this operation with intelligent agent coordination'
            }
        
        return None
    
    def get_swarm_status_command(self) -> List[str]:
        """Get command to check swarm status."""
        return ["npx", "claude-flow", "status"]
    
    def get_agent_list_command(self) -> List[str]:
        """Get command to list active agents."""
        return ["npx", "claude-flow", "agent", "list", "--json"]


# Global instance for use across hooks
_global_manager: Optional[ProcessManager] = None
_manager_lock = threading.Lock()


def get_process_manager() -> ProcessManager:
    """Get the global process manager instance."""
    global _global_manager
    
    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                _global_manager = ProcessManager()
    
    return _global_manager


def managed_subprocess_run(
    command: List[str],
    timeout: Optional[float] = None,
    max_memory_mb: Optional[int] = None,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Drop-in replacement for subprocess.run() with process management.
    
    Usage:
        result = managed_subprocess_run(['ls', '-la'], timeout=10)
    """
    manager = get_process_manager()
    return manager.run(
        command=command,
        timeout=timeout,
        max_memory_mb=max_memory_mb,
        **kwargs
    )


def managed_subprocess_popen(
    command: List[str],
    timeout: Optional[float] = None,
    max_memory_mb: Optional[int] = None,
    cleanup_callback: Optional[Callable] = None,
    **kwargs
) -> subprocess.Popen:
    """
    Drop-in replacement for subprocess.Popen() with process management.
    
    Usage:
        process = managed_subprocess_popen(['long-running-command'])
    """
    manager = get_process_manager()
    return manager.popen(
        command=command,
        timeout=timeout,
        max_memory_mb=max_memory_mb,
        cleanup_callback=cleanup_callback,
        **kwargs
    )


@contextmanager
def managed_process_context(
    command: List[str],
    timeout: Optional[float] = None,
    max_memory_mb: Optional[int] = None,
    **kwargs
):
    """
    Context manager for managed processes.
    
    Usage:
        with managed_process_context(['command']) as process:
            stdout, stderr = process.communicate()
    """
    manager = get_process_manager()
    with manager.managed_process(
        command=command,
        timeout=timeout,
        max_memory_mb=max_memory_mb,
        **kwargs
    ) as process:
        yield process


# Emergency cleanup function
def emergency_cleanup():
    """Emergency cleanup of all managed processes."""
    global _global_manager
    if _global_manager:
        _global_manager.shutdown_all()


# Claude-Flow integration helpers
def run_claude_flow_swarm(
    task: str,
    strategy: str = "development",
    max_agents: int = 5,
    parallel: bool = True,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a claude-flow swarm for a specific task.
    
    Args:
        task: Task description
        strategy: Swarm strategy (development, analysis, testing, etc.)
        max_agents: Maximum number of agents
        parallel: Enable parallel execution
        **kwargs: Additional subprocess arguments
    
    Returns:
        CompletedProcess result
    """
    manager = get_process_manager()
    context = {
        'task_type': f'{strategy}_swarm',
        'description': task,
        'parameters': {
            'task': task,
            'strategy': strategy,
            'max_agents': str(max_agents)
        }
    }
    
    result = manager.run_claude_flow(context, **kwargs)
    if result is None:
        # Fallback to direct command
        command = ["npx", "claude-flow", "swarm", task, "--strategy", strategy, 
                   "--max-agents", str(max_agents)]
        if parallel:
            command.append("--parallel")
        
        result = manager.run(command, timeout=300.0, **kwargs)
    
    return result


def run_claude_flow_sparc(
    mode: str,
    task: str,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a claude-flow SPARC mode.
    
    Args:
        mode: SPARC mode (tdd, dev, etc.)
        task: Task or feature description
        **kwargs: Additional subprocess arguments
    
    Returns:
        CompletedProcess result
    """
    manager = get_process_manager()
    
    if mode == "tdd":
        command = ["npx", "claude-flow", "sparc", "tdd", task]
    else:
        command = ["npx", "claude-flow", "sparc", "run", mode, task]
    
    return manager.run(command, timeout=300.0, **kwargs)


def suggest_claude_flow_for_command(command: List[str]) -> Optional[Dict[str, Any]]:
    """
    Get claude-flow optimization suggestions for a command.
    
    Args:
        command: Command to analyze
    
    Returns:
        Optimization suggestions or None
    """
    manager = get_process_manager()
    return manager.claude_flow.analyze_command_for_optimization(command)


def get_claude_flow_status() -> subprocess.CompletedProcess:
    """Get current claude-flow status."""
    manager = get_process_manager()
    return manager.run(["npx", "claude-flow", "status"], timeout=30.0)


def list_claude_flow_agents() -> subprocess.CompletedProcess:
    """List all active claude-flow agents."""
    manager = get_process_manager()
    return manager.run(["npx", "claude-flow", "agent", "list", "--json"], timeout=30.0)