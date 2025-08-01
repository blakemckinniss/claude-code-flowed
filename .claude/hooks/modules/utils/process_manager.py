#!/usr/bin/env python3
"""
Centralized Subprocess Manager for Claude Hooks

Prevents runaway processes, orphaned subprocesses, and zombie processes
by providing centralized process lifecycle management with monitoring,
resource limits, and automatic cleanup.
"""

import psutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
import logging

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
                logger.error(f"Error in process monitoring loop: {e}")
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
                        logger.error(f"Cleanup callback failed for PID {pid}: {e}")
                
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
                logger.error(f"Error terminating process {pid}: {e}")
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
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Run a command with process management.
        
        Args:
            command: Command to execute
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            cleanup_callback: Function to call before termination
            tags: Additional metadata for the process
            **kwargs: Additional arguments for subprocess.run()
        
        Returns:
            subprocess.CompletedProcess result
        
        Raises:
            ProcessManagerError: If limits are exceeded or startup fails
        """
        if len(self._processes) >= self.max_processes:
            raise ProcessManagerError(f"Maximum process limit ({self.max_processes}) reached")
        
        # Set default timeout if not specified
        if timeout is None:
            timeout = kwargs.get('timeout', 60.0)
        kwargs['timeout'] = timeout
        
        try:
            # Start the process
            result = subprocess.run(command, **kwargs)
            return result
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out: {' '.join(command)}")
            raise
        except Exception as e:
            logger.error(f"Command failed: {' '.join(command)}: {e}")
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
            logger.error(f"Failed to start process: {' '.join(command)}: {e}")
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


class ProcessManagerError(Exception):
    """Exception raised by ProcessManager."""
    pass


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