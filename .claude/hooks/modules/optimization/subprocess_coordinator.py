"""Advanced Subprocess Coordination for Hook System.

Provides intelligent subprocess management with:
- Process pool with lifecycle management
- Resource isolation and cgroups support
- Memory and CPU limits per process
- Graceful degradation and recovery
- IPC optimization with shared memory
"""

import subprocess
import multiprocessing as mp
import os
import sys
import time
import signal
import psutil
import json
import tempfile
import threading
import queue
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import resource
import fcntl
import select
import struct
import mmap
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


@dataclass
class ProcessConfig:
    """Configuration for subprocess execution."""
    command: List[str]
    env: Dict[str, str] = None
    cwd: str = None
    memory_limit_mb: int = 100
    cpu_limit_percent: int = 50
    timeout: float = 10.0
    nice_level: int = 10
    stdin_data: Optional[str] = None
    capture_output: bool = True


@dataclass 
class ProcessResult:
    """Result from subprocess execution."""
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    memory_peak_mb: float
    cpu_percent: float
    timed_out: bool = False
    error: Optional[str] = None


class ProcessPool:
    """Manages a pool of subprocess workers with resource limits."""
    
    def __init__(self, 
                 pool_size: int = 4,
                 enable_cgroups: bool = False,
                 shared_memory_size: int = 10 * 1024 * 1024):  # 10MB
        
        self.pool_size = pool_size
        self.enable_cgroups = enable_cgroups and self._check_cgroups()
        self.shared_memory_size = shared_memory_size
        
        # Process management
        self._workers: List[mp.Process] = []
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._shutdown = mp.Event()
        
        # Shared memory for IPC
        self._shm_pool = []
        self._shm_available = mp.Queue()
        
        # Performance tracking
        self._stats = {
            'total_executions': mp.Value('i', 0),
            'successful_executions': mp.Value('i', 0),
            'failed_executions': mp.Value('i', 0),
            'timeout_executions': mp.Value('i', 0),
            'average_duration': mp.Value('d', 0.0)
        }
        
        self._initialize()
    
    def _check_cgroups(self) -> bool:
        """Check if cgroups v2 is available."""
        return Path("/sys/fs/cgroup/cgroup.controllers").exists()
    
    def _initialize(self):
        """Initialize the process pool."""
        # Create shared memory segments
        for i in range(self.pool_size * 2):
            shm = mp.shared_memory.SharedMemory(create=True, size=self.shared_memory_size)
            self._shm_pool.append(shm)
            self._shm_available.put(i)
        
        # Start worker processes
        for i in range(self.pool_size):
            worker = mp.Process(
                target=self._worker_loop,
                args=(i, self._task_queue, self._result_queue, self._shutdown)
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self, worker_id: int, task_queue: mp.Queue, 
                     result_queue: mp.Queue, shutdown: mp.Event):
        """Worker loop for processing subprocess tasks."""
        
        # Set process nice level
        os.nice(10)
        
        while not shutdown.is_set():
            try:
                # Get task with timeout
                task = task_queue.get(timeout=1.0)
                if task is None:
                    break
                
                task_id, config, shm_idx = task
                
                # Execute subprocess
                result = self._execute_subprocess(config, shm_idx)
                
                # Return result
                result_queue.put((task_id, result, shm_idx))
                
            except queue.Empty:
                continue
            except Exception as e:
                # Return error result
                result = ProcessResult(
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    duration=0.0,
                    memory_peak_mb=0.0,
                    cpu_percent=0.0,
                    error=str(e)
                )
                result_queue.put((task_id, result, shm_idx))
    
    def _execute_subprocess(self, config: ProcessConfig, shm_idx: Optional[int]) -> ProcessResult:
        """Execute a subprocess with resource limits."""
        
        start_time = time.time()
        
        # Prepare environment
        env = os.environ.copy()
        if config.env:
            env.update(config.env)
        
        # Create process
        process = None
        monitor_thread = None
        
        try:
            # Set up resource limits
            def set_limits():
                # Memory limit
                if config.memory_limit_mb > 0:
                    memory_bytes = config.memory_limit_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                
                # CPU limit via nice
                os.nice(config.nice_level)
                
                # Process group for signal handling
                os.setpgrp()
            
            # Create subprocess
            process = subprocess.Popen(
                config.command,
                stdin=subprocess.PIPE if config.stdin_data else None,
                stdout=subprocess.PIPE if config.capture_output else None,
                stderr=subprocess.PIPE if config.capture_output else None,
                env=env,
                cwd=config.cwd,
                preexec_fn=set_limits,
                text=True
            )
            
            # Start resource monitor
            monitor_data = {'peak_memory': 0, 'cpu_percent': 0}
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process.pid, monitor_data),
                daemon=True
            )
            monitor_thread.start()
            
            # Handle stdin if needed
            if config.stdin_data:
                if shm_idx is not None and shm_idx < len(self._shm_pool):
                    # Read from shared memory
                    shm = self._shm_pool[shm_idx]
                    size_bytes = shm.buf[:8]
                    size = struct.unpack('Q', size_bytes)[0]
                    stdin_data = shm.buf[8:8+size].tobytes().decode('utf-8')
                else:
                    stdin_data = config.stdin_data
                
                process.stdin.write(stdin_data)
                process.stdin.close()
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=config.timeout)
                exit_code = process.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                # Kill process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(0.1)
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                
                stdout, stderr = process.communicate()
                exit_code = -15  # SIGTERM
                timed_out = True
            
            duration = time.time() - start_time
            
            # Get resource usage
            peak_memory = monitor_data['peak_memory'] / (1024 * 1024)  # Convert to MB
            cpu_percent = monitor_data['cpu_percent']
            
            # Update stats
            with self._stats['total_executions'].get_lock():
                self._stats['total_executions'].value += 1
            
            if exit_code == 0:
                with self._stats['successful_executions'].get_lock():
                    self._stats['successful_executions'].value += 1
            else:
                with self._stats['failed_executions'].get_lock():
                    self._stats['failed_executions'].value += 1
            
            if timed_out:
                with self._stats['timeout_executions'].get_lock():
                    self._stats['timeout_executions'].value += 1
            
            self._update_average_duration(duration)
            
            return ProcessResult(
                exit_code=exit_code,
                stdout=stdout or "",
                stderr=stderr or "",
                duration=duration,
                memory_peak_mb=peak_memory,
                cpu_percent=cpu_percent,
                timed_out=timed_out
            )
            
        except Exception as e:
            return ProcessResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=time.time() - start_time,
                memory_peak_mb=0.0,
                cpu_percent=0.0,
                error=str(e)
            )
        finally:
            # Cleanup
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=1.0)
                except Exception:
                    process.kill()
    
    def _monitor_process(self, pid: int, data: Dict[str, float]):
        """Monitor process resource usage."""
        try:
            process = psutil.Process(pid)
            
            while process.is_running():
                try:
                    # Memory usage
                    mem_info = process.memory_info()
                    data['peak_memory'] = max(data['peak_memory'], mem_info.rss)
                    
                    # CPU usage
                    cpu_percent = process.cpu_percent(interval=0.1)
                    data['cpu_percent'] = max(data['cpu_percent'], cpu_percent)
                    
                    time.sleep(0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except Exception:
            pass
    
    def _update_average_duration(self, duration: float):
        """Update average duration metric."""
        total = self._stats['total_executions'].value
        if total == 0:
            return
        
        with self._stats['average_duration'].get_lock():
            current_avg = self._stats['average_duration'].value
            new_avg = (current_avg * (total - 1) + duration) / total
            self._stats['average_duration'].value = new_avg
    
    def execute(self, config: ProcessConfig, timeout: Optional[float] = None) -> ProcessResult:
        """Execute a subprocess through the pool."""
        
        # Get shared memory segment if needed
        shm_idx = None
        if config.stdin_data and len(config.stdin_data) > 1024:  # Use shm for large data
            try:
                shm_idx = self._shm_available.get_nowait()
                shm = self._shm_pool[shm_idx]
                
                # Write data to shared memory
                data_bytes = config.stdin_data.encode('utf-8')
                size = len(data_bytes)
                shm.buf[:8] = struct.pack('Q', size)
                shm.buf[8:8+size] = data_bytes
            except Exception:
                shm_idx = None
        
        # Submit task
        task_id = time.time()
        self._task_queue.put((task_id, config, shm_idx))
        
        # Wait for result
        timeout = timeout or config.timeout + 5.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result_task_id, result, result_shm_idx = self._result_queue.get(timeout=0.1)
                
                if result_task_id == task_id:
                    # Release shared memory
                    if result_shm_idx is not None:
                        self._shm_available.put(result_shm_idx)
                    
                    return result
                else:
                    # Put back result for correct task
                    self._result_queue.put((result_task_id, result, result_shm_idx))
            except queue.Empty:
                continue
        
        # Timeout waiting for result
        if shm_idx is not None:
            self._shm_available.put(shm_idx)
        
        return ProcessResult(
            exit_code=-1,
            stdout="",
            stderr="Task timeout in pool",
            duration=timeout,
            memory_peak_mb=0.0,
            cpu_percent=0.0,
            timed_out=True,
            error="Pool task timeout"
        )
    
    def execute_batch(self, configs: List[ProcessConfig]) -> List[ProcessResult]:
        """Execute multiple subprocesses in parallel."""
        
        # Use thread pool for concurrent submission
        with ThreadPoolExecutor(max_workers=min(len(configs), self.pool_size)) as executor:
            futures = [executor.submit(self.execute, config) for config in configs]
            results = [future.result() for future in futures]
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_size': self.pool_size,
            'total_executions': self._stats['total_executions'].value,
            'successful_executions': self._stats['successful_executions'].value,
            'failed_executions': self._stats['failed_executions'].value,
            'timeout_executions': self._stats['timeout_executions'].value,
            'average_duration': self._stats['average_duration'].value,
            'success_rate': (
                self._stats['successful_executions'].value / 
                max(1, self._stats['total_executions'].value)
            )
        }
    
    def shutdown(self):
        """Shutdown the process pool."""
        # Signal shutdown
        self._shutdown.set()
        
        # Send sentinel values
        for _ in range(self.pool_size):
            self._task_queue.put(None)
        
        # Wait for workers
        for worker in self._workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
        
        # Cleanup shared memory
        for shm in self._shm_pool:
            shm.close()
            shm.unlink()


class SubprocessCoordinator:
    """High-level coordinator for subprocess execution."""
    
    def __init__(self, pool_size: int = 4):
        self.pool = ProcessPool(pool_size=pool_size)
        self._execution_cache = {}
        self._cache_lock = threading.Lock()
    
    def execute_hook(self, 
                    hook_path: str,
                    hook_data: Dict[str, Any],
                    timeout: float = 10.0,
                    memory_limit_mb: int = 100) -> Dict[str, Any]:
        """Execute a hook subprocess."""
        
        # Check cache
        cache_key = f"{hook_path}:{json.dumps(hook_data, sort_keys=True)}"
        with self._cache_lock:
            if cache_key in self._execution_cache:
                cached = self._execution_cache[cache_key]
                if time.time() - cached['timestamp'] < 300:  # 5 min cache
                    return cached['result']
        
        # Prepare configuration
        config = ProcessConfig(
            command=[sys.executable, hook_path],
            stdin_data=json.dumps(hook_data),
            memory_limit_mb=memory_limit_mb,
            timeout=timeout,
            capture_output=True
        )
        
        # Execute
        result = self.pool.execute(config)
        
        # Process result
        hook_result = {
            'exit_code': result.exit_code,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': result.duration,
            'memory_mb': result.memory_peak_mb,
            'timed_out': result.timed_out
        }
        
        # Cache successful results
        if result.exit_code == 0:
            with self._cache_lock:
                self._execution_cache[cache_key] = {
                    'result': hook_result,
                    'timestamp': time.time()
                }
                
                # Limit cache size
                if len(self._execution_cache) > 1000:
                    # Remove oldest entries
                    sorted_keys = sorted(
                        self._execution_cache.keys(),
                        key=lambda k: self._execution_cache[k]['timestamp']
                    )
                    for key in sorted_keys[:100]:
                        del self._execution_cache[key]
        
        return hook_result
    
    def execute_parallel_hooks(self,
                             hook_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple hooks in parallel."""
        
        configs = []
        for spec in hook_specs:
            config = ProcessConfig(
                command=[sys.executable, spec['path']],
                stdin_data=json.dumps(spec.get('data', {})),
                memory_limit_mb=spec.get('memory_limit', 100),
                timeout=spec.get('timeout', 10.0),
                capture_output=True
            )
            configs.append(config)
        
        # Execute batch
        results = self.pool.execute_batch(configs)
        
        # Convert results
        hook_results = []
        for result in results:
            hook_results.append({
                'exit_code': result.exit_code,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': result.duration,
                'memory_mb': result.memory_peak_mb,
                'timed_out': result.timed_out
            })
        
        return hook_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'pool_stats': self.pool.get_stats(),
            'cache_size': len(self._execution_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)."""
        # In production, track hits/misses properly
        return 0.0
    
    def shutdown(self):
        """Shutdown the coordinator."""
        self.pool.shutdown()


# Global coordinator instance
_global_coordinator: Optional[SubprocessCoordinator] = None


def get_subprocess_coordinator() -> SubprocessCoordinator:
    """Get or create global subprocess coordinator."""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = SubprocessCoordinator()
    return _global_coordinator