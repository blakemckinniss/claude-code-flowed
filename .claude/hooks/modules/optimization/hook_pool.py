"""Hook Execution Pool for eliminating cold start penalties.

This module maintains a pool of persistent Python processes that can execute
hooks without the overhead of spawning new processes for each operation.
"""

import subprocess
import sys
import json

# Path setup handled by centralized resolver when importing this module
import queue
import threading
import time
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import process manager for managed subprocess operations
from ..utils.process_manager import managed_subprocess_popen, get_process_manager


class HookWorkerProcess:
    """A persistent Python process for hook execution."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.process = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.running = False
        self.stdout_thread = None
        self.stderr_thread = None
        self._start_process()
    
    def _start_process(self):
        """Start the worker process."""
        worker_script = Path(__file__).parent / "hook_worker.py"
        
        # Create worker script if it doesn't exist
        if not worker_script.exists():
            self._create_worker_script(worker_script)
        
        # Thread-safe process initialization
        with threading.Lock():
            try:
                self.process = managed_subprocess_popen(
                    [sys.executable, "-u", str(worker_script), str(self.worker_id)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    timeout=None,  # Long-running worker process
                    max_memory_mb=100,  # 100MB memory limit for worker
                    tags={"hook": "worker-pool", "worker_id": self.worker_id}
                )
                
                # Verify process started successfully
                if self.process.poll() is not None:
                    raise RuntimeError(f"Worker process {self.worker_id} failed to start")
                
                # Set running flag BEFORE starting threads to prevent race condition
                self.running = True
                
                # Start reader threads only after confirming process is running and running flag is set
                self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
                self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
                
                self.stdout_thread.start()
                self.stderr_thread.start()
                
            except Exception as e:
                # Ensure running is False and cleanup properly
                self.running = False
                if self.process:
                    try:
                        self.process.terminate()
                        # Wait briefly for graceful termination
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                    finally:
                        self.process = None
                raise RuntimeError(f"Failed to start worker process {self.worker_id}: {e!s}")
    
    def _create_worker_script(self, path: Path):
        """Create the worker script."""
        worker_code = '''#!/usr/bin/env python3
"""Hook worker process for persistent execution."""

import sys
import json
import importlib.util
import traceback

def run_worker(worker_id):
    """Run the worker loop."""
    print(f"Worker {worker_id} started", file=sys.stderr)
    
    while True:
        try:
            # Read request from stdin
            line = sys.stdin.readline()
            if not line:
                break
            
            request = json.loads(line.strip())
            
            # Execute hook
            result = execute_hook(request)
            
            # Send response
            print(json.dumps(result))
            sys.stdout.flush()
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(json.dumps(error_result))
            sys.stdout.flush()

def execute_hook(request):
    """Execute a hook request."""
    hook_type = request.get("hook_type")
    hook_path = request.get("hook_path")
    hook_data = request.get("hook_data", {})
    
    # Load and execute hook module
    spec = importlib.util.spec_from_file_location("hook_module", hook_path)
    module = importlib.util.module_from_spec(spec)
    
    # Redirect stdin for the hook
    import io
    original_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps(hook_data))
    
    try:
        spec.loader.exec_module(module)
        
        # Call main function if it exists
        if hasattr(module, "main"):
            module.main()
        
        return {"success": True, "exit_code": 0}
        
    except SystemExit as e:
        return {"success": True, "exit_code": e.code if e.code is not None else 0}
        
    finally:
        sys.stdin = original_stdin

if __name__ == "__main__":
    worker_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_worker(worker_id)
'''
        path.write_text(worker_code)
        path.chmod(0o755)
    
    def _read_stdout(self):
        """Read stdout from process."""
        while self.running:
            try:
                # Check if process is still valid
                if self.process is None or self.process.stdout is None:
                    break
                    
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(json.loads(line.strip()))
                elif self.process.poll() is not None:
                    # Process has terminated
                    break
            except Exception as e:
                self.error_queue.put(str(e))
                break
    
    def _read_stderr(self):
        """Read stderr from process."""
        while self.running:
            try:
                # Check if process is still valid
                if self.process is None or self.process.stderr is None:
                    break
                    
                line = self.process.stderr.readline()
                if line:
                    # Log stderr but don't treat as error
                    print(f"Worker {self.worker_id}: {line.strip()}", file=sys.stderr)
                elif self.process.poll() is not None:
                    # Process has terminated
                    break
            except Exception:
                break
    
    def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a hook request."""
        if not self.running:
            raise RuntimeError(f"Worker {self.worker_id} is not running")
        
        # Check if process is still valid
        if self.process is None or self.process.stdin is None:
            return {
                "success": False,
                "error": f"Worker {self.worker_id} process is not available",
                "exit_code": 1
            }
        
        # Send request
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            return {
                "success": False,
                "error": f"Failed to send request to worker {self.worker_id}: {e!s}",
                "exit_code": 1
            }
        
        # Wait for response (with timeout)
        try:
            result = self.output_queue.get(timeout=60)
            return result
        except queue.Empty:
            return {
                "success": False,
                "error": "Hook execution timeout",
                "exit_code": 1
            }
    
    def shutdown(self):
        """Shutdown the worker process."""
        # Set running flag to False to signal threads to stop
        self.running = False
        
        # Wait for reader threads to finish cleanly
        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout=2)
        if self.stderr_thread and self.stderr_thread.is_alive():
            self.stderr_thread.join(timeout=2)
        
        if self.process:
            try:
                # Close stdin to signal the worker to exit gracefully
                if self.process.stdin:
                    self.process.stdin.close()
                
                # Wait for graceful shutdown
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force terminate if graceful shutdown fails
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            finally:
                self.process = None
                self.stdout_thread = None
                self.stderr_thread = None


class HookExecutionPool:
    """Pool of persistent hook execution processes."""
    
    def __init__(self, pool_size: int = 3):
        self.pool_size = pool_size
        self.workers: List[HookWorkerProcess] = []
        self.available_workers = queue.Queue()
        self.lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the worker pool."""
        for i in range(self.pool_size):
            worker = HookWorkerProcess(i)
            self.workers.append(worker)
            self.available_workers.put(worker)
    
    def execute_hook(self, hook_type: str, hook_path: str, hook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a hook using an available worker."""
        # Get available worker
        try:
            worker = self.available_workers.get(timeout=30)
        except queue.Empty:
            # All workers busy, create temporary worker
            worker = HookWorkerProcess(len(self.workers))
        
        try:
            # Execute hook
            request = {
                "hook_type": hook_type,
                "hook_path": hook_path,
                "hook_data": hook_data
            }
            
            result = worker.execute(request)
            
            # Return worker to pool
            if worker in self.workers:
                self.available_workers.put(worker)
            else:
                # Shutdown temporary worker
                worker.shutdown()
            
            return result
            
        except Exception as e:
            # Return worker to pool even on error
            if worker in self.workers:
                self.available_workers.put(worker)
            
            return {
                "success": False,
                "error": str(e),
                "exit_code": 1
            }
    
    def shutdown(self):
        """Shutdown all workers in the pool."""
        for worker in self.workers:
            worker.shutdown()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global pool instance for reuse across hook executions
_global_pool: Optional[HookExecutionPool] = None


def get_global_pool() -> HookExecutionPool:
    """Get or create the global hook execution pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = HookExecutionPool()
    return _global_pool