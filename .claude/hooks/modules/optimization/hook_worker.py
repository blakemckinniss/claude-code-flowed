#!/usr/bin/env python3
"""Hook worker process for persistent execution."""

import sys
import json

# Path setup handled by centralized resolver when importing this module
import importlib.util
import traceback
from pathlib import Path

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
    
    if not hook_path:
        raise ValueError("hook_path is required but not provided")
    
    if not Path(hook_path).exists():
        raise FileNotFoundError(f"Hook file not found: {hook_path}")
    
    # Load and execute hook module
    spec = importlib.util.spec_from_file_location("hook_module", hook_path)
    if spec is None:
        raise ValueError(f"Failed to create module spec for hook path: {hook_path}")
    if spec.loader is None:
        raise ValueError(f"Module spec has no loader for hook path: {hook_path}")
    
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
