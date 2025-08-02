#!/usr/bin/env python3
"""Path resolver module for Claude hook files.

Eliminates the need for messy sys.path.insert() calls scattered throughout hook files.
Provides a clean, centralized way to manage Python import paths for the hook system.

Usage:
    from modules.utils.path_resolver import setup_hook_paths
    setup_hook_paths()  # Sets up all necessary paths
    
    # Or use individual functions:
    from modules.utils.path_resolver import ensure_modules_path, ensure_utils_path
    ensure_modules_path()  # For modules.* imports
    ensure_utils_path()   # For process_manager imports
"""

import sys
from pathlib import Path


def get_hook_root() -> Path:
    """Get the hook system root directory.
    
    Returns:
        Path: The .claude/hooks directory path
    """
    # This file is at .claude/hooks/modules/utils/path_resolver.py
    # So we need to go up 3 levels to get to .claude/hooks/
    current_file = Path(__file__)
    hook_root = current_file.parent.parent.parent
    return hook_root.resolve()


def get_modules_path() -> Path:
    """Get the modules directory path.
    
    Returns:
        Path: The .claude/hooks/modules directory path
    """
    return get_hook_root() / "modules"


def get_utils_path() -> Path:
    """Get the utils directory path.
    
    Returns:
        Path: The .claude/hooks/modules/utils directory path
    """
    return get_modules_path() / "utils"


def ensure_path_in_sys(path: Path) -> bool:
    """Ensure a path is in sys.path.
    
    Args:
        path: The path to add to sys.path
        
    Returns:
        bool: True if path was added, False if already present
    """
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        return True
    return False


def ensure_hook_root_path() -> bool:
    """Ensure the hook root directory is in sys.path.
    
    This enables imports like:
    - from modules.optimization import ...
    - from modules.pre_tool import ...
    - from modules.post_tool import ...
    
    Returns:
        bool: True if path was added, False if already present
    """
    hook_root = get_hook_root()
    return ensure_path_in_sys(hook_root)


def ensure_modules_path() -> bool:
    """Ensure the modules directory is in sys.path.
    
    This enables direct imports from the modules directory.
    
    Returns:
        bool: True if path was added, False if already present
    """
    modules_path = get_modules_path()
    return ensure_path_in_sys(modules_path)


def ensure_utils_path() -> bool:
    """Ensure the utils directory is in sys.path.
    
    This enables imports like:
    - from process_manager import managed_subprocess_run
    
    Returns:
        bool: True if path was added, False if already present
    """
    utils_path = get_utils_path()
    return ensure_path_in_sys(utils_path)


def setup_hook_paths() -> dict[str, bool]:
    """Set up all necessary paths for hook file imports.
    
    This is the main function that hook files should call.
    It sets up all the common import paths needed by hook files.
    
    Returns:
        dict: Status of each path setup (True if added, False if already present)
    """
    return {
        "hook_root": ensure_hook_root_path(),
        "utils": ensure_utils_path(),
    }


def remove_path_from_sys(path: Path) -> bool:
    """Remove a path from sys.path if present.
    
    Args:
        path: The path to remove from sys.path
        
    Returns:
        bool: True if path was removed, False if not present
    """
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
        return True
    return False


def cleanup_hook_paths() -> dict[str, bool]:
    """Clean up hook paths from sys.path.
    
    Useful for testing or when you want to reset the path state.
    
    Returns:
        dict: Status of each path cleanup (True if removed, False if not present)
    """
    return {
        "hook_root": remove_path_from_sys(get_hook_root()),
        "utils": remove_path_from_sys(get_utils_path()),
    }


def get_import_path_info() -> dict[str, str | bool]:
    """Get information about current import paths.
    
    Useful for debugging import issues.
    
    Returns:
        dict: Information about resolved paths
    """
    return {
        "hook_root": str(get_hook_root()),
        "modules_path": str(get_modules_path()),
        "utils_path": str(get_utils_path()),
        "hook_root_in_syspath": str(get_hook_root()) in sys.path,
        "utils_in_syspath": str(get_utils_path()) in sys.path,
    }


# Convenience function for backwards compatibility
def setup_paths() -> dict[str, bool]:
    """Alias for setup_hook_paths() for backwards compatibility."""
    return setup_hook_paths()


if __name__ == "__main__":
    """Command-line interface for path resolver."""
    import json
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            result = setup_hook_paths()
            print(f"Path setup result: {json.dumps(result, indent=2)}")
            
        elif command == "cleanup":
            result = cleanup_hook_paths()
            print(f"Path cleanup result: {json.dumps(result, indent=2)}")
            
        elif command == "info":
            info = get_import_path_info()
            print(f"Import path info: {json.dumps(info, indent=2)}")
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup, cleanup, info")
            sys.exit(1)
    else:
        # Default: show info
        info = get_import_path_info()
        print(f"Import path info: {json.dumps(info, indent=2)}")