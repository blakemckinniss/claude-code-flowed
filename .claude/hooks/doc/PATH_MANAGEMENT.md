# Centralized Path Management for Claude Hooks

## Overview

This document describes the centralized path management system implemented to replace scattered `sys.path.insert()` calls throughout the Claude hooks project.

## Problem Solved

Previously, hook files contained repetitive and error-prone path management code:

```python
# Old approach - scattered throughout files
import sys
import os
hooks_dir = Path(__file__).parent
sys.path.insert(0, str(hooks_dir))
```

This approach had several issues:
- **Code duplication** - Same path setup logic repeated in 20+ files
- **Maintenance burden** - Any path changes required updating multiple files
- **Error prone** - Easy to forget or incorrectly implement path setup
- **Inconsistent patterns** - Different files used slightly different approaches

## Solution: Centralized Path Resolver

All path management is now centralized in [`modules/utils/path_resolver.py`](.claude/hooks/modules/utils/path_resolver.py:1).

### Usage Pattern

All hook files now use this simple pattern:

```python
# New approach - centralized and consistent
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Now all module imports work correctly
from modules.utils.process_manager import managed_subprocess_run
```

### Key Benefits

1. **Single Source of Truth** - All path logic in one place
2. **Easy Maintenance** - Changes only need to be made in one file
3. **Consistent Behavior** - All hooks use identical path setup
4. **Error Prevention** - Harder to get path setup wrong
5. **Clean Code** - Removes boilerplate from all hook files

## Implementation Details

### Core Functions

- [`setup_hook_paths()`](.claude/hooks/modules/utils/path_resolver.py:108) - Main function that sets up all necessary paths
- [`ensure_hook_root_path()`](.claude/hooks/modules/utils/path_resolver.py:68) - Ensures hook root is in sys.path
- [`ensure_utils_path()`](.claude/hooks/modules/utils/path_resolver.py:95) - Ensures utils directory is in sys.path

### Path Structure

The resolver manages these key paths:
- `.claude/hooks/` (hook root)
- `.claude/hooks/modules/` (modules directory)  
- `.claude/hooks/modules/utils/` (utils directory)

### sys Usage Policy

After this refactoring:

- **✅ Allowed**: [`modules/utils/path_resolver.py`](.claude/hooks/modules/utils/path_resolver.py:1) - Central path management
- **✅ Allowed**: Hook files using `sys.stdin`, `sys.stderr`, `sys.exit()` for I/O operations
- **❌ Removed**: All `sys.path.insert()` calls from individual hook files
- **❌ Removed**: All `os.path.dirname(__file__)` path manipulation in hooks

## Files Updated

### Main Hook Files
- [`performance_monitor.py`](.claude/hooks/performance_monitor.py:1)
- [`performance_dashboard.py`](.claude/hooks/performance_dashboard.py:1) 
- [`session_start.py`](.claude/hooks/session_start.py:1)
- [`session_restore.py`](.claude/hooks/session_restore.py:1)
- [`session_end.py`](.claude/hooks/session_end.py:1)
- [`user_prompt_submit.py`](.claude/hooks/user_prompt_submit.py:1)
- [`neural_pattern_training.py`](.claude/hooks/neural_pattern_training.py:1)
- [`pre_compact.py`](.claude/hooks/pre_compact.py:1)

### Module Files
- [`modules/utils/helpers.py`](.claude/hooks/modules/utils/helpers.py:1)
- [`modules/optimization/cache.py`](.claude/hooks/modules/optimization/cache.py:1)
- [`modules/optimization/hook_pool.py`](.claude/hooks/modules/optimization/hook_pool.py:1)
- [`modules/optimization/async_db.py`](.claude/hooks/modules/optimization/async_db.py:1)
- [`modules/optimization/hook_worker.py`](.claude/hooks/modules/optimization/hook_worker.py:1)
- [`modules/post_tool/manager.py`](.claude/hooks/modules/post_tool/manager.py:1)
- [`modules/pre_tool/manager.py`](.claude/hooks/modules/pre_tool/manager.py:1)
- [`modules/post_tool/core/guidance_system.py`](.claude/hooks/modules/post_tool/core/guidance_system.py:1)

### Already Using Correct Pattern
- [`stop.py`](.claude/hooks/stop.py:1)
- [`pre_tool_use.py`](.claude/hooks/pre_tool_use.py:1)
- [`post_tool_use.py`](.claude/hooks/post_tool_use.py:1)

## Verification

The centralized approach has been tested and verified:

```bash
cd .claude/hooks
python -c "from modules.utils.path_resolver import setup_hook_paths; setup_hook_paths(); from modules.utils.process_manager import managed_subprocess_run; print('✅ Import test successful')"
```

## Future Maintenance

### Adding New Hook Files
When creating new hook files, always use this pattern:

```python
#!/usr/bin/env python3
"""Your hook description here."""

import sys  # Only if you need sys.stdin, sys.stderr, sys.exit()
import json
# ... other standard imports

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Now you can import from modules
from modules.utils.process_manager import managed_subprocess_run
```

### Modifying Path Logic
- **Never** add `sys.path.insert()` calls to individual hook files
- **Always** modify [`path_resolver.py`](.claude/hooks/modules/utils/path_resolver.py:1) if path changes are needed
- Test changes with the verification command above

## Summary

This refactoring successfully:
- ✅ Removed all scattered `sys.path.insert()` usage (41 files analyzed)
- ✅ Centralized path management in [`path_resolver.py`](.claude/hooks/modules/utils/path_resolver.py:1)
- ✅ Maintained all existing functionality
- ✅ Verified imports work correctly
- ✅ Improved code maintainability and consistency

The Claude hooks project now has clean, maintainable path management that follows Python best practices.