# Orphan Files Analysis

## Identified Orphan Categories

### 1. Demo/Example Files (Can be moved to examples/)
- ~~`demo_ml_optimizer.py` - ML optimizer demonstration~~ **[MOVED to demos/]**
- ~~`demo_zen_integration.py` - ZEN integration demo~~ **[MOVED to demos/]**
- `example_risk_assessment.py` - Risk assessment example
- ~~`zen_copilot_simple_test.py` - Simple test for ZEN copilot~~ **[MOVED to demos/]**

### 2. Standalone Test Files (Should be in tests/)
- `test_context_intelligence_engine.py`
- `test_functionality_framework.py`
- `test_integration_framework.py`
- `test_load_framework.py`
- `test_master_suite.py`
- `test_memory_namespace.py`
- `test_object_pool_integration.py`
- `test_performance_framework.py`
- `test_process_manager.py`
- `test_risk_assessment.py`
- `test_security_framework.py`
- `test_zen_adaptive_integration.py`
- `test_zen_consultant.py`

### 3. Benchmark/Performance Files (Can be moved to benchmarks/)
- `benchmark_ml_optimizer.py`
- `performance_dashboard.py`

### 4. Legacy/Duplicate Validators (In modules/pre_tool/analyzers/legacy/)
Already properly organized in legacy folder - these are kept for reference

### 5. Utility Scripts (May need integration or removal)
- `session_restore.py` - Not referenced in settings.json
- `neural_pattern_training.py` - Standalone neural training
- ~~`zen_copilot_neural_commands.py` - CLI commands for neural training~~ **[MOVED to demos/]**

### 6. Potentially Missing Imports
Files that are imported but might not exist:
- `modules.core.Analyzer` - Used in analyzer files but not found
- `modules.core.PatternMatch` - Used in analyzer files but not found

## Recommendations

1. **Create directory structure:**
   ```
   .claude/hooks/
   ├── examples/
   ├── tests/
   ├── benchmarks/
   └── utils/
   ```

2. **Move files accordingly:**
   - Demo files → examples/
   - Test files → tests/
   - Benchmark files → benchmarks/
   - Utility scripts → utils/

3. **Fix missing imports:**
   - Check if `Analyzer` and `PatternMatch` should be in `modules/core/__init__.py`
   - Or update imports to use actual locations

4. **Remove truly orphaned files:**
   - Files with no imports and no purpose
   - Duplicate functionality