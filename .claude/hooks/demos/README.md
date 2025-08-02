# ZEN Hook System Demos

This directory contains demonstration and test scripts for the ZEN functionality integrated into the hook system.

## Files

### zen_copilot_simple_test.py
A simplified test for the ZEN Co-pilot Phase 2 Adaptive Learning Engine. Tests 4 specialized learning models without complex dependencies.

### zen_copilot_neural_commands.py
CLI interface for enhanced neural training commands. Provides commands like:
- `neural-train` - Enhanced with 4 specialized models
- `pattern-learn` - Advanced pattern learning with behavioral analysis
- `model-update` - Adaptive model updates with memory integration
- `zen-learn` - Comprehensive ZEN learning orchestration

### demo_zen_integration.py
Demonstrates complete ZEN workflow integration including:
- ZenConsultant integration with consensus
- Hook system integration
- Memory learning patterns
- Performance comparisons

### demo_ml_optimizer.py
ML-Enhanced Adaptive Learning Engine performance demonstration showing:
- Resource utilization optimization
- Real-time learning and adaptation
- Performance metrics and benchmarking

## Usage

These scripts are for testing and demonstration purposes. The actual ZEN functionality is integrated into the core hook modules at:
- `.claude/hooks/modules/core/zen_adaptive_learning_engine.py`
- `.claude/hooks/modules/core/zen_consultant.py`
- `.claude/hooks/modules/core/zen_memory_pipeline.py`
- And other ZEN modules in the core directory

To run demos:
```bash
python demo_zen_integration.py
python demo_ml_optimizer.py
python zen_copilot_simple_test.py
```