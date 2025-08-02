# Universal Tool Feedback System

## Architecture Overview

This module implements a complete modular architecture for expanding the non-blocking stderr exit(2) feedback system to all common tool matchers, designed by the **stderr-system-architect**.

### Key Components

```
📁 post_tool/
├── 📁 core/                    # Core system architecture
│   ├── tool_analyzer_base.py   # Protocol interfaces & base classes
│   ├── analyzer_registry.py    # Dynamic analyzer registration
│   ├── hook_integration.py     # PostToolUse hook integration
│   ├── performance_optimizer.py # Caching & async execution
│   └── system_integration.py   # Complete system coordinator
├── 📁 analyzers/
│   └── 📁 specialized/         # Tool-specific analyzers
│       ├── file_operations_analyzer.py      # File system operations
│       ├── mcp_coordination_analyzer.py     # MCP hierarchy validation
│       └── execution_safety_analyzer.py     # Command security analysis
└── 📁 manager.py              # Integration with existing hooks
```

## Performance Targets

- **Sub-100ms stderr feedback generation** ⚡
- **Non-blocking async-first execution** 🚀
- **Intelligent caching with LRU eviction** 🧠
- **Priority-based analyzer execution** 📊
- **Circuit breaker protection** 🛡️

## Integration

### PostToolUse Hook Integration

```python
from post_tool.core.system_integration import analyze_tool_with_universal_system_sync

# In your PostToolUse hook:
exit_code = analyze_tool_with_universal_system_sync(
    tool_name, tool_input, tool_response, session_context
)

if exit_code == 1:
    sys.exit(1)  # Block operation
elif exit_code == 2:
    sys.exit(2)  # Provide guidance
# exit_code == 0 or None: Continue normally
```

### Analyzer Development

```python
from post_tool.core.tool_analyzer_base import BaseToolAnalyzer, ToolContext, FeedbackResult

class MyCustomAnalyzer(BaseToolAnalyzer):
    def get_analyzer_name(self) -> str:
        return "my_custom_analyzer"
    
    def get_supported_tools(self) -> List[str]:
        return ["MyTool", "AnotherTool"]
    
    async def analyze_tool(self, context: ToolContext) -> Optional[FeedbackResult]:
        # Your analysis logic here
        if needs_feedback:
            return FeedbackResult(
                message="Your feedback message",
                severity=FeedbackSeverity.MEDIUM,
                exit_code=2
            )
        return None
```

## Testing

Run the comprehensive integration test:

```bash
python test_universal_system_integration.py
```

This validates:
- ✅ System initialization (<500ms)
- ✅ Individual analyzer functionality  
- ✅ Registry integration
- ✅ Performance benchmarks (sub-100ms target)
- ✅ Hook integration compatibility
- ✅ Real-world usage scenarios

## Architecture Benefits

1. **Modular Design**: Easy to add new analyzers without touching existing code
2. **High Performance**: Async-first with intelligent caching and circuit breakers
3. **Backward Compatible**: Seamless integration with existing PostToolUse patterns
4. **Extensible**: Plugin architecture supports unlimited analyzer types
5. **Robust**: Circuit breakers and fallbacks ensure system reliability
6. **Observable**: Comprehensive metrics and diagnostics

## Usage in Claude Hook Ecosystem

This system integrates seamlessly with the Claude Hook → ZEN → Claude Flow architecture:

- **Hooks** (Primary Intelligence): Use this system for stderr feedback
- **ZEN** (Orchestration): Analyzers can suggest ZEN coordination patterns
- **Claude Flow** (Execution): Feedback can recommend agent spawning

The system maintains the golden rule: **MCP Tools = Coordination ONLY**, never execution.

---

*Built for the Claude Code Intelligence System with ❤️*