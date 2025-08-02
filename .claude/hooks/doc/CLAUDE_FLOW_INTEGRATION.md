# Claude-Flow Integration for Hook System

## Overview

The hook system now integrates with `claude-flow` CLI to provide intelligent command suggestions and workflow optimizations. When you run commands through the process manager, it can automatically suggest more powerful claude-flow alternatives.

## Features

### 1. **Automatic Command Analysis**

When running bash commands, the system analyzes them and suggests claude-flow alternatives:

```bash
# Running: npm test
# Suggestion: npx claude-flow swarm "Run comprehensive tests" --strategy testing
# Benefit: Parallel test execution with intelligent agent coordination
```

### 2. **Context-Aware Command Generation**

The system can generate appropriate claude-flow commands based on task context:

```python
from modules.utils.process_manager import ClaudeFlowIntegration

cf = ClaudeFlowIntegration()
context = {
    'task_type': 'api_development',
    'description': 'Build REST API with authentication',
    'parameters': {}
}
command, description = cf.suggest_command(context)
# Returns: ['npx', 'claude-flow', 'swarm', 'Build REST API...', '--strategy', 'development', ...]
```

### 3. **Pre-Tool Hook Integration**

The `ClaudeFlowSuggesterValidator` runs as a pre-tool hook that:
- Analyzes Bash commands before execution
- Suggests claude-flow alternatives when beneficial
- Provides non-blocking suggestions (allows original command to proceed)

### 4. **Helper Functions**

Convenient helper functions for common claude-flow operations:

```python
from modules.utils.process_manager import (
    run_claude_flow_swarm,
    run_claude_flow_sparc,
    get_claude_flow_status,
    list_claude_flow_agents
)

# Run a development swarm
result = run_claude_flow_swarm(
    task="Build user authentication system",
    strategy="development",
    max_agents=5,
    parallel=True
)

# Run SPARC TDD workflow
result = run_claude_flow_sparc(
    mode="tdd",
    task="User registration feature"
)

# Check status
status = get_claude_flow_status()

# List active agents
agents = list_claude_flow_agents()
```

## Command Patterns

The integration recognizes these patterns and suggests appropriate claude-flow commands:

### Development Tasks
- **Pattern**: `api`, `rest`, `endpoint` in task description
- **Suggestion**: `npx claude-flow swarm "{task}" --strategy development --max-agents 5 --parallel`

### Code Review
- **Pattern**: `review`, `audit`, `analyze` in task description
- **Suggestion**: `npx claude-flow swarm "{task}" --strategy analysis --read-only`

### Security Audit
- **Pattern**: `security`, `vulnerability` in description
- **Suggestion**: `npx claude-flow swarm "{task}" --strategy analysis --analysis --monitor`

### Performance Optimization
- **Pattern**: `optimize`, `performance`, `speed` in task description
- **Suggestion**: `npx claude-flow swarm "{task}" --strategy optimization --parallel --ui`

### Testing
- **Pattern**: `test`, `spec`, `unit` in task description
- **Suggestion**: `npx claude-flow swarm "{task}" --strategy testing --max-agents 4`

### SPARC Development
- **TDD Pattern**: `sparc` + `tdd` in description
- **Suggestion**: `npx claude-flow sparc tdd "{feature}"`
- **Dev Pattern**: `sparc` + `dev` in description
- **Suggestion**: `npx claude-flow sparc run dev "{task}"`

## Process Manager Integration

The ProcessManager class now includes:

1. **Automatic Suggestions**: When `suggest_claude_flow=True` (default), commands are analyzed for optimization opportunities

2. **Claude-Flow Execution**: Direct methods to run claude-flow commands with process management:
   ```python
   manager = get_process_manager()
   
   # Run claude-flow based on context
   result = manager.run_claude_flow(context, timeout=300)
   
   # Run asynchronously with callback
   thread = manager.run_claude_flow_async(context, callback=handle_result)
   ```

3. **Resource Management**: All claude-flow processes are managed with:
   - Timeout enforcement
   - Memory limits
   - Process tracking
   - Automatic cleanup

## Configuration

The claude-flow suggester is enabled by default in the pre-tool validators. To disable:

1. Edit `.claude/hooks/modules/pre_tool/manager.py`
2. Remove `"claude_flow_suggester"` from the `enabled_validators` list

Priority: 500 (Medium) - Runs after critical validators but provides helpful suggestions

## Best Practices

1. **Use Helper Functions**: Instead of manually constructing claude-flow commands, use the provided helpers
2. **Context is Key**: Provide detailed task descriptions for better command suggestions
3. **Monitor Resources**: Claude-flow swarms can spawn multiple agents - the process manager tracks them all
4. **Leverage Suggestions**: When you see a suggestion, consider using it for enhanced capabilities

## Example Hook Output

When running a command that could benefit from claude-flow:

```
💡 Claude-Flow Enhancement Available!
   Current command: npm test
   • Consider using: npx claude-flow swarm "Run comprehensive tests" --strategy testing
     Benefit: Parallel test execution with intelligent agent coordination

   🐝 Based on your task description, consider:
      npx claude-flow swarm "Run comprehensive tests" --strategy testing --max-agents 4
      Purpose: Deploy testing swarm for comprehensive coverage
```

## Future Enhancements

- [ ] Auto-execute claude-flow commands when confidence is high
- [ ] Learn from user preferences (accept/reject suggestions)
- [ ] Integration with hive-mind wizard for complex workflows
- [ ] Real-time swarm monitoring integration
- [ ] Automatic fallback to claude-flow on command failures