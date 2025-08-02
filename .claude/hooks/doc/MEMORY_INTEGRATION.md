# Claude Flow Memory Integration Documentation

## Overview

The Claude Flow Memory Integration system provides project-specific namespace isolation and semantic memory management for Claude hooks. This system enables persistent memory storage across sessions, intelligent context retrieval, and seamless integration with the claude-flow MCP tools.

## Architecture

### Core Components

1. **Project Memory Manager** (`modules/memory/project_memory_manager.py`)
   - Manages project-specific memory namespaces
   - Handles semantic context analysis
   - Provides memory storage and retrieval operations
   - Integrates with claude-flow MCP backend

2. **Hook Memory Integration** (`modules/memory/hook_memory_integration.py`)
   - Captures memories throughout the hook lifecycle
   - Manages session-based memory operations
   - Provides context-aware memory persistence

3. **Retrieval Patterns** (`modules/memory/retrieval_patterns.py`)
   - Implements intelligent memory retrieval patterns
   - Provides context-aware memory search
   - Enables error prevention through historical analysis

4. **CLI Integration** (`modules/memory/claude_flow_cli.py`)
   - Command-line interface for memory operations
   - Integrates with `npx claude-flow memory` commands
   - Provides manual memory management capabilities

## Configuration

### Project Configuration File

The system uses `.claude/project_config.json` to define project-specific settings:

```json
{
  "project": {
    "id": "flowed",
    "name": "Claude Flow Project",
    "version": "1.0.0"
  },
  "memory": {
    "namespace": "flowed",
    "persistence": {
      "enabled": true,
      "backend": "claude-flow",
      "ttl": 2592000,
      "maxSize": "100MB"
    },
    "semantic": {
      "enabled": true,
      "indexing": true,
      "contextDepth": 5
    },
    "categories": {
      "project": { "priority": 1 },
      "architecture": { "priority": 2 },
      "patterns": { "priority": 3 },
      "tasks": { "priority": 4 },
      "errors": { "priority": 5 },
      "optimization": { "priority": 6 }
    }
  },
  "hooks": {
    "memory": {
      "enabled": true,
      "autoCapture": true,
      "captureThreshold": 0.7,
      "semanticAnalysis": true,
      "lifecycle": {
        "pre_tool": true,
        "post_tool": true,
        "session_start": true,
        "session_end": true,
        "user_prompt": true
      }
    }
  }
}
```

## Memory Categories

The system organizes memories into semantic categories:

1. **project** - Project-specific configurations and patterns
2. **architecture** - Architectural decisions and system design
3. **patterns** - Development patterns and best practices
4. **tasks** - Task history and execution patterns
5. **errors** - Error patterns and resolutions
6. **optimization** - Performance optimizations and improvements

## Hook Lifecycle Integration

### Session Start
- Initializes memory integration
- Loads relevant memories from previous sessions
- Establishes project namespace context

### Pre-Tool Execution
- Captures tool invocation context
- Analyzes semantic significance
- Stores relevant tool patterns

### Post-Tool Execution
- Captures execution results
- Records performance metrics
- Identifies error patterns

### Session End
- Persists session memories
- Syncs with claude-flow backend
- Generates session summary

## Semantic Analysis

The system uses semantic analysis to determine memory significance:

```python
# Semantic scoring factors:
- Keyword matching
- Category relevance
- Pattern frequency
- Error indicators
- Optimization opportunities

# Capture threshold: 0.7 (configurable)
```

## Memory Retrieval Patterns

### Contextual Retrieval
```python
# Get memories relevant to current operation
memories = await retrieval.get_contextual_memories(
    tool_name="Write",
    tool_input={"file_path": "/src/app.py"},
    max_results=10
)
```

### Error Prevention
```python
# Get historical error patterns
errors = await retrieval.get_error_prevention_memories(
    tool_name="Bash",
    tool_input={"command": "rm -rf"}
)
```

### Optimization Patterns
```python
# Get successful optimization patterns
optimizations = await retrieval.get_optimization_patterns(
    operation_type="file_write"
)
```

## CLI Usage

### Store Memory
```bash
python3 .claude/hooks/modules/memory/claude_flow_cli.py store \
  --key "architecture/decision/auth" \
  --value '{"decision": "Use JWT for authentication"}' \
  --category "architecture" \
  --ttl 86400
```

### Retrieve Memory
```bash
python3 .claude/hooks/modules/memory/claude_flow_cli.py get \
  --key "architecture/decision/auth"
```

### Search Memories
```bash
python3 .claude/hooks/modules/memory/claude_flow_cli.py search \
  --pattern "authentication" \
  --category "architecture"
```

### Get Namespace Info
```bash
python3 .claude/hooks/modules/memory/claude_flow_cli.py namespace
```

## Integration with Claude Flow MCP

The system integrates with claude-flow MCP tools:

```javascript
// Store memory via MCP
mcp__claude-flow__memory_usage({
  action: "store",
  key: "project/pattern/concurrent",
  value: "Always batch operations",
  namespace: "flowed",
  ttl: 86400
})

// Retrieve memory via MCP
mcp__claude-flow__memory_usage({
  action: "retrieve",
  key: "project/pattern/concurrent",
  namespace: "flowed"
})
```

## Performance Optimization

1. **Local Caching** - Frequently accessed memories are cached locally
2. **Async Operations** - All memory operations are asynchronous
3. **Semantic Hashing** - Prevents duplicate memory storage
4. **Bounded Storage** - Automatic cleanup of old memories

## Security Considerations

1. **Namespace Isolation** - Each project has isolated memory space
2. **No Sensitive Data** - Never store passwords, tokens, or secrets
3. **TTL Enforcement** - Memories expire after configured TTL
4. **Access Control** - Memory access limited to project hooks

## Best Practices

1. **Semantic Keys** - Use descriptive, hierarchical keys
   ```
   Good: "architecture/auth/jwt_implementation"
   Bad: "mem1" or "temp"
   ```

2. **Category Usage** - Always specify appropriate category
3. **TTL Management** - Set appropriate TTL for memory lifecycle
4. **Error Patterns** - Capture and learn from errors
5. **Optimization Tracking** - Record successful optimizations

## Troubleshooting

### Memory Not Persisting
- Check project configuration exists
- Verify namespace is correctly set
- Ensure hooks have write permissions

### Search Not Working
- Verify claude-flow MCP is installed
- Check namespace matches project
- Ensure semantic indexing is enabled

### High Memory Usage
- Review TTL settings
- Check for memory leaks in patterns
- Use bounded storage limits

## Future Enhancements

1. **Vector Embeddings** - Semantic search using embeddings
2. **Cross-Project Sharing** - Controlled memory sharing
3. **Memory Compression** - Automatic compression for large memories
4. **Visual Memory Map** - Graphical representation of memory relationships
5. **ML-based Retrieval** - Machine learning for smarter retrieval

## Summary

The Claude Flow Memory Integration system provides a robust foundation for project-specific memory management within the Claude hooks ecosystem. By leveraging semantic analysis, intelligent retrieval patterns, and seamless MCP integration, it enables hooks to learn from past operations and continuously improve their effectiveness.