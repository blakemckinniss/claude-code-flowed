# ZEN Workflow Orchestrator - Phase 2 Implementation

## Overview

The ZEN Workflow Orchestrator is a sophisticated workflow state management system that enables intelligent orchestration of multi-step development patterns. It integrates seamlessly with the existing ConversationThread metadata system to provide persistent workflow state tracking.

## Key Features

### 🎯 Pattern Detection
- **Design→Review→Implement**: Detects collaborative development workflows
- **Analyze→Plan→Execute**: Identifies systematic problem-solving patterns  
- **Debug→Investigate→Fix**: Recognizes troubleshooting workflows
- **Semantic Matching**: Advanced keyword matching with semantic understanding
- **Caching**: Performance optimization through pattern result caching

### 🔄 State Management
- **State Machine**: Robust state transition validation with confidence thresholds
- **Workflow States**: INITIAL → DESIGNING → REVIEWING → IMPLEMENTING → TESTING → COMPLETED
- **Transition Rules**: Configurable trigger patterns and validation rules
- **Context Awareness**: State transitions based on user prompts and workflow context

### 🎭 Orchestration
- **Multi-step Planning**: Automatic breakdown of complex workflows into manageable steps
- **Tool Recommendations**: Context-aware suggestions for ZEN and Claude Flow tools
- **Dependency Tracking**: Step dependencies and execution order management
- **Customization**: Workflow adaptation based on complexity and team size

### 🧵 Integration
- **ConversationThread Metadata**: Persistent workflow state storage
- **Memory Manager Integration**: Seamless integration with ZEN memory system
- **Hook System Compatible**: Designed for Claude Hook system integration
- **Singleton Pattern**: Global access through get_zen_workflow_orchestrator()

## Architecture

```
┌─────────────────────────────────────────┐
│           ZEN Workflow Orchestrator     │
├─────────────────────────────────────────┤
│  Pattern Detection Engine               │
│  • Keyword matching                     │
│  • Semantic analysis                    │
│  • Confidence scoring                   │
├─────────────────────────────────────────┤
│  State Management System                │
│  • Workflow states                      │
│  • Transition validation                │
│  • Context tracking                     │
├─────────────────────────────────────────┤
│  Orchestration Engine                   │
│  • Step generation                      │
│  • Tool recommendations                 │
│  • Dependency management                │
├─────────────────────────────────────────┤
│  Integration Layer                      │
│  • ConversationThread metadata         │
│  • Memory manager integration           │
│  • Hook system compatibility            │
└─────────────────────────────────────────┘
```

## Usage Examples

### Basic Pattern Detection
```python
from zen_workflow_orchestrator import get_zen_workflow_orchestrator

orchestrator = get_zen_workflow_orchestrator()

# Detect workflow pattern
prompt = "Let's design then review and implement user authentication"
workflow_type = orchestrator.detect_workflow_pattern(prompt)
print(f"Detected: {workflow_type}")  # → design_review_implement
```

### State Transition Management
```python
# Track state transitions
current_state = WorkflowState.INITIAL
next_state = orchestrator.detect_transition(current_state, "let's start designing")
print(f"Transition: {current_state} → {next_state}")  # → initial → designing

# Update conversation metadata
orchestrator.update_conversation_metadata(
    thread_id="conversation-123",
    workflow_state=next_state,
    workflow_data={"complexity": "high"}
)
```

### Multi-step Orchestration
```python
# Generate workflow steps
steps = orchestrator.orchestrate_multi_step(
    WorkflowType.DESIGN_REVIEW_IMPLEMENT,
    context={"complexity": "high", "team_size": 3}
)

for step in steps:
    print(f"Step: {step.name}")
    print(f"Tools: {step.required_tools}")
    print(f"Duration: {step.estimated_duration} minutes")
```

### Workflow Guidance
```python
# Get contextual guidance
guidance = orchestrator.get_workflow_guidance(
    current_state=WorkflowState.DESIGNING,
    workflow_type=WorkflowType.DESIGN_REVIEW_IMPLEMENT
)

print(f"Recommended tools: {guidance['recommended_tools']}")
print(f"Next steps: {guidance['next_steps']}")
print(f"Tips: {guidance['tips']}")
```

## Workflow Types

### 1. Design→Review→Implement
**Trigger Keywords**: "design then review", "plan and implement", "architectural review"
**Steps**: Design → Review → Implement → Test
**Use Case**: Collaborative development with validation gates

### 2. Analyze→Plan→Execute  
**Trigger Keywords**: "analyze then plan", "systematic approach", "research then implement"
**Steps**: Analyze → Plan → Execute
**Use Case**: Problem-solving and systematic development

### 3. Debug→Investigate→Fix
**Trigger Keywords**: "debug then fix", "root cause analysis", "systematic debugging"
**Steps**: Debug → Investigate → Fix
**Use Case**: Troubleshooting and issue resolution

## Integration Points

### ConversationThread Metadata
```python
# Workflow state is stored in thread metadata
thread.metadata["workflow"] = {
    "workflow_state": "designing",
    "updated_at": "2024-08-02T10:30:00Z",
    "state_history": [
        {"state": "initial", "timestamp": "2024-08-02T10:25:00Z"},
        {"state": "designing", "timestamp": "2024-08-02T10:30:00Z"}
    ]
}
```

### Hook System Integration
The orchestrator is designed to be called from hook systems:
```python
# In a hook (e.g., UserPromptSubmit)
orchestrator = get_zen_workflow_orchestrator()
workflow_type = orchestrator.detect_workflow_pattern(user_prompt)

if workflow_type:
    # Inject workflow guidance
    print(f"🎯 ZEN Workflow Detected: {workflow_type.value}")
    steps = orchestrator.orchestrate_multi_step(workflow_type)
    # Guide user through workflow...
```

## Configuration

### Pattern Sensitivity
```python
# Adjust detection thresholds
orchestrator.PATTERN_THRESHOLD = 0.4  # Lower = more sensitive
```

### State Transition Confidence
```python
# Modify transition confidence thresholds
transition.confidence_threshold = 0.4  # Adjust per transition
```

### Custom Workflows
```python
# Add custom workflow patterns
orchestrator.WORKFLOW_PATTERNS[WorkflowType.CUSTOM] = {
    "keywords": ["custom", "pattern", "keywords"],
    "steps": [custom_steps...]
}
```

## Performance Features

- **Pattern Caching**: Results cached for repeated prompts
- **Singleton Pattern**: Single instance for memory efficiency  
- **Lazy Loading**: Components loaded on demand
- **Graceful Degradation**: Works without memory manager integration

## Testing

Run the comprehensive test suite:
```bash
python test_zen_workflow_orchestrator.py
```

Run the demonstration:
```bash
python zen_workflow_orchestrator.py
```

Debug pattern detection:
```bash
python debug_patterns.py
```

## Files

- `zen_workflow_orchestrator.py` - Main implementation
- `test_zen_workflow_orchestrator.py` - Comprehensive test suite
- `debug_patterns.py` - Debug utilities
- `__init__.py` - Module exports
- `README.md` - This documentation

## Next Steps

1. **Hook Integration**: Integrate with UserPromptSubmit and other hooks
2. **Memory Persistence**: Enhanced memory integration for cross-session workflows
3. **Custom Patterns**: Support for user-defined workflow patterns
4. **Analytics**: Workflow success tracking and optimization
5. **UI Integration**: Visual workflow state representation

## Summary

The ZEN Workflow Orchestrator successfully implements Phase 2 of ZEN integration by providing:

✅ **Pattern Detection** - Identifies Design→Review→Implement and other collaborative patterns
✅ **State Management** - Robust workflow state tracking with transition validation  
✅ **Orchestration** - Multi-step workflow breakdown with tool recommendations
✅ **Integration** - Seamless ConversationThread metadata integration
✅ **Extensibility** - Modular design for future workflow pattern additions

The implementation provides the foundation for intelligent workflow guidance in the Claude Hook → ZEN → Claude Flow ecosystem, enabling users to navigate complex development patterns with AI-powered orchestration and state management.