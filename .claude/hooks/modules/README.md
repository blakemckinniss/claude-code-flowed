# Claude Hook Module System

This modular hook system provides intelligent context injection for Claude Code based on pattern matching in user prompts.

## Architecture

```
modules/
├── core/                  # Core components
│   ├── analyzer.py       # Base analyzer class
│   ├── context_builder.py # Context construction
│   └── config.py         # Configuration management
├── patterns/             # Pattern analyzers
│   ├── development.py    # Development patterns
│   ├── github.py        # GitHub workflow patterns
│   ├── testing.py       # Testing patterns
│   ├── performance.py   # Performance patterns
│   └── swarm.py        # Swarm coordination patterns
├── analyzers/           # Analyzer management
│   └── manager.py      # Coordinates multiple analyzers
└── utils/              # Utilities
    ├── helpers.py      # Common helper functions
    └── custom_patterns.py # Custom pattern support
```

## Features

### 1. **Modular Pattern Analyzers**
Each analyzer focuses on specific domains:
- **SwarmAnalyzer**: Detects coordination and parallel execution needs
- **DevelopmentAnalyzer**: Identifies development tasks (API, frontend, database)
- **GitHubAnalyzer**: Recognizes GitHub workflows (PR, issues, releases)
- **TestingAnalyzer**: Spots testing requirements (TDD, unit, integration)
- **PerformanceAnalyzer**: Finds optimization opportunities

### 2. **Priority-Based Pattern Matching**
- Patterns are matched with priorities (0-100)
- Higher priority patterns appear first in context
- Deduplication prevents redundant suggestions

### 3. **Custom Pattern Support**
Add your own patterns via `hook_config.json`:
```json
{
  "custom_patterns": [
    {
      "pattern": "(docker|container|kubernetes)",
      "message": "🐳 CONTAINERIZATION DETECTED\n• Use specific suggestions...",
      "metadata": {
        "category": "containers",
        "suggested_agents": ["backend-dev", "cicd-engineer"]
      }
    }
  ]
}
```

### 4. **Extensible Architecture**
Create new analyzers by extending the base `Analyzer` class:
```python
from modules.core import Analyzer

class MyAnalyzer(Analyzer):
    def get_name(self) -> str:
        return "myanalyzer"
    
    def _initialize_patterns(self) -> None:
        self.add_pattern(
            r"(my|pattern)",
            "My context message",
            {"category": "custom"}
        )
```

## Configuration

The system uses `hook_config.json` for configuration:

```json
{
  "enabled_analyzers": ["swarm", "development", "github", "testing", "performance"],
  "quick_tips": {
    "enabled": true,
    "message": "💡 Quick tips..."
  },
  "deduplication": true,
  "custom_pattern_priority": 75,
  "custom_patterns": []
}
```

## Usage

The hook is automatically invoked by Claude Code when processing user prompts. To debug:

```bash
export CLAUDE_HOOKS_DEBUG=1
```

## Adding New Features

### 1. **New Pattern Analyzer**
1. Create a new file in `modules/patterns/`
2. Extend the `Analyzer` class
3. Add to `patterns/__init__.py`
4. Register in `AnalyzerManager.ANALYZER_REGISTRY`

### 2. **New Context Format**
Modify `ContextBuilder` in `core/context_builder.py`

### 3. **New Configuration Options**
1. Update `Config._get_default_config()` 
2. Document in `hook_config.json`

## Pattern Writing Tips

1. **Be Specific**: Target clear keywords relevant to the domain
2. **Provide Value**: Include actionable suggestions and agent recommendations
3. **Use Metadata**: Store additional context for future features
4. **Consider Priority**: More specific patterns should have higher priority

## Future Enhancements

- **Context Learning**: Track which patterns are most useful
- **Dynamic Priority**: Adjust priorities based on user behavior
- **Pattern Composition**: Combine multiple patterns for complex scenarios
- **Integration Hooks**: Connect with other Claude Code features