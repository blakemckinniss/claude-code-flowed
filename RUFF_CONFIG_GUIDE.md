# Ruff Configuration Guide for Claude Flow Project

## Overview

This document explains the optimal Ruff configuration created for the Claude Flow project. The configuration in [`pyproject.toml`](pyproject.toml) is specifically tailored for this sophisticated Python project with its complex hook-based architecture, async patterns, and intelligent workflow automation system.

## Configuration Philosophy

The Ruff configuration balances **strict code quality enforcement** with **practical flexibility** for the project's unique patterns:

- **High Standards**: Comprehensive rule coverage for maintainable, secure code
- **Project-Aware**: Custom ignores for legitimate hook patterns and system interactions
- **Performance-Oriented**: Auto-fixing enabled for rapid development cycles
- **Modern Practices**: Python 3.8+ features, type hints, and contemporary conventions

## Key Configuration Sections

### Basic Settings

```toml
line-length = 88          # Black-compatible line length
target-version = "py38"   # Python 3.8+ compatibility
fix = true               # Enable auto-fixing
show-fixes = true        # Display what was fixed
```

**Rationale**: Ensures compatibility with Black formatter and supports modern Python features while maintaining readability.

### Rule Selection Strategy

#### Enabled Rule Categories

| Category | Codes | Purpose |
|----------|-------|---------|
| **Core Quality** | `F`, `E`, `W` | Pyflakes errors, PEP 8 compliance |
| **Import Management** | `I`, `ICN`, `TID` | Proper import organization and conventions |
| **Documentation** | `D` | Google-style docstring enforcement |
| **Type Safety** | `ANN`, `TCH` | Type hint completeness and optimization |
| **Security** | `S`, `BLE` | Bandit security checks, exception handling |
| **Performance** | `PERF`, `C4` | Efficient code patterns |
| **Modern Python** | `UP`, `FLY` | Contemporary syntax and features |
| **Code Quality** | `B`, `SIM`, `RET` | Bugbear patterns, simplifications |

#### Strategic Rule Ignores

The configuration includes carefully chosen ignores for project-specific patterns:

##### Hook-Specific Allowances
```toml
# Allow print statements in hooks (used for communication)
"T201"

# Allow subprocess usage (needed for hook execution)  
"S603", "S607"

# Allow broad exception catching in fallback scenarios
"BLE001"
```

##### Architecture Patterns
```toml
# Allow complex functions for business logic
"PLR0913", "PLR0915", "C901"

# Allow boolean positional arguments for configuration
"FBT001", "FBT002"

# Allow magic values for configuration defaults
"PLR2004"
```

##### Development Flexibility
```toml
# Allow TODO comments during development
"FIX002", "TD002", "TD003"

# Allow assert statements (useful for debugging)
"S101"
```

### Per-File Customization

The configuration provides targeted rules for different file types:

#### Test Files (`**/test_*.py`)
- Relaxed annotation requirements
- Allow magic values and assertions
- Focus on functionality over strict typing

#### Hook Files (`**/*hook*.py`)  
- Allow print statements for communication
- Permit complex branching for hook logic
- Enable system interactions

#### Init Files (`**/__init__.py`)
- Allow unused imports (re-exports)
- Skip docstring requirements
- Support package structure patterns

### Code Quality Thresholds

```toml
[tool.ruff.lint.mccabe]
max-complexity = 12       # Reasonable complexity limit

[tool.ruff.lint.pylint]
max-args = 8             # Accommodate complex initialization
max-branches = 15        # Allow decision trees
max-statements = 60      # Support comprehensive functions
```

**Rationale**: These thresholds balance code quality with the project's sophisticated business logic requirements.

### Import Organization

```toml
[tool.ruff.lint.isort]
known-first-party = ["coordination", "memory", "claude"]
known-local-folder = ["modules", "core", "analyzers", "optimization"]
section-order = [
    "future",
    "standard-library", 
    "third-party",
    "first-party",
    "local-folder"
]
```

**Benefits**: Ensures consistent import ordering that reflects the project's modular architecture.

## Usage Examples

### Running Ruff Checks

```bash
# Check entire project
ruff check .

# Check specific file with full output
ruff check .claude/hooks/pre_tool_use.py --output-format=full

# Auto-fix issues
ruff check . --fix

# Check with unsafe fixes (use carefully)
ruff check . --fix --unsafe-fixes
```

### Integration with Development Workflow

1. **Pre-commit**: Run `ruff check --fix` before commits
2. **CI/CD**: Include `ruff check` in automated pipelines  
3. **IDE Integration**: Enable Ruff in VS Code/PyCharm for real-time feedback
4. **Code Reviews**: Use Ruff output to ensure consistent standards

## Expected Ruff Output

Based on testing, Ruff will:

### ✅ Auto-Fix Common Issues
- Formatting inconsistencies (quotes, whitespace, line length)
- Import ordering and unused imports
- Simple code improvements (f-strings, comprehensions)
- Documentation formatting

### ⚠️ Flag Legitimate Patterns
- Complex functions in hook architecture (intentional)
- Global variable usage for optimization (necessary)
- Subprocess calls for system integration (required)
- Relative imports in modular structure (acceptable)

### 🚨 Catch Real Issues
- Type annotation inconsistencies
- Security vulnerabilities
- Performance anti-patterns
- Logic errors and unused variables

## Maintenance Guidelines

### When to Update Configuration

1. **Adding New Modules**: Update `known-first-party` imports
2. **Architecture Changes**: Adjust per-file ignores if patterns change
3. **Python Version Upgrades**: Update `target-version`
4. **New Ruff Versions**: Review new rules and add appropriate ignores

### Configuration Validation

Regularly validate the configuration effectiveness:

```bash
# Check configuration syntax
ruff check --show-settings

# Test on representative files
ruff check .claude/hooks/pre_tool_use.py
ruff check .claude/hooks/modules/pre_tool/analyzers/neural_pattern_validator.py
ruff check .claude/hooks/session_end.py
```

### Customization for Team Preferences

The configuration can be adjusted for team preferences:

- **Strictness Level**: Remove ignores for stricter enforcement
- **Docstring Style**: Change from Google to NumPy/Sphinx if preferred
- **Line Length**: Adjust from 88 to 80/100 based on team standards
- **Complexity Limits**: Tighten thresholds for simpler codebases

## Integration with Other Tools

This Ruff configuration is designed to work harmoniously with:

- **Black**: Compatible line length and formatting style
- **mypy**: Type checking complements Ruff's type rules
- **pytest**: Test-specific rules support testing patterns
- **pre-commit**: Ready for pre-commit hook integration

## Performance Considerations

The configuration is optimized for performance:

- **Targeted Exclusions**: Skip cache and build directories
- **Efficient Rules**: Focus on high-value checks
- **Auto-fixing**: Reduces manual correction time
- **Parallel Execution**: Ruff's Rust implementation provides fast checking

## Conclusion

This Ruff configuration provides enterprise-grade code quality assurance while respecting the Claude Flow project's sophisticated architecture. It ensures consistency, catches real issues, and maintains development velocity through intelligent rule selection and auto-fixing capabilities.

For questions or suggestions about the configuration, refer to this guide and the inline comments in [`pyproject.toml`](pyproject.toml).