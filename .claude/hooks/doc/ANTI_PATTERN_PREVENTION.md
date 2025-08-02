# Anti-Pattern Prevention System

## Overview

The Anti-Pattern Prevention System is a comprehensive set of pre-tool validators designed to prevent common dangerous patterns that AI coding assistants often make. These validators run before any file operation to detect and block potentially harmful actions.

## Validators

### 1. Duplication Detection Validator (Priority: 860)
**Purpose**: Prevents AI from creating duplicate code/files

**Detects**:
- Files with duplicate suffixes (`_copy`, `_new`, `_v2`, etc.)
- Functions with duplicate prefixes (`new_`, `my_`, `custom_`, etc.)
- Similar file names that might indicate duplication
- Task descriptions suggesting reimplementation

**Example Blocked Operations**:
```bash
# Creating duplicate files
Write { file_path: "auth_new.js" }  # Blocked - suggests duplication

# Functions with duplicate prefixes
Edit { content: "function new_authenticate() {...}" }  # Warning
```

### 2. Rogue System Validator (Priority: 890)
**Purpose**: Prevents creation of parallel/competing systems

**Detects**:
- New frameworks alongside existing ones
- Redundant service/API layers
- Duplicate state management systems
- Competing authentication systems

**Monitored Systems**:
- State management (Redux, MobX, Context API)
- Authentication (Auth, JWT, OAuth)
- Routing systems
- API layers
- Database layers
- Configuration systems
- Logging systems
- Event systems

**Example Blocked Operations**:
```bash
# Creating competing state management
Write { file_path: "newStateManager.js" }  # Blocked if Redux exists

# Building parallel API layer
Task { description: "Create new API service" }  # Warning if API exists
```

### 3. Conflicting Architecture Validator (Priority: 880)
**Purpose**: Ensures consistency with project patterns

**Detects**:
- Mixed module systems (ES6 vs CommonJS)
- Inconsistent async patterns (callbacks vs promises vs async/await)
- Naming convention violations
- Framework mixing (React + Vue, Jest + Mocha)
- API paradigm conflicts (REST vs GraphQL)

**Example Blocked Operations**:
```javascript
// Mixing module systems
const x = require('module');  // Blocked in ES6 project
import y from 'module';       // Blocked in CommonJS project

// Framework conflicts
import Vue from 'vue';        // Blocked in React project
```

### 4. Overwrite Protection Validator (Priority: 940)
**Purpose**: Prevents accidental overwrites of critical files

**Critical Protected Files**:
- `package.json`, `package-lock.json`
- `.env` files
- `tsconfig.json`, `webpack.config.js`
- Docker configurations
- Dependency files (`requirements.txt`, `Cargo.toml`, etc.)

**Protection Mechanisms**:
- Blocks overwrites of critical files
- Warns when overwriting without reading first
- Detects complete file replacements
- Validates batch operations

**Example Blocked Operations**:
```bash
# Overwriting critical file
Write { file_path: "package.json" }  # Blocked

# Overwriting without reading
Write { file_path: "config.js" }     # Warning if not read first

# Dangerous deletions
Bash { command: "rm -rf src/*" }     # Blocked
```

## Configuration

The validators can be configured in `.claude/hooks/pre_tool_config.json`:

```json
{
  "enabled_validators": [
    "duplication_detection_validator",
    "rogue_system_validator",
    "conflicting_architecture_validator",
    "overwrite_protection_validator"
  ],
  "validation_settings": {
    "block_dangerous_operations": true,
    "suggest_optimizations": true,
    "require_zen_for_complex": true
  },
  "blocking_behavior": {
    "block_on_critical": true,
    "block_on_dangerous": true,
    "allow_overrides": false
  }
}
```

## Severity Levels

1. **CRITICAL** (Exit code 2): Operation blocked, would cause severe damage
2. **BLOCK** (Exit code 2): Operation blocked, violates project patterns
3. **WARN** (Exit code 0): Operation allowed with warning
4. **SUGGEST** (Exit code 0): Operation allowed with optimization suggestion
5. **ALLOW** (Exit code 0): Operation optimal, proceed

## Integration with Hive System

The anti-pattern validators integrate seamlessly with the existing Queen ZEN hierarchy:

1. **Queen ZEN** provides strategic oversight
2. **Anti-pattern validators** enforce tactical safety
3. **Flow Workers** coordinate safe operations
4. **Storage Workers** execute validated operations

## Usage Examples

### Example 1: Preventing Duplication
```bash
# AI attempts to create a duplicate auth system
> Write { file_path: "auth_v2.js" }

🚨 DUPLICATION DETECTED: File 'auth_v2.js' appears to be a duplicate
💡 Suggested: Update the existing file instead of creating a duplicate
```

### Example 2: Blocking Rogue Systems
```bash
# AI tries to create parallel state management
> Write { file_path: "myStateManager.js", content: "class StateManager {...}" }

🚨 ROGUE SYSTEM DETECTED: State management system already exists
💡 Suggested: Extend existing Redux store instead of creating a new one
```

### Example 3: Maintaining Architecture
```bash
# AI mixes module systems
> Edit { content: "const x = require('express')" }  # In ES6 project

🚨 ARCHITECTURE CONFLICT: Using CommonJS require() in ES6 module project
💡 Suggested: Use ES6 import syntax: import express from 'express'
```

### Example 4: Protecting Critical Files
```bash
# AI attempts to overwrite package.json
> Write { file_path: "package.json" }

🚨 CRITICAL FILE OVERWRITE BLOCKED: 'package.json' - Project dependencies
💡 Suggested: Use Edit to modify specific parts or create a backup first
```

## Best Practices

1. **Always Read Before Write**: The system tracks file reads to ensure understanding
2. **Work Within Patterns**: Extend existing systems rather than creating new ones
3. **Incremental Changes**: Use Edit for modifications rather than full overwrites
4. **Batch Operations**: Group related file operations for better validation

## Troubleshooting

### False Positives
If a validator incorrectly blocks an operation:
1. Check if the operation truly needs to violate the pattern
2. Consider alternative approaches suggested by the validator
3. Use Queen ZEN coordination for complex architectural decisions

### Performance Impact
The validators run before tool execution with minimal overhead:
- Pattern matching: < 1ms
- File system checks: < 5ms per file
- Total validation time: < 20ms typical

## Override Mechanism

The system includes a controlled override mechanism for cases where violations are intentional and necessary.

### How to Override

Include one of these keywords with justification in your tool input:

```bash
# Override keywords:
FORCE: <justification>
OVERRIDE: <justification>  
BYPASS: <justification>
QUEEN_ZEN_APPROVED: <justification>  # For CRITICAL severity
EMERGENCY: <justification>            # For urgent fixes
HOTFIX: <justification>              # For production issues
```

### Examples

```bash
# Override duplication check
Write { 
  file_path: "auth_v2.js",
  content: "// OVERRIDE: Creating v2 for breaking API changes that require separate implementation"
}

# Override critical file protection
Write {
  file_path: "package.json", 
  content: '{"name": "project", "version": "2.0.0"} // QUEEN_ZEN_APPROVED: Major version bump with breaking changes'
}

# Override architecture conflict
Edit {
  file_path: "legacy.js",
  content: "const x = require('old-module') // FORCE: Legacy code requires CommonJS until migration complete"
}
```

### Override Rules

1. **Justification Required**: Must provide meaningful explanation (min 10 characters)
2. **Severity Restrictions**:
   - CRITICAL: Requires `QUEEN_ZEN_APPROVED` or `EMERGENCY`
   - BLOCK: Standard override keywords work
   - WARN/SUGGEST: Overrides always allowed
3. **Non-Overridable**: Safety validators cannot be bypassed
4. **Logging**: All overrides are logged to `.claude/hooks/logs/overrides.json`

### Override Statistics

The system tracks override usage:
- Total overrides per session
- Most overridden validators
- Common justification patterns
- Override frequency by severity

This data helps improve validator accuracy and identify legitimate use cases.

## Future Enhancements

Planned improvements:
1. Machine learning for pattern detection
2. Project-specific pattern learning
3. Cross-file dependency validation
4. Semantic code analysis
5. Integration with code review systems
6. Smart override learning - reduce false positives based on override patterns