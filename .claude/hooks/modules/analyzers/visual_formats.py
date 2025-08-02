#!/usr/bin/env python3
"""Visual Format Templates for Claude Code.

Provides consistent visual formatting for progress tracking and status displays.
"""

import re
from typing import List, Dict, Any, Optional
from modules.core import Analyzer, PatternMatch


class VisualFormatsAnalyzer(Analyzer):
    """Provides visual format templates and guidance."""
    
    def __init__(self):
        super().__init__()
        self.name = "visual_formats"
        self.priority = 70
        
    def analyze(self, prompt: str) -> List[PatternMatch]:
        """Detect when visual formats might be helpful."""
        matches = []
        prompt_lower = prompt.lower()
        
        # Task progress tracking
        if any(term in prompt_lower for term in ['progress', 'status', 'task', 'todo']):
            matches.append(PatternMatch(
                analyzer=self.name,
                pattern="task_progress",
                confidence=0.8,
                metadata={
                    "message": self._get_task_progress_format(),
                    "category": "visual_format",
                    "format_type": "task_progress"
                }
            ))
        
        # Swarm status display
        if any(term in prompt_lower for term in ['swarm', 'agent', 'coordination']):
            matches.append(PatternMatch(
                analyzer=self.name,
                pattern="swarm_status",
                confidence=0.8,
                metadata={
                    "message": self._get_swarm_status_format(),
                    "category": "visual_format",
                    "format_type": "swarm_status"
                }
            ))
        
        return matches
    
    def _get_task_progress_format(self) -> str:
        """Get task progress visual format template."""
        return """📊 VISUAL TASK TRACKING FORMAT

Use this format when displaying task progress:

```
📊 Progress Overview
   ├── Total Tasks: X
   ├── ✅ Completed: X (X%)
   ├── 🔄 In Progress: X (X%)
   ├── ⭕ Todo: X (X%)
   └── ❌ Blocked: X (X%)

📋 Todo (X)
   └── 🔴 001: [Task description] [PRIORITY] ▶

🔄 In progress (X)
   ├── 🟡 002: [Task description] ↳ X deps ▶
   └── 🔴 003: [Task description] [PRIORITY] ▶

✅ Completed (X)
   ├── ✅ 004: [Task description]
   └── ... (more completed tasks)

Priority indicators: 🔴 HIGH/CRITICAL, 🟡 MEDIUM, 🟢 LOW
Dependencies: ↳ X deps | Actionable: ▶
```"""
    
    def _get_swarm_status_format(self) -> str:
        """Get swarm status visual format template."""
        return """🎨 VISUAL SWARM STATUS FORMAT

Use this format when showing swarm status:

```
🐝 Swarm Status: ACTIVE
├── 🏗️ Topology: hierarchical
├── 👥 Agents: 6/8 active
├── ⚡ Mode: parallel execution
├── 📊 Tasks: 12 total (4 complete, 6 in-progress, 2 pending)
└── 🧠 Memory: 15 coordination points stored

Agent Activity:
├── 🟢 architect: Designing database schema...
├── 🟢 coder-1: Implementing auth endpoints...
├── 🟢 coder-2: Building user CRUD operations...
├── 🟢 analyst: Optimizing query performance...
├── 🟡 tester: Waiting for auth completion...
└── 🟢 coordinator: Monitoring progress...
```

Status indicators:
- 🟢 Active and working
- 🟡 Waiting/Idle
- 🔴 Error/Blocked
- ⚫ Not started"""
    
    def get_format_template(self, format_type: str) -> Optional[str]:
        """Get a specific format template."""
        templates = {
            "task_progress": self._get_task_progress_format(),
            "swarm_status": self._get_swarm_status_format(),
            "memory_coordination": self._get_memory_format(),
            "batch_operations": self._get_batch_format()
        }
        return templates.get(format_type)
    
    def _get_memory_format(self) -> str:
        """Get memory coordination format."""
        return """🔄 MEMORY COORDINATION PATTERN

Every agent coordination step MUST use memory:

```javascript
// After each major decision or implementation
mcp__claude-flow__memory_usage {
  action: "store",
  key: "swarm-{id}/agent-{name}/{step}",
  value: {
    timestamp: Date.now(),
    decision: "what was decided",
    implementation: "what was built",
    nextSteps: ["step1", "step2"],
    dependencies: ["dep1", "dep2"]
  }
}

// To retrieve coordination data
mcp__claude-flow__memory_usage {
  action: "retrieve",
  key: "swarm-{id}/agent-{name}/{step}"
}

// To check all swarm progress
mcp__claude-flow__memory_usage {
  action: "list",
  pattern: "swarm-{id}/*"
}
```"""
    
    def _get_batch_format(self) -> str:
        """Get batch operations format."""
        return """📦 BATCH OPERATIONS FORMAT

Group operations by type in single messages:

**File Operations:**
```javascript
// ✅ CORRECT - One message
Read("file1.js")
Read("file2.js")
Write("output1.js", content1)
Write("output2.js", content2)
Edit("config.js", oldText, newText)
```

**Command Operations:**
```javascript
// ✅ CORRECT - One message
Bash("mkdir -p app/{src,tests,docs}")
Bash("npm install")
Bash("npm test")
Bash("npm run build")
```

**Agent Operations:**
```javascript
// ✅ CORRECT - One message
Task("Agent 1", "instructions", "type-1")
Task("Agent 2", "instructions", "type-2")
Task("Agent 3", "instructions", "type-3")
TodoWrite({ todos: [/* 5-10+ todos */] })
```"""