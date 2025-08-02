#!/usr/bin/env python3
"""Concurrent Execution Analyzer for Claude Code.

Detects sequential operations and provides real-time guidance
for batching operations in single messages.
"""

import re
from typing import List, Dict, Any, Optional
from modules.core import Analyzer, PatternMatch


class ConcurrentExecutionAnalyzer(Analyzer):
    """Enforces concurrent execution patterns."""
    
    def __init__(self):
        super().__init__()
        self.name = "concurrent_execution"
        self.priority = 100  # High priority
        
        # Track recent operations to detect sequential patterns
        self._recent_operations = []
        self._operation_window = 5  # Look at last 5 operations
        
    def analyze(self, prompt: str) -> List[PatternMatch]:
        """Analyze prompt for sequential operation patterns."""
        matches = []
        
        # Detect TodoWrite patterns
        if re.search(r'todowrite|todo\s*write|update.*todo', prompt, re.IGNORECASE):
            if self._is_single_todo(prompt):
                matches.append(PatternMatch(
                    analyzer=self.name,
                    pattern="single_todo",
                    confidence=0.95,
                    metadata={
                        "message": "🚨 BATCH VIOLATION DETECTED!\n\n"
                                   "❌ WRONG: Single TodoWrite operation\n"
                                   "✅ CORRECT: Include 5-10+ todos in ONE TodoWrite call\n\n"
                                   "Example:\n"
                                   "```javascript\n"
                                   "TodoWrite { todos: [\n"
                                   "  {id: '1', content: 'Task 1', status: 'in_progress', priority: 'high'},\n"
                                   "  {id: '2', content: 'Task 2', status: 'pending', priority: 'high'},\n"
                                   "  {id: '3', content: 'Task 3', status: 'pending', priority: 'medium'},\n"
                                   "  // ... 5-10+ todos total\n"
                                   "]}\n"
                                   "```",
                        "category": "batch_violation",
                        "severity": "high"
                    }
                ))
        
        # Detect Task tool patterns
        if re.search(r'task\s*\(|spawn.*agent|create.*agent', prompt, re.IGNORECASE):
            if self._is_single_agent(prompt):
                matches.append(PatternMatch(
                    analyzer=self.name,
                    pattern="single_agent",
                    confidence=0.95,
                    metadata={
                        "message": "⚡ CONCURRENT EXECUTION REQUIRED!\n\n"
                                   "❌ WRONG: Spawning single agent\n"
                                   "✅ CORRECT: Spawn ALL agents in ONE message\n\n"
                                   "Example:\n"
                                   "```javascript\n"
                                   "[Single Message]:\n"
                                   "  - Task('Agent 1', 'full instructions', 'agent-type-1')\n"
                                   "  - Task('Agent 2', 'full instructions', 'agent-type-2')\n"
                                   "  - Task('Agent 3', 'full instructions', 'agent-type-3')\n"
                                   "  - Task('Agent 4', 'full instructions', 'agent-type-4')\n"
                                   "```",
                        "category": "batch_violation",
                        "severity": "high"
                    }
                ))
        
        # Detect file operations
        if re.search(r'read\s*\(|write\s*\(|edit\s*\(', prompt, re.IGNORECASE):
            if self._is_single_file_op(prompt):
                matches.append(PatternMatch(
                    analyzer=self.name,
                    pattern="single_file_op",
                    confidence=0.9,
                    metadata={
                        "message": "📦 BATCH FILE OPERATIONS!\n\n"
                                   "Combine multiple file operations in ONE message:\n"
                                   "- Read 10 files? → One message with 10 Read calls\n"
                                   "- Write 5 files? → One message with 5 Write calls\n"
                                   "- Edit 1 file many times? → One MultiEdit call",
                        "category": "optimization",
                        "severity": "medium"
                    }
                ))
        
        # Golden Rule reminder
        if matches:
            matches.append(PatternMatch(
                analyzer=self.name,
                pattern="golden_rule",
                confidence=1.0,
                metadata={
                    "message": "⚡ GOLDEN RULE: '1 MESSAGE = ALL RELATED OPERATIONS'\n\n"
                               "Before sending ANY message, ask yourself:\n"
                               "✅ Are ALL TodoWrite operations batched?\n"
                               "✅ Are ALL Task spawning operations together?\n"
                               "✅ Are ALL file operations batched?\n"
                               "✅ Are ALL bash commands grouped?\n"
                               "✅ Are ALL memory operations concurrent?",
                    "category": "reminder",
                    "severity": "info"
                }
            ))
        
        return matches
    
    def _is_single_todo(self, prompt: str) -> bool:
        """Check if prompt suggests single todo operation."""
        # Look for patterns suggesting single todo
        single_patterns = [
            r'single\s*todo',
            r'one\s*todo',
            r'add\s*a\s*todo',
            r'update\s*this\s*todo',
            r'todo.*:\s*["\']([^"\']+)["\']'  # Single quoted todo
        ]
        return any(re.search(p, prompt, re.IGNORECASE) for p in single_patterns)
    
    def _is_single_agent(self, prompt: str) -> bool:
        """Check if prompt suggests single agent spawn."""
        # Look for patterns suggesting single agent
        single_patterns = [
            r'spawn\s*an?\s*agent',
            r'create\s*one\s*agent',
            r'single\s*agent',
            r'task\s*\(\s*["\']([^"\']+)["\']'  # Single task call
        ]
        return any(re.search(p, prompt, re.IGNORECASE) for p in single_patterns)
    
    def _is_single_file_op(self, prompt: str) -> bool:
        """Check if prompt suggests single file operation."""
        # Count file operation patterns
        ops = re.findall(r'(read|write|edit)\s*\(', prompt, re.IGNORECASE)
        return len(ops) == 1