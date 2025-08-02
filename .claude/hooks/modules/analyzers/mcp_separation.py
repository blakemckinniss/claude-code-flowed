#!/usr/bin/env python3
"""MCP Tool Separation Analyzer for Claude Code.

Enforces the critical separation between MCP coordination tools
and Claude Code execution tools.
"""

import re
from typing import List, Dict, Any, Optional
from modules.core import Analyzer, PatternMatch


class MCPSeparationAnalyzer(Analyzer):
    """Enforces MCP tool vs Claude Code execution separation."""
    
    def __init__(self):
        super().__init__()
        self.name = "mcp_separation"
        self.priority = 95  # Very high priority
        
        # MCP tools that should NEVER do execution
        self.mcp_coordination_tools = {
            "mcp__claude-flow__": "coordination and planning",
            "mcp__zen__": "orchestration and analysis",
            "mcp__filesystem__": "file system navigation",
            "mcp__github__": "GitHub API operations"
        }
        
        # Claude Code execution tools
        self.execution_tools = {
            "Read": "file reading",
            "Write": "file writing",
            "Edit": "file editing",
            "MultiEdit": "multiple file edits",
            "Bash": "command execution",
            "TodoWrite": "task management",
            "Task": "agent spawning",
            "Grep": "file searching",
            "Glob": "pattern matching"
        }
        
    def analyze(self, prompt: str) -> List[PatternMatch]:
        """Detect and correct tool usage violations."""
        matches = []
        prompt_lower = prompt.lower()
        
        # Check for MCP execution attempts
        if self._detects_mcp_execution(prompt):
            matches.append(PatternMatch(
                analyzer=self.name,
                pattern="mcp_execution_violation",
                confidence=0.95,
                metadata={
                    "message": "🚨 CRITICAL SEPARATION VIOLATION!\n\n"
                               "❌ MCP Tools NEVER execute - they only coordinate!\n"
                               "✅ Claude Code ALWAYS executes - all actual work!\n\n"
                               "REMEMBER:\n"
                               "• MCP tools = Brain (planning)\n"
                               "• Claude Code = Hands (execution)\n\n"
                               "NEVER use MCP tools for:\n"
                               "- File operations (use Read, Write, Edit)\n"
                               "- Bash commands (use Bash tool)\n"
                               "- Code generation (use Write tool)\n"
                               "- TodoWrite (use TodoWrite tool)",
                    "category": "critical_violation",
                    "severity": "critical"
                }
            ))
        
        # Check for workflow patterns
        if 'workflow' in prompt_lower or 'swarm' in prompt_lower:
            matches.append(PatternMatch(
                analyzer=self.name,
                pattern="workflow_pattern",
                confidence=0.9,
                metadata={
                    "message": self._get_correct_workflow_pattern(),
                    "category": "workflow_guidance",
                    "severity": "info"
                }
            ))
        
        # Tool usage reminder
        if any(tool in prompt for tool in ['mcp__', 'Task', 'TodoWrite']):
            matches.append(PatternMatch(
                analyzer=self.name,
                pattern="tool_usage_reminder",
                confidence=0.8,
                metadata={
                    "message": self._get_tool_usage_guide(),
                    "category": "tool_guidance",
                    "severity": "info"
                }
            ))
        
        return matches
    
    def _detects_mcp_execution(self, prompt: str) -> bool:
        """Detect if prompt suggests MCP tools doing execution."""
        violation_patterns = [
            r'mcp__.*__(write|create|execute|run|build)',
            r'mcp__.*terminal.*execute',
            r'mcp__.*file.*create',
            r'claude-flow.*write.*file',
            r'zen.*execute.*code'
        ]
        return any(re.search(p, prompt, re.IGNORECASE) for p in violation_patterns)
    
    def _get_correct_workflow_pattern(self) -> str:
        """Get the correct workflow pattern."""
        return """🔄 CORRECT WORKFLOW EXECUTION PATTERN

✅ CORRECT Workflow:
1. **MCP**: `mcp__claude-flow__swarm_init` (coordination setup)
2. **MCP**: `mcp__claude-flow__agent_spawn` (planning agents)
3. **MCP**: `mcp__claude-flow__task_orchestrate` (task coordination)
4. **Claude Code**: `Task` tool to spawn agents with instructions
5. **Claude Code**: `TodoWrite` with ALL todos batched (5-10+ in ONE call)
6. **Claude Code**: `Read`, `Write`, `Edit`, `Bash` (actual work)
7. **MCP**: `mcp__claude-flow__memory_usage` (store results)

❌ WRONG Workflow:
1. MCP doing file operations (DON'T DO THIS)
2. MCP executing commands (DON'T DO THIS)
3. Sequential Task calls (DON'T DO THIS)
4. Individual TodoWrite calls (DON'T DO THIS)"""
    
    def _get_tool_usage_guide(self) -> str:
        """Get tool usage guide."""
        return """🎯 CLAUDE CODE IS THE ONLY EXECUTOR

✅ Claude Code ALWAYS Handles:
- 🔧 ALL file operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- 💻 ALL code generation and programming tasks
- 🖥️ ALL bash commands and system operations
- 🏗️ ALL actual implementation work
- 🔍 ALL project navigation and code analysis
- 📝 ALL TodoWrite and task management
- 🔄 ALL git operations (commit, push, merge)
- 📦 ALL package management (npm, pip, etc.)
- 🧪 ALL testing and validation
- 🔧 ALL debugging and troubleshooting

🧠 MCP Tools ONLY Handle:
- 🎯 Coordination only - Planning Claude Code's actions
- 💾 Memory management - Storing decisions and context
- 🤖 Neural features - Learning from Claude Code's work
- 📊 Performance tracking - Monitoring Claude Code's efficiency
- 🐝 Swarm orchestration - Coordinating multiple Claude Code instances
- 🔗 GitHub integration - Advanced repository coordination

⚠️ Key Principle:
MCP tools coordinate, Claude Code executes!"""
    
    def validate_tool_usage(self, tool_name: str, operation: str) -> Dict[str, Any]:
        """Validate if a tool is being used correctly."""
        is_mcp = any(prefix in tool_name for prefix in self.mcp_coordination_tools)
        is_execution_op = operation in ['write', 'create', 'execute', 'run', 'build']
        
        if is_mcp and is_execution_op:
            return {
                "valid": False,
                "reason": f"MCP tool '{tool_name}' cannot perform execution operation '{operation}'",
                "suggestion": f"Use Claude Code tools for {operation} operations"
            }
        
        return {"valid": True}