#!/usr/bin/env python3
"""Analyzer dispatch system for post-tool processing.

Provides a flexible framework for registering and executing tool-specific analyzers
with async support and intelligent pattern matching.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Pattern, Union, Callable
from enum import Enum
import logging
import time

# Import ZEN consultant patterns for intelligence
try:
    from ..core.zen_consultant import ZenConsultant, ComplexityLevel, CoordinationType
    ZEN_AVAILABLE = True
except ImportError:
    ZEN_AVAILABLE = False
    ZenConsultant = None


class AnalyzerResult(Enum):
    """Result types from analyzer execution."""
    # Status codes for analyzer results (not sensitive data)
    PASS = "pass"           # No action needed  # noqa: S105
    GUIDANCE = "guidance"   # Provide guidance but don't block
    BLOCK = "block"         # Block the operation
    WARN = "warn"          # Show warning but continue


@dataclass
class AnalysisResult:
    """Result from tool analysis."""
    result_type: AnalyzerResult
    message: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hook processing."""
        needs_action = self.result_type != AnalyzerResult.PASS
        action_type = self.result_type.value if needs_action else None
        
        return {
            "needs_action": needs_action,
            "action_type": action_type,
            "message": self.message,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class ToolMatcher:
    """Flexible tool matching system for analyzer registration."""
    
    def __init__(self, pattern: Union[str, Pattern, Callable[[str], bool]]):
        """Initialize matcher with pattern, regex, or function."""
        if isinstance(pattern, str):
            if pattern == "*":
                # Match all tools
                self._matcher = lambda tool: True
            elif "*" in pattern:
                # Glob-style pattern
                regex_pattern = pattern.replace("*", ".*")
                self._regex = re.compile(f"^{regex_pattern}$")
                self._matcher = lambda tool: bool(self._regex.match(tool))
            else:
                # Exact match
                self._matcher = lambda tool: tool == pattern
        elif isinstance(pattern, Pattern):
            # Compiled regex
            self._matcher = lambda tool: bool(pattern.match(tool))
        elif callable(pattern):
            # Custom function
            self._matcher = pattern
        else:
            raise TypeError(f"Invalid pattern type: {type(pattern)}")
    
    def matches(self, tool_name: str) -> bool:
        """Check if tool matches this pattern."""
        try:
            return self._matcher(tool_name)
        except Exception:
            return False


class ToolAnalyzer(ABC):
    """Base class for tool-specific analyzers."""
    
    def __init__(self, name: str, priority: int = 100):
        """Initialize analyzer with name and priority."""
        self.name = name
        self.priority = priority
        self.logger = logging.getLogger(f"analyzer.{name}")
        self._execution_count = 0
        self._total_time = 0.0
    
    @abstractmethod
    def analyze(self, tool_name: str, tool_input: Dict[str, Any], 
                tool_response: Dict[str, Any]) -> AnalysisResult:
        """Analyze tool usage and return result."""
        pass
    
    async def analyze_async(self, tool_name: str, tool_input: Dict[str, Any], 
                           tool_response: Dict[str, Any]) -> AnalysisResult:
        """Async version of analyze. Override for true async analyzers."""
        return self.analyze(tool_name, tool_input, tool_response)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer performance statistics."""
        avg_time = self._total_time / self._execution_count if self._execution_count > 0 else 0
        return {
            "name": self.name,
            "priority": self.priority,
            "execution_count": self._execution_count,
            "total_time": self._total_time,
            "average_time": avg_time
        }
    
    def _track_execution(self, execution_time: float):
        """Track execution metrics."""
        self._execution_count += 1
        self._total_time += execution_time


@dataclass
class AnalyzerRegistration:
    """Registration info for an analyzer."""
    analyzer: ToolAnalyzer
    matcher: ToolMatcher
    async_enabled: bool = False


class AnalyzerRegistry:
    """Registry for tool analyzers with pattern matching."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._registrations: List[AnalyzerRegistration] = []
        self._logger = logging.getLogger("analyzer.registry")
    
    def register(self, analyzer: ToolAnalyzer, 
                 pattern: Union[str, Pattern, Callable[[str], bool]],
                 async_enabled: bool = False) -> None:
        """Register an analyzer for matching tools."""
        matcher = ToolMatcher(pattern)
        registration = AnalyzerRegistration(analyzer, matcher, async_enabled)
        
        # Insert sorted by priority (highest first)
        inserted = False
        for i, reg in enumerate(self._registrations):
            if analyzer.priority > reg.analyzer.priority:
                self._registrations.insert(i, registration)
                inserted = True
                break
        
        if not inserted:
            self._registrations.append(registration)
        
        self._logger.debug(f"Registered analyzer '{analyzer.name}' with priority {analyzer.priority}")
    
    def get_analyzers(self, tool_name: str) -> List[AnalyzerRegistration]:
        """Get all analyzers that match the tool name."""
        matches = []
        for registration in self._registrations:
            if registration.matcher.matches(tool_name):
                matches.append(registration)
        return matches
    
    def get_all_analyzers(self) -> List[AnalyzerRegistration]:
        """Get all registered analyzers."""
        return self._registrations.copy()


class AnalyzerDispatcher:
    """Main dispatcher for executing tool analyzers."""
    
    def __init__(self, max_execution_time: float = 5.0):
        """Initialize dispatcher with registry."""
        self.registry = AnalyzerRegistry()
        self.max_execution_time = max_execution_time
        self._logger = logging.getLogger("analyzer.dispatcher")
        self._zen_consultant = ZenConsultant() if ZEN_AVAILABLE else None
        
        # Register built-in analyzers
        self._register_builtin_analyzers()
    
    def _register_builtin_analyzers(self):
        """Register built-in core analyzers."""
        # TodoWrite analyzer for batch validation
        self.registry.register(
            TodoWriteAnalyzer(), 
            "TodoWrite",
            async_enabled=False
        )
        
        # Bash analyzer for command validation
        self.registry.register(
            BashAnalyzer(),
            "Bash", 
            async_enabled=False
        )
        
        # General MCP analyzer for all MCP tools
        self.registry.register(
            MCPAnalyzer(),
            "mcp__*",
            async_enabled=True
        )
        
        # File operation analyzer
        self.registry.register(
            FileOperationAnalyzer(),
            lambda tool: tool in ["Write", "Edit", "MultiEdit", "Read"],
            async_enabled=False
        )
    
    def analyze_tool(self, tool_name: str, tool_input: Dict[str, Any], 
                    tool_response: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tool usage and return action result."""
        analyzers = self.registry.get_analyzers(tool_name)
        
        if not analyzers:
            return {"needs_action": False}
        
        # Execute analyzers in priority order
        for registration in analyzers:
            try:
                start_time = time.time()
                
                if registration.async_enabled:
                    # Run async analyzer
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(
                        asyncio.wait_for(
                            registration.analyzer.analyze_async(tool_name, tool_input, tool_response),
                            timeout=self.max_execution_time
                        )
                    )
                else:
                    # Run sync analyzer
                    result = registration.analyzer.analyze(tool_name, tool_input, tool_response)
                
                execution_time = time.time() - start_time
                registration.analyzer._track_execution(execution_time)
                
                # Return first non-pass result
                if result.result_type != AnalyzerResult.PASS:
                    return result.to_dict()
                    
            except asyncio.TimeoutError:
                self._logger.warning(f"Analyzer '{registration.analyzer.name}' timed out")
            except Exception as e:
                self._logger.exception(f"Analyzer '{registration.analyzer.name}' failed: {e}")
        
        return {"needs_action": False}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher and analyzer statistics."""
        analyzer_stats = []
        for registration in self.registry.get_all_analyzers():
            stats = registration.analyzer.get_stats()
            stats["async_enabled"] = registration.async_enabled
            analyzer_stats.append(stats)
        
        return {
            "total_analyzers": len(analyzer_stats),
            "max_execution_time": self.max_execution_time,
            "analyzers": analyzer_stats
        }


# Built-in analyzers

class TodoWriteAnalyzer(ToolAnalyzer):
    """Analyzer for TodoWrite operations to ensure proper batching."""
    
    def __init__(self):
        super().__init__("TodoWrite", priority=800)
        self._last_todo_count = 0
    
    def analyze(self, tool_name: str, tool_input: Dict[str, Any], 
                tool_response: Dict[str, Any]) -> AnalysisResult:
        """Analyze TodoWrite for batching best practices."""
        todos = tool_input.get("todos", [])
        todo_count = len(todos)
        self._last_todo_count = todo_count
        
        # Guidance for small todo lists
        if todo_count < 5:
            return AnalysisResult(
                result_type=AnalyzerResult.GUIDANCE,
                message=f"""
🎯 TODO BATCHING GUIDANCE

Current todos: {todo_count} (Recommended: 5-10+)

💡 OPTIMIZATION SUGGESTIONS:
  • Batch related tasks into single TodoWrite call
  • Include setup, implementation, and validation steps
  • Add error handling and cleanup tasks
  • Consider cross-cutting concerns (testing, docs)

🚀 GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"
""",
                confidence=0.8,
                metadata={"todo_count": todo_count, "recommendation": "batch_more"}
            )
        
        # Perfect batching
        if 5 <= todo_count <= 12:
            return AnalysisResult(
                result_type=AnalyzerResult.PASS,
                message="Excellent todo batching!",
                confidence=1.0,
                metadata={"todo_count": todo_count, "status": "optimal"}
            )
        
        # Too many todos - might be overwhelming
        if todo_count > 12:
            return AnalysisResult(
                result_type=AnalyzerResult.WARN,
                message=f"""
⚠️ TODO LIST SIZE WARNING

Current todos: {todo_count} (Recommended: 5-12)

Consider breaking into phases or sub-projects for better focus.
""",
                confidence=0.7,
                metadata={"todo_count": todo_count, "recommendation": "split_phases"}
            )
        
        return AnalysisResult(AnalyzerResult.PASS, "Todo analysis complete")


class BashAnalyzer(ToolAnalyzer):
    """Analyzer for Bash commands to detect patterns and suggest optimizations."""
    
    def __init__(self):
        super().__init__("Bash", priority=600)
        self._command_history: List[str] = []
    
    def analyze(self, tool_name: str, tool_input: Dict[str, Any], 
                tool_response: Dict[str, Any]) -> AnalysisResult:
        """Analyze Bash commands for optimization opportunities."""
        command = tool_input.get("command", "")
        self._command_history.append(command)
        
        # Keep only recent commands
        if len(self._command_history) > 10:
            self._command_history = self._command_history[-10:]
        
        # Check for repeated similar commands
        similar_commands = [cmd for cmd in self._command_history[-5:] 
                          if self._commands_similar(command, cmd)]
        
        if len(similar_commands) >= 3:
            return AnalysisResult(
                result_type=AnalyzerResult.GUIDANCE,
                message=f"""
🔄 REPEATED COMMAND PATTERN DETECTED

Command pattern: {self._extract_command_pattern(command)}
Occurrences: {len(similar_commands)}

💡 OPTIMIZATION SUGGESTIONS:
  • Consider creating a shell script for repeated operations
  • Use shell loops or parameter expansion for batch operations
  • Combine related commands with && or ; operators

🚀 BATCHING OPPORTUNITY: Group similar operations in one message
""",
                confidence=0.8,
                metadata={
                    "pattern": self._extract_command_pattern(command),
                    "occurrences": len(similar_commands)
                }
            )
        
        # Check for dangerous commands
        dangerous_patterns = ["rm -rf", "sudo rm", "chmod 777", "> /dev/"]
        if any(pattern in command for pattern in dangerous_patterns):
            return AnalysisResult(
                result_type=AnalyzerResult.WARN,
                message=f"""
⚠️ POTENTIALLY DANGEROUS COMMAND

Command: {command}

Please verify this operation is intended and safe.
""",
                confidence=0.9,
                metadata={"command": command, "risk_level": "high"}
            )
        
        return AnalysisResult(AnalyzerResult.PASS, "Command analysis complete")
    
    def _commands_similar(self, cmd1: str, cmd2: str) -> bool:
        """Check if two commands are similar (same base command)."""
        base1 = cmd1.split()[0] if cmd1.split() else ""
        base2 = cmd2.split()[0] if cmd2.split() else ""
        return base1 == base2 and base1 != ""
    
    def _extract_command_pattern(self, command: str) -> str:
        """Extract the base pattern of a command."""
        parts = command.split()
        if not parts:
            return command
        return f"{parts[0]} ..."


class MCPAnalyzer(ToolAnalyzer):
    """Analyzer for MCP tools to provide coordination insights."""
    
    def __init__(self):
        super().__init__("MCP", priority=900)
        self._mcp_sequence: List[str] = []
    
    def analyze(self, tool_name: str, tool_input: Dict[str, Any], 
                tool_response: Dict[str, Any]) -> AnalysisResult:
        """Synchronous analysis of MCP tool usage patterns."""
        self._mcp_sequence.append(tool_name)
        
        # Keep recent sequence
        if len(self._mcp_sequence) > 20:
            self._mcp_sequence = self._mcp_sequence[-20:]
        
        # Check for proper ZEN → Flow coordination
        if tool_name.startswith("mcp__claude-flow__") and not self._has_recent_zen():
            return AnalysisResult(
                result_type=AnalyzerResult.GUIDANCE,
                message=f"""
👑 ZEN COORDINATION OPPORTUNITY

Flow tool used: {tool_name}
Missing: Recent Queen ZEN consultation

💡 RECOMMENDED FLOW:
  1. mcp__zen__chat - Get intelligent guidance
  2. mcp__zen__planner - Structure the approach  
  3. {tool_name} - Execute with coordination

🎯 Queen ZEN's wisdom enhances Flow Worker effectiveness!  
""",
                confidence=0.8,
                metadata={
                    "tool": tool_name,
                    "missing": "zen_consultation",
                    "sequence": self._mcp_sequence[-5:]
                }
            )
        
        return AnalysisResult(AnalyzerResult.PASS, "MCP coordination analysis complete")
    
    async def analyze_async(self, tool_name: str, tool_input: Dict[str, Any], 
                           tool_response: Dict[str, Any]) -> AnalysisResult:
        """Async analysis of MCP tool usage patterns."""
        # Delegate to synchronous version for now
        return self.analyze(tool_name, tool_input, tool_response)
    
    def _has_recent_zen(self, lookback: int = 10) -> bool:
        """Check if recent MCP sequence included ZEN tools."""
        recent_tools = self._mcp_sequence[-lookback:]
        return any("mcp__zen__" in tool for tool in recent_tools)


class FileOperationAnalyzer(ToolAnalyzer):
    """Analyzer for file operations to detect patterns."""
    
    def __init__(self):
        super().__init__("FileOps", priority=700)
        self._file_operations: List[Dict[str, Any]] = []
    
    def analyze(self, tool_name: str, tool_input: Dict[str, Any], 
                tool_response: Dict[str, Any]) -> AnalysisResult:
        """Analyze file operations for batching opportunities."""
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        
        self._file_operations.append({
            "tool": tool_name,
            "path": file_path,
            "timestamp": time.time()
        })
        
        # Keep recent operations
        if len(self._file_operations) > 20:
            self._file_operations = self._file_operations[-20:]
        
        # Check for rapid file operations on same file
        recent_ops = [op for op in self._file_operations[-5:] 
                     if op["path"] == file_path and op["path"]]
        
        if len(recent_ops) >= 3:
            return AnalysisResult(
                result_type=AnalyzerResult.GUIDANCE,
                message=f"""
📁 REPEATED FILE OPERATIONS DETECTED

File: {file_path}
Operations: {len(recent_ops)} in recent sequence

💡 OPTIMIZATION OPPORTUNITY:
  • Consider using MultiEdit for multiple changes
  • Plan file modifications before executing
  • Use mcp__zen__planner for complex file workflows

🚀 Reduce file I/O with batched operations
""",
                confidence=0.7,
                metadata={
                    "file_path": file_path,
                    "operation_count": len(recent_ops),
                    "operations": [op["tool"] for op in recent_ops]
                }
            )
        
        return AnalysisResult(AnalyzerResult.PASS, "File operation analysis complete")