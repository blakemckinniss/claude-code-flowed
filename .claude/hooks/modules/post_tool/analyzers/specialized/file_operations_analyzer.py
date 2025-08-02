"""File Operations Analyzer.

Specialized analyzer for file-related operations (Read, Write, Edit, MultiEdit)
with focus on code quality, path safety, and batch optimization opportunities.
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ...core.tool_analyzer_base import (
    BaseToolAnalyzer, ToolContext, FeedbackResult, FeedbackSeverity, ToolCategory
)


class FileOperationsAnalyzer(BaseToolAnalyzer):
    """Analyzer for file operations with code quality and safety checks."""
    
    def __init__(self, priority: int = 800, enable_ruff_integration: bool = True):
        """Initialize file operations analyzer.
        
        Args:
            priority: Analyzer priority
            enable_ruff_integration: Whether to integrate with Ruff for Python files
        """
        super().__init__(priority)
        self.enable_ruff_integration = enable_ruff_integration
        self.unsafe_path_patterns = [
            r'\.\./',           # Parent directory traversal
            r'/etc/',           # System configuration
            r'/root/',          # Root directory
            r'/proc/',          # Process filesystem
            r'/sys/',           # System filesystem
        ]
        self.batch_opportunity_threshold = 3
    
    def get_analyzer_name(self) -> str:
        return "file_operations_analyzer"
    
    def get_supported_tools(self) -> List[str]:
        return ["Read", "Write", "Edit", "MultiEdit", "mcp__filesystem__*"]
    
    def get_tool_categories(self) -> List[ToolCategory]:
        return [ToolCategory.FILE_OPERATIONS, ToolCategory.CODE_QUALITY]
    
    async def _analyze_tool_impl(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze file operations for safety, quality, and optimization."""
        tool_name = context.tool_name
        tool_input = context.tool_input
        
        # Path safety analysis
        path_safety_result = self._analyze_path_safety(tool_input)
        if path_safety_result:
            return path_safety_result
        
        # Code quality analysis for Python files
        if tool_name in ["Write", "Edit", "MultiEdit"]:
            code_quality_result = await self._analyze_code_quality(tool_input, context)
            if code_quality_result:
                return code_quality_result
        
        # Hook file violation check
        hook_violation_result = self._check_hook_file_violations(tool_input)
        if hook_violation_result:
            return hook_violation_result
        
        # Batch optimization opportunities
        batch_result = self._analyze_batch_opportunities(context)
        if batch_result:
            return batch_result
        
        # File system efficiency analysis
        efficiency_result = self._analyze_file_system_efficiency(tool_input, context)
        if efficiency_result:
            return efficiency_result
        
        return None
    
    def _analyze_path_safety(self, tool_input: Dict[str, Any]) -> Optional[FeedbackResult]:
        """Analyze file paths for security issues."""
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        if not file_path:
            return None
        
        # Check for unsafe path patterns
        for pattern in self.unsafe_path_patterns:
            if re.search(pattern, file_path):
                return FeedbackResult(
                    severity=FeedbackSeverity.ERROR,
                    message=f"Unsafe file path detected: {file_path}",
                    suggestions=[
                        "Use relative paths within the project directory",
                        "Avoid system directories and parent directory traversal",
                        "Consider using mcp__filesystem__* tools for safer operations"
                    ],
                    analyzer_name=self.get_analyzer_name()
                )
        
        # Check for non-existent parent directories
        if not os.path.isabs(file_path):
            full_path = os.path.join("/home/devcontainers/flowed", file_path)
        else:
            full_path = file_path
        
        parent_dir = os.path.dirname(full_path)
        if not os.path.exists(parent_dir):
            return FeedbackResult(
                severity=FeedbackSeverity.WARNING,
                message=f"Parent directory does not exist: {parent_dir}",
                suggestions=[
                    f"Create parent directory first: mkdir -p {parent_dir}",
                    "Use mcp__filesystem__create_directory for safer directory creation"
                ],
                analyzer_name=self.get_analyzer_name()
            )
        
        return None
    
    async def _analyze_code_quality(
        self, 
        tool_input: Dict[str, Any], 
        context: ToolContext
    ) -> Optional[FeedbackResult]:
        """Analyze code quality for Python files using Ruff."""
        if not self.enable_ruff_integration:
            return None
        
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        if not file_path or not file_path.endswith(".py"):
            return None
        
        # Convert to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.join("/home/devcontainers/flowed", file_path)
        
        # Check if file exists (for Write operations, it might not exist yet)
        if not os.path.exists(file_path):
            return None
        
        try:
            # Run Ruff check
            result = subprocess.run(
                ["ruff", "check", file_path, "--output-format=json"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd="/home/devcontainers/flowed"
            )
            
            if result.stdout:
                try:
                    issues = json.loads(result.stdout)
                    if issues:
                        return self._format_ruff_feedback(file_path, issues)
                except json.JSONDecodeError:
                    pass
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # Ruff not available or failed - don't block
            pass
        
        return None
    
    def _format_ruff_feedback(self, file_path: str, issues: List[Dict]) -> FeedbackResult:
        """Format Ruff issues into feedback result."""
        total_issues = len(issues)
        
        # Categorize issues
        errors = [i for i in issues if i.get("type") == "E" or i.get("code", "").startswith("E")]
        security_issues = [i for i in issues if i.get("code", "").startswith("S")]
        
        # Determine severity
        if security_issues:
            severity = FeedbackSeverity.ERROR
            primary_message = f"Security issues found in {os.path.basename(file_path)}"
        elif errors:
            severity = FeedbackSeverity.WARNING
            primary_message = f"Code errors found in {os.path.basename(file_path)}"
        else:
            severity = FeedbackSeverity.WARNING
            primary_message = f"Code quality issues found in {os.path.basename(file_path)}"
        
        # Create suggestions
        suggestions = [
            f"Run: ruff check {os.path.relpath(file_path, '/home/devcontainers/flowed')} --fix",
            f"Auto-format: ruff format {os.path.relpath(file_path, '/home/devcontainers/flowed')}",
        ]
        
        if security_issues:
            suggestions.insert(0, "⚠️  Security issues require immediate attention")
        
        return FeedbackResult(
            severity=severity,
            message=f"{primary_message} ({total_issues} issues total)",
            suggestions=suggestions,
            metadata={
                "total_issues": total_issues,
                "security_issues": len(security_issues),
                "errors": len(errors),
                "file_path": file_path
            },
            analyzer_name=self.get_analyzer_name()
        )
    
    def _check_hook_file_violations(self, tool_input: Dict[str, Any]) -> Optional[FeedbackResult]:
        """Check for violations in hook files (sys.path manipulations)."""
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        if not file_path:
            return None
        
        # Convert to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.join("/home/devcontainers/flowed", file_path)
        
        # Check if it's a hook file
        hooks_dir = "/home/devcontainers/flowed/.claude/hooks"
        if not file_path.startswith(hooks_dir) or not file_path.endswith(".py"):
            return None
        
        # Skip path_resolver.py - it's allowed to use sys.path
        if file_path.endswith("path_resolver.py"):
            return None
        
        # Get content being written/edited
        content = self._extract_file_content(tool_input)
        if not content:
            return None
        
        # Check for sys.path manipulations
        sys_path_pattern = r'sys\.path\.(insert|append|extend)\s*\('
        if re.search(sys_path_pattern, content):
            return FeedbackResult(
                severity=FeedbackSeverity.ERROR,
                message="sys.path manipulations not allowed in hook files",
                suggestions=[
                    "Use centralized path management:",
                    "from modules.utils.path_resolver import setup_hook_paths",
                    "setup_hook_paths()",
                    "See: .claude/hooks/doc/PATH_MANAGEMENT.md for details"
                ],
                metadata={"file_path": file_path},
                analyzer_name=self.get_analyzer_name()
            )
        
        return None
    
    def _extract_file_content(self, tool_input: Dict[str, Any]) -> str:
        """Extract content being written or edited."""
        # For Write operations
        if "content" in tool_input:
            return tool_input["content"]
        
        # For Edit operations
        if "new_string" in tool_input:
            return tool_input["new_string"]
        
        # For MultiEdit operations
        if "edits" in tool_input:
            edits = tool_input["edits"]
            return "\n".join(edit.get("new_string", "") for edit in edits)
        
        return ""
    
    def _analyze_batch_opportunities(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze opportunities for batching file operations."""
        recent_tools = context.workflow_history[-5:] if context.workflow_history else []
        
        # Count recent file operations
        file_ops = [tool for tool in recent_tools if tool in ["Read", "Write", "Edit"]]
        
        if len(file_ops) >= self.batch_opportunity_threshold:
            return FeedbackResult(
                severity=FeedbackSeverity.WARNING,
                message=f"Multiple file operations detected ({len(file_ops)} in recent workflow)",
                suggestions=[
                    "Consider using mcp__filesystem__read_multiple_files for batch reading",
                    "Use MultiEdit for multiple file modifications",
                    "Group related file operations in single messages",
                    "Consider using mcp__zen__planner for coordinated file operations"
                ],
                metadata={"file_operations_count": len(file_ops)},
                analyzer_name=self.get_analyzer_name()
            )
        
        return None
    
    def _analyze_file_system_efficiency(
        self, 
        tool_input: Dict[str, Any], 
        context: ToolContext
    ) -> Optional[FeedbackResult]:
        """Analyze file system efficiency patterns."""
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        if not file_path:
            return None
        
        # Check for repeated operations on same file
        recent_tools = context.workflow_history[-3:] if context.workflow_history else []
        if len(recent_tools) >= 2 and all(tool in ["Read", "Write", "Edit"] for tool in recent_tools):
            return FeedbackResult(
                severity=FeedbackSeverity.WARNING,
                message="Repeated file operations on same file detected",
                suggestions=[
                    "Consider reading file once and making all changes together",
                    "Use MultiEdit for multiple changes to same file",
                    "Plan file operations to minimize I/O overhead"
                ],
                analyzer_name=self.get_analyzer_name()
            )
        
        # Check for large file operations
        if context.tool_name == "Write":
            content = tool_input.get("content", "")
            if len(content) > 100000:  # >100KB
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message=f"Large file write operation ({len(content):,} characters)",
                    suggestions=[
                        "Consider breaking large files into smaller modules",
                        "Use streaming operations for very large files",
                        "Monitor memory usage for large file operations"
                    ],
                    metadata={"file_size": len(content)},
                    analyzer_name=self.get_analyzer_name()
                )
        
        return None