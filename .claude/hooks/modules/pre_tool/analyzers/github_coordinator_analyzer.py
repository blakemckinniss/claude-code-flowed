"""GitHub Coordinator Analyzer - claude-flow integration for repository intelligence.

Integrates claude-flow's GitHub workflow coordination capabilities into the existing
Queen ZEN hierarchy system, providing intelligent guidance for GitHub operations.

Priority: 825 (High priority GitHub workflow coordination)
"""

import re
from typing import Dict, Any, List, Optional, Set
from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class GitHubWorkflowPattern:
    """Represents GitHub workflow patterns for intelligent coordination."""
    
    # GitHub tool patterns
    GITHUB_TOOLS = {
        "mcp__github__create_pull_request",
        "mcp__github__merge_pull_request", 
        "mcp__github__create_issue",
        "mcp__github__update_issue",
        "mcp__github__get_pull_request",
        "mcp__github__list_pull_requests",
        "mcp__github__search_code",
        "mcp__github__push_files",
        "mcp__github__create_or_update_file"
    }
    
    # Repository management patterns
    REPO_MANAGEMENT_TOOLS = {
        "mcp__github__create_repository",
        "mcp__github__fork_repository",
        "mcp__github__create_branch",
        "mcp__github__list_branches"
    }
    
    # Code review patterns
    CODE_REVIEW_TOOLS = {
        "mcp__github__create_and_submit_pull_request_review",
        "mcp__github__add_pull_request_review_comment_to_pending_review",
        "mcp__github__submit_pending_pull_request_review",
        "mcp__github__request_copilot_review"
    }
    
    # Issue management patterns
    ISSUE_MANAGEMENT_TOOLS = {
        "mcp__github__create_issue",
        "mcp__github__update_issue", 
        "mcp__github__get_issue",
        "mcp__github__list_issues",
        "mcp__github__search_issues"
    }


class GitHubCoordinatorAnalyzer(HiveWorkflowValidator):
    """GitHub Coordinator Analyzer implementing claude-flow repository intelligence.
    
    Priority: 825 (High priority for GitHub workflow coordination)
    Provides intelligent guidance for GitHub operations and repository management.
    """
    
    def __init__(self, priority: int = 825):
        super().__init__(priority)
        self.github_operations_count = 0
        self.coordination_suggestions = 0
        self.workflow_optimizations = 0
    
    def get_validator_name(self) -> str:
        return "github_coordinator_analyzer"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate GitHub operations and provide repository intelligence."""
        
        # Only analyze GitHub-related tools
        if not self._is_github_operation(tool_name):
            return None
        
        self.github_operations_count += 1
        
        # Analyze GitHub workflow patterns
        workflow_analysis = self._analyze_github_workflow(tool_name, tool_input, context)
        
        if workflow_analysis:
            return workflow_analysis
        
        # Check for GitHub coordination opportunities
        coordination_suggestion = self._suggest_github_coordination(tool_name, tool_input, context)
        
        if coordination_suggestion:
            self.coordination_suggestions += 1
            return coordination_suggestion
        
        # Check for workflow optimization opportunities
        optimization_suggestion = self._suggest_workflow_optimization(tool_name, tool_input, context)
        
        if optimization_suggestion:
            self.workflow_optimizations += 1
            return optimization_suggestion
        
        return None
    
    def _is_github_operation(self, tool_name: str) -> bool:
        """Check if this is a GitHub-related operation."""
        return (tool_name.startswith("mcp__github__") or
                self._is_git_bash_command(tool_name))
    
    def _is_git_bash_command(self, tool_name: str) -> bool:
        """Check if this is a git-related bash command."""
        return tool_name == "Bash"  # Will be validated in tool_input analysis
    
    def _analyze_github_workflow(self, tool_name: str, tool_input: Dict[str, Any],
                                context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Analyze GitHub workflow patterns for optimization opportunities."""
        
        # Complex PR operations without ZEN planning
        if self._is_complex_pr_operation(tool_name, tool_input):
            if not context.has_zen_coordination(3):
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.BYPASSING_ZEN,
                    message="🐙 Complex GitHub PR operation should have Queen ZEN's strategic planning",
                    suggested_alternative="mcp__zen__thinkdeep { step: \"Plan GitHub PR workflow strategy\" }",
                    hive_guidance="Queen ZEN's GitHub intelligence prevents PR conflicts and optimizes review workflows",
                    priority_score=75
                )
        
        # Repository management without coordination
        if tool_name in GitHubWorkflowPattern.REPO_MANAGEMENT_TOOLS:
            if not context.has_zen_coordination(2) and not context.has_flow_coordination(2):
                return ValidationResult(
                    severity=ValidationSeverity.SUGGEST,
                    violation_type=WorkflowViolationType.MISSING_COORDINATION,
                    message="🏗️ Repository management benefits from hive coordination",
                    suggested_alternative="mcp__claude-flow__github_swarm for repository orchestration",
                    hive_guidance="GitHub Flow Workers can coordinate repository structure and workflow optimization",
                    priority_score=60
                )
        
        # Code review workflow optimization
        if tool_name in GitHubWorkflowPattern.CODE_REVIEW_TOOLS:
            return self._analyze_code_review_workflow(tool_name, tool_input, context)
        
        # Issue management workflow
        if tool_name in GitHubWorkflowPattern.ISSUE_MANAGEMENT_TOOLS:
            return self._analyze_issue_management_workflow(tool_name, tool_input, context)
        
        return None
    
    def _is_complex_pr_operation(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Determine if this is a complex PR operation requiring coordination."""
        
        if tool_name == "mcp__github__create_pull_request":
            # Large PR descriptions or complex titles
            body = tool_input.get("body", "")
            title = tool_input.get("title", "")
            return (len(body) > 500 or 
                   len(title) > 100 or
                   any(keyword in (body + title).lower() 
                       for keyword in ["breaking change", "major refactor", "architecture", "migration"]))
        
        elif tool_name == "mcp__github__push_files":
            # Multiple files or large changes
            files = tool_input.get("files", [])
            return len(files) > 5 or any(len(f.get("content", "")) > 2000 for f in files)
        
        elif tool_name == "mcp__github__merge_pull_request":
            # Merge operations are always complex
            return True
        
        return False
    
    def _analyze_code_review_workflow(self, tool_name: str, tool_input: Dict[str, Any],
                                    context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Analyze code review workflow patterns."""
        
        if tool_name == "mcp__github__create_and_submit_pull_request_review":
            # Suggest using pending review workflow for complex reviews
            body = tool_input.get("body", "")
            if len(body) > 200:
                return ValidationResult(
                    severity=ValidationSeverity.SUGGEST,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message="📝 Complex code review detected - consider pending review workflow",
                    suggested_alternative="mcp__github__create_pending_pull_request_review followed by detailed comments",
                    hive_guidance="Pending review workflow allows comprehensive feedback before submission",
                    priority_score=50
                )
        
        elif tool_name == "mcp__github__request_copilot_review":
            # Suggest combining with human review for important changes
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.MISSING_COORDINATION,
                message="🤖 Copilot review works better with human oversight",
                suggested_alternative="Combine Copilot review with human reviewer assignment",
                hive_guidance="Hybrid AI + human review provides optimal code quality assurance",
                priority_score=40
            )
        
        return None
    
    def _analyze_issue_management_workflow(self, tool_name: str, tool_input: Dict[str, Any],
                                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Analyze issue management workflow patterns."""
        
        if tool_name == "mcp__github__create_issue":
            # Complex issues should have proper labeling and assignment
            body = tool_input.get("body", "")
            labels = tool_input.get("labels", [])
            assignees = tool_input.get("assignees", [])
            
            if len(body) > 500 and (not labels or not assignees):
                return ValidationResult(
                    severity=ValidationSeverity.SUGGEST,
                    violation_type=WorkflowViolationType.MISSING_COORDINATION,
                    message="🎯 Complex issue creation should include labels and assignees",
                    suggested_alternative="Add relevant labels and assign team members for better triage",
                    hive_guidance="Proper issue classification accelerates resolution and improves workflow",
                    priority_score=45
                )
        
        return None
    
    def _suggest_github_coordination(self, tool_name: str, tool_input: Dict[str, Any],
                                   context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Suggest GitHub coordination improvements."""
        
        # Git bash commands should use GitHub MCP tools when possible
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            
            # Git operations that have MCP equivalents
            git_mcp_suggestions = {
                r"git\s+push": "mcp__github__push_files for better error handling",
                r"git\s+clone": "Consider using mcp__github__fork_repository for collaboration",
                r"gh\s+pr\s+create": "mcp__github__create_pull_request for structured workflow",
                r"gh\s+issue\s+create": "mcp__github__create_issue for better integration"
            }
            
            for pattern, suggestion in git_mcp_suggestions.items():
                if re.search(pattern, command):
                    return ValidationResult(
                        severity=ValidationSeverity.SUGGEST,
                        violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                        message=f"🔧 Consider MCP GitHub tool for better coordination: {suggestion}",
                        suggested_alternative=suggestion,
                        hive_guidance="MCP GitHub tools provide better error handling and hive integration",
                        priority_score=35
                    )
        
        return None
    
    def _suggest_workflow_optimization(self, tool_name: str, tool_input: Dict[str, Any],
                                     context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Suggest GitHub workflow optimizations."""
        
        # Multiple GitHub operations in sequence - suggest batching
        recent_tools = context._recent_tools[-5:]
        github_tool_count = sum(1 for tool in recent_tools if tool.startswith("mcp__github__"))
        
        if github_tool_count >= 3 and tool_name.startswith("mcp__github__"):
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.FRAGMENTED_WORKFLOW,
                message="🔄 Multiple GitHub operations detected - consider workflow coordination",
                suggested_alternative="mcp__claude-flow__github_swarm for coordinated GitHub operations",
                hive_guidance="GitHub Flow Workers can batch and optimize multiple repository operations",
                priority_score=55
            )
        
        # File operations before GitHub operations - suggest integration
        if (tool_name.startswith("mcp__github__") and 
            any(tool in recent_tools for tool in ["Write", "Edit", "MultiEdit"])):
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="📁 File operations before GitHub - consider integrated workflow",
                suggested_alternative="mcp__github__push_files for atomic file + repository operations",
                hive_guidance="Integrated file and GitHub operations prevent inconsistencies",
                priority_score=40
            )
        
        return None
    
    def get_github_metrics(self) -> Dict[str, Any]:
        """Get GitHub coordination metrics for monitoring."""
        return {
            "github_operations_analyzed": self.github_operations_count,
            "coordination_suggestions_provided": self.coordination_suggestions,
            "workflow_optimizations_suggested": self.workflow_optimizations,
            "github_intelligence_effectiveness": (
                (self.coordination_suggestions + self.workflow_optimizations) / 
                max(self.github_operations_count, 1) * 100
            )
        }