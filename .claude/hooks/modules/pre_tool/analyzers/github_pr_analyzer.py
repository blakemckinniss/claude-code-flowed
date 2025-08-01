"""GitHub Pull Request Analyzer - Claude Flow Integration Phase 2

Advanced PR workflow intelligence with merge strategy optimization,
review coordination, and conflict resolution suggestions.
"""

from typing import Dict, Any, List, Optional
from ..core.workflow_validator import HiveWorkflowValidator, ValidationResult, ValidationSeverity


class GitHubPRAnalyzer(HiveWorkflowValidator):
    """Analyzes GitHub pull request operations for workflow optimization."""
    
    def __init__(self, priority: int = 820):
        super().__init__(priority)
        self.name = "GitHubPRAnalyzer"
        self.pr_patterns = [
            "pull_request", "pr", "merge", "review", "draft",
            "conflict", "rebase", "squash", "branch"
        ]
    
    def get_validator_name(self) -> str:
        """Get name of this validator."""
        return "github_pr_analyzer"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context) -> Optional[ValidationResult]:
        """Validate PR workflow and provide guidance."""
        
        analysis = self.analyze(tool_name, tool_input, {"recent_tools": []})
        
        if analysis["priority"] == 0:
            return None
        
        suggestions = analysis.get("suggestions", [])
        if not suggestions:
            return None
        
        return ValidationResult(
            severity=ValidationSeverity.SUGGEST,
            violation_type=None,
            message=f"PR workflow optimization available: {suggestions[0]}",
            suggested_alternative=" | ".join(suggestions[:3]),
            hive_guidance=f"🐝 GitHub PR intelligence from {self.name}",
            priority_score=self.priority
        )
    
    def analyze(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PR-related operations."""
        if not self._is_pr_operation(tool_name, params):
            return {"priority": 0, "suggestions": []}
        
        analysis = {
            "priority": self.priority,
            "analyzer": self.name,
            "suggestions": [],
            "coordination_needed": False,
            "github_context": True
        }
        
        # Analyze PR workflow patterns
        pr_insights = self._analyze_pr_workflow(tool_name, params, context)
        analysis["suggestions"].extend(pr_insights)
        
        # Check for merge strategy optimization
        merge_suggestions = self._analyze_merge_strategy(tool_name, params)
        analysis["suggestions"].extend(merge_suggestions)
        
        # Review coordination analysis
        review_suggestions = self._analyze_review_coordination(tool_name, params)
        analysis["suggestions"].extend(review_suggestions)
        
        # Conflict resolution guidance
        conflict_suggestions = self._analyze_conflict_resolution(params)
        analysis["suggestions"].extend(conflict_suggestions)
        
        # Set coordination flag if complex operations detected
        analysis["coordination_needed"] = len(analysis["suggestions"]) > 2
        
        return analysis
    
    def _is_pr_operation(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Check if operation is PR-related."""
        # Check tool name
        if any(pattern in tool_name.lower() for pattern in self.pr_patterns):
            return True
        
        # Check parameters for PR-related content
        param_text = str(params).lower()
        if any(pattern in param_text for pattern in self.pr_patterns):
            return True
        
        # Check for GitHub PR MCP tools
        github_pr_tools = [
            "mcp__github__create_pull_request",
            "mcp__github__get_pull_request",
            "mcp__github__update_pull_request",
            "mcp__github__merge_pull_request",
            "mcp__github__get_pull_request_reviews",
            "mcp__github__create_and_submit_pull_request_review"
        ]
        
        return tool_name in github_pr_tools
    
    def _analyze_pr_workflow(self, tool_name: str, params: Dict[str, Any], 
                           context: Dict[str, Any]) -> List[str]:
        """Analyze PR workflow patterns and suggest optimizations."""
        suggestions = []
        
        # PR creation analysis
        if "create_pull_request" in tool_name:
            suggestions.append("📋 Consider using PR templates for consistent descriptions")
            suggestions.append("🔄 Add 'Draft' status for work-in-progress PRs")
            
            # Check for description quality
            description = params.get("body", "")
            if len(description) < 50:
                suggestions.append("📝 Add detailed PR description for better reviews")
        
        # PR review analysis
        if "review" in tool_name.lower():
            suggestions.append("👥 Use mcp__github__request_copilot_review for AI insights")
            suggestions.append("🎯 Consider creating pending review for batch comments")
        
        # Merge analysis
        if "merge" in tool_name.lower():
            suggestions.append("✅ Verify all checks pass before merging")
            suggestions.append("🔍 Use squash merge for feature branches")
        
        # Multi-PR operations
        if self._detect_multi_pr_operation(context):
            suggestions.append("🐝 Consider using GitHub swarm coordination for multiple PRs")
            suggestions.append("⚡ Batch PR operations for better performance")
        
        return suggestions
    
    def _analyze_merge_strategy(self, tool_name: str, params: Dict[str, Any]) -> List[str]:
        """Analyze and suggest optimal merge strategies."""
        suggestions = []
        
        if "merge" not in tool_name.lower():
            return suggestions
        
        merge_method = params.get("merge_method", "")
        
        # Merge strategy recommendations
        if not merge_method:
            suggestions.append("🔀 Specify merge strategy: 'squash' for features, 'merge' for releases")
        
        if merge_method == "rebase":
            suggestions.append("⚠️ Rebase merge removes commit history - use carefully")
        
        if merge_method == "squash":
            suggestions.append("✨ Squash merge - excellent for clean feature integration")
        
        # Branch analysis
        base_branch = params.get("base", "")
        head_branch = params.get("head", "")
        
        if base_branch == "main" and "feature" in head_branch:
            suggestions.append("🎯 Feature→main merge: recommend squash strategy")
        
        if base_branch == "main" and "hotfix" in head_branch:
            suggestions.append("🚨 Hotfix→main merge: consider merge commit for traceability")
        
        return suggestions
    
    def _analyze_review_coordination(self, tool_name: str, params: Dict[str, Any]) -> List[str]:
        """Analyze review coordination and suggest improvements."""
        suggestions = []
        
        # Review creation
        if "create" in tool_name and "review" in tool_name:
            suggestions.append("📝 Use pending reviews to batch comments effectively")
            suggestions.append("🎯 Include both praise and improvement suggestions")
        
        # Review submission
        if "submit" in tool_name and "review" in tool_name:
            event = params.get("event", "")
            
            if event == "REQUEST_CHANGES":
                suggestions.append("🔧 Provide specific, actionable feedback for changes")
                suggestions.append("💡 Include code examples when possible")
            
            if event == "APPROVE":
                suggestions.append("✅ Consider adding positive feedback with approval")
            
            if event == "COMMENT":
                suggestions.append("💬 Use comments for questions and discussion")
        
        # Multiple reviewers
        reviewers = params.get("reviewers", [])
        if isinstance(reviewers, list) and len(reviewers) > 3:
            suggestions.append("👥 Large review team - consider review delegation")
        
        return suggestions
    
    def _analyze_conflict_resolution(self, params: Dict[str, Any]) -> List[str]:
        """Analyze potential conflicts and suggest resolution strategies."""
        suggestions = []
        
        # Check for conflict indicators
        param_text = str(params).lower()
        conflict_indicators = ["conflict", "rebase", "merge conflict", "diverged"]
        
        if any(indicator in param_text for indicator in conflict_indicators):
            suggestions.append("⚠️ Conflicts detected - use git rebase for clean resolution")
            suggestions.append("🔧 Consider mcp__github__update_pull_request_branch")
            suggestions.append("📋 Document conflict resolution in PR comments")
        
        # Large diff analysis
        if "diff" in param_text or "files_changed" in str(params):
            files_changed = params.get("files_changed", 0)
            if files_changed > 20:
                suggestions.append("📊 Large PR detected - consider breaking into smaller PRs")
                suggestions.append("🎯 Focus reviews on critical changes first")
        
        return suggestions
    
    def _detect_multi_pr_operation(self, context: Dict[str, Any]) -> bool:
        """Detect if this is part of a multi-PR operation."""
        recent_tools = context.get("recent_tools", [])
        
        pr_tool_count = sum(1 for tool in recent_tools 
                           if any(pattern in tool.lower() for pattern in self.pr_patterns))
        
        return pr_tool_count > 2
    
    def get_pattern_suggestions(self) -> List[str]:
        """Get general PR workflow pattern suggestions."""
        return [
            "🔄 Use draft PRs for work-in-progress features",
            "📋 Include comprehensive PR descriptions with context",
            "👥 Request appropriate reviewers based on code changes",
            "✅ Ensure CI/CD checks pass before requesting review",
            "🎯 Use squash merge for feature branches",
            "📝 Add meaningful commit messages for review clarity",
            "🐝 Consider GitHub swarm coordination for complex PR workflows"
        ]