"""GitHub Issue Analyzer - Claude Flow Integration Phase 2

Intelligent issue triage, labeling automation, and workflow optimization
for GitHub issue management operations.
"""

from typing import Dict, Any, List, Optional
from ..core.workflow_validator import HiveWorkflowValidator, ValidationResult, ValidationSeverity


class GitHubIssueAnalyzer(HiveWorkflowValidator):
    """Analyzes GitHub issue operations for workflow optimization."""
    
    def __init__(self, priority: int = 815):
        super().__init__(priority)
        self.name = "GitHubIssueAnalyzer" 
        self.issue_patterns = [
            "issue", "bug", "feature", "enhancement", "question",
            "documentation", "duplicate", "invalid", "wontfix", 
            "help wanted", "good first issue", "priority", "milestone"
        ]
        
        # Issue type classification
        self.bug_keywords = ["bug", "error", "exception", "crash", "fail", "broken"]
        self.feature_keywords = ["feature", "enhancement", "improvement", "add", "implement"]
        self.question_keywords = ["question", "help", "how to", "clarification", "discuss"]
        self.doc_keywords = ["documentation", "docs", "readme", "guide", "tutorial"]
    
    def get_validator_name(self) -> str:
        """Get name of this validator."""
        return "github_issue_analyzer"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context) -> Optional[ValidationResult]:
        """Validate issue workflow and provide guidance."""
        
        analysis = self.analyze(tool_name, tool_input, {"recent_tools": []})
        
        if analysis["priority"] == 0:
            return None
        
        suggestions = analysis.get("suggestions", [])
        if not suggestions:
            return None
        
        return ValidationResult(
            severity=ValidationSeverity.SUGGEST,
            violation_type=None,
            message=f"Issue workflow optimization available: {suggestions[0]}",
            suggested_alternative=" | ".join(suggestions[:3]),
            hive_guidance=f"🐝 GitHub Issue intelligence from {self.name}",
            priority_score=self.priority
        )
    
    def analyze(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze issue-related operations."""
        if not self._is_issue_operation(tool_name, params):
            return {"priority": 0, "suggestions": []}
        
        analysis = {
            "priority": self.priority,
            "analyzer": self.name,
            "suggestions": [],
            "coordination_needed": False,
            "github_context": True,
            "issue_insights": {}
        }
        
        # Analyze issue content and classification
        issue_insights = self._analyze_issue_content(tool_name, params)
        analysis["issue_insights"] = issue_insights
        analysis["suggestions"].extend(issue_insights.get("suggestions", []))
        
        # Triage and labeling suggestions  
        triage_suggestions = self._analyze_issue_triage(tool_name, params, issue_insights)
        analysis["suggestions"].extend(triage_suggestions)
        
        # Workflow automation suggestions
        workflow_suggestions = self._analyze_issue_workflow(tool_name, params, context)
        analysis["suggestions"].extend(workflow_suggestions)
        
        # Assignment and milestone suggestions
        assignment_suggestions = self._analyze_issue_assignment(params)
        analysis["suggestions"].extend(assignment_suggestions)
        
        # Set coordination flag for complex operations
        analysis["coordination_needed"] = len(analysis["suggestions"]) > 3
        
        return analysis
    
    def _is_issue_operation(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Check if operation is issue-related."""
        # Check tool name
        if "issue" in tool_name.lower():
            return True
        
        # Check parameters for issue-related content
        param_text = str(params).lower()
        if any(pattern in param_text for pattern in self.issue_patterns):
            return True
        
        # Check for GitHub issue MCP tools
        github_issue_tools = [
            "mcp__github__create_issue",
            "mcp__github__get_issue", 
            "mcp__github__update_issue",
            "mcp__github__list_issues",
            "mcp__github__get_issue_comments",
            "mcp__github__add_issue_comment",
            "mcp__github__search_issues"
        ]
        
        return tool_name in github_issue_tools
    
    def _analyze_issue_content(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze issue content for classification and quality."""
        insights = {
            "issue_type": "unknown",
            "priority_level": "medium", 
            "quality_score": 0.5,
            "suggestions": []
        }
        
        title = params.get("title", "").lower()
        body = params.get("body", "").lower()
        content = f"{title} {body}"
        
        # Classify issue type
        if any(keyword in content for keyword in self.bug_keywords):
            insights["issue_type"] = "bug"
            insights["suggestions"].append("🐛 Bug report detected - ensure reproduction steps included")
        elif any(keyword in content for keyword in self.feature_keywords):
            insights["issue_type"] = "feature"
            insights["suggestions"].append("✨ Feature request detected - add user story format")
        elif any(keyword in content for keyword in self.question_keywords):
            insights["issue_type"] = "question" 
            insights["suggestions"].append("❓ Question detected - consider discussions instead")
        elif any(keyword in content for keyword in self.doc_keywords):
            insights["issue_type"] = "documentation"
            insights["suggestions"].append("📚 Documentation issue - link to relevant sections")
        
        # Priority analysis
        priority_indicators = {
            "critical": ["critical", "urgent", "broken", "crash", "security"],
            "high": ["important", "major", "blocker", "regression"],
            "low": ["minor", "nice to have", "enhancement", "cosmetic"]
        }
        
        for priority, keywords in priority_indicators.items():
            if any(keyword in content for keyword in keywords):
                insights["priority_level"] = priority
                break
        
        # Quality assessment
        quality_factors = {
            "has_title": len(params.get("title", "")) > 10,
            "has_description": len(params.get("body", "")) > 50,
            "has_labels": bool(params.get("labels", [])),
            "has_assignees": bool(params.get("assignees", [])),
            "has_milestone": bool(params.get("milestone"))
        }
        
        insights["quality_score"] = sum(quality_factors.values()) / len(quality_factors)
        
        # Quality improvement suggestions
        if not quality_factors["has_description"]:
            insights["suggestions"].append("📝 Add detailed issue description for clarity")
        
        if insights["quality_score"] < 0.6:
            insights["suggestions"].append("🎯 Improve issue quality with templates and guidelines")
        
        return insights
    
    def _analyze_issue_triage(self, tool_name: str, params: Dict[str, Any], 
                             insights: Dict[str, Any]) -> List[str]:
        """Analyze issue triage and suggest labeling."""
        suggestions = []
        
        issue_type = insights.get("issue_type", "unknown")
        priority = insights.get("priority_level", "medium")
        
        # Type-based labeling
        type_labels = {
            "bug": ["bug", "needs-investigation"],
            "feature": ["enhancement", "feature-request"],
            "question": ["question", "needs-clarification"],  
            "documentation": ["documentation", "good-first-issue"]
        }
        
        if issue_type in type_labels:
            suggested_labels = type_labels[issue_type]
            suggestions.append(f"🏷️ Suggested labels: {', '.join(suggested_labels)}")
        
        # Priority labeling
        if priority == "critical":
            suggestions.append("🚨 Critical priority - add 'priority:critical' label")
        elif priority == "high": 
            suggestions.append("⚡ High priority - add 'priority:high' label")
        elif priority == "low":
            suggestions.append("📌 Low priority - add 'priority:low' label")
        
        # Automated triage suggestions
        if "create_issue" in tool_name:
            suggestions.append("🤖 Consider using issue templates for consistent triage")
            suggestions.append("📋 Auto-assign based on component/area labels")
        
        # Duplicate detection
        title = params.get("title", "")
        if title and len(title) > 10:
            suggestions.append("🔍 Check for duplicate issues before creating")
        
        return suggestions
    
    def _analyze_issue_workflow(self, tool_name: str, params: Dict[str, Any],
                               context: Dict[str, Any]) -> List[str]:
        """Analyze issue workflow automation opportunities."""
        suggestions = []
        
        # Workflow automation based on operation
        if "create_issue" in tool_name:
            suggestions.append("⚡ Auto-assign to component owners using CODEOWNERS")
            suggestions.append("📅 Consider auto-milestone assignment for release planning")
        
        if "update_issue" in tool_name:
            state = params.get("state", "")
            if state == "closed":
                suggestions.append("✅ Consider linking closing commit/PR for traceability")
        
        # Multi-issue operations
        if self._detect_bulk_operations(context):
            suggestions.append("🐝 Bulk issue operations - consider GitHub swarm coordination")
            suggestions.append("📊 Use issue templates for consistency across bulk creation")
        
        # Issue linking
        body = params.get("body", "")
        if "related" in body.lower() or "fixes" in body.lower():
            suggestions.append("🔗 Use GitHub keywords (fixes, closes, resolves) for auto-linking")
        
        # Project board integration
        suggestions.append("📋 Consider adding to project board for tracking")
        
        return suggestions
    
    def _analyze_issue_assignment(self, params: Dict[str, Any]) -> List[str]:
        """Analyze issue assignment and suggest optimizations."""
        suggestions = []
        
        assignees = params.get("assignees", [])
        milestone = params.get("milestone")
        labels = params.get("labels", [])
        
        # Assignment analysis
        if not assignees:
            suggestions.append("👤 Consider assigning to appropriate team member")
        elif len(assignees) > 3:
            suggestions.append("👥 Multiple assignees - clarify primary responsibility")
        
        # Milestone analysis
        if not milestone and any(label in ["bug", "critical"] for label in labels):
            suggestions.append("🎯 Critical issues should have milestone assignment")
        
        # Label-based assignment suggestions
        if "good-first-issue" in labels:
            suggestions.append("🌟 Good first issue - ensure beginner-friendly description")
        
        if "help-wanted" in labels:
            suggestions.append("🤝 Help wanted - provide clear contribution guidelines")
        
        return suggestions
    
    def _detect_bulk_operations(self, context: Dict[str, Any]) -> bool:
        """Detect bulk issue operations."""
        recent_tools = context.get("recent_tools", [])
        
        issue_tool_count = sum(1 for tool in recent_tools 
                              if "issue" in tool.lower())
        
        return issue_tool_count > 3
    
    def get_pattern_suggestions(self) -> List[str]:
        """Get general issue management pattern suggestions."""
        return [
            "📋 Use issue templates for consistent bug reports and feature requests",
            "🏷️ Implement consistent labeling strategy for better organization", 
            "🤖 Set up automated issue triage with GitHub Actions",
            "📊 Use project boards for visual issue tracking",
            "👥 Implement CODEOWNERS for automatic assignment",
            "🔗 Use linking keywords to connect issues with PRs",
            "📅 Regular milestone planning for release organization",
            "🐝 Consider GitHub swarm coordination for bulk issue management"
        ]