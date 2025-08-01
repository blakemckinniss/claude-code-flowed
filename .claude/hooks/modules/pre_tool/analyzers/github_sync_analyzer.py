"""GitHub Sync Analyzer - Claude Flow Integration Phase 2

Multi-repository synchronization intelligence with cross-repo coordination,
dependency management, and workflow orchestration optimization.
"""

from typing import Dict, Any, List, Optional, Set
from ..core.workflow_validator import HiveWorkflowValidator, ValidationResult, ValidationSeverity


class GitHubSyncAnalyzer(HiveWorkflowValidator):
    """Analyzes GitHub synchronization operations across repositories."""
    
    def __init__(self, priority: int = 800):
        super().__init__(priority)
        self.name = "GitHubSyncAnalyzer"
        self.sync_patterns = [
            "sync", "fork", "upstream", "downstream", "mirror",
            "merge", "rebase", "cherry-pick", "backport",
            "cross-repo", "multi-repo", "monorepo"
        ]
        
        # Repository relationship patterns
        self.relationship_indicators = {
            "fork": ["fork", "origin", "upstream"],
            "dependency": ["dependency", "package", "library", "module"],
            "template": ["template", "boilerplate", "scaffold"],
            "mirror": ["mirror", "backup", "sync", "replica"]
        }
    
    def get_validator_name(self) -> str:
        """Get name of this validator."""
        return "github_sync_analyzer"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context) -> Optional[ValidationResult]:
        """Validate sync workflow and provide guidance."""
        
        analysis = self.analyze(tool_name, tool_input, {"recent_tools": []})
        
        if analysis["priority"] == 0:
            return None
        
        suggestions = analysis.get("suggestions", [])
        if not suggestions:
            return None
        
        return ValidationResult(
            severity=ValidationSeverity.SUGGEST,
            violation_type=None,
            message=f"Sync workflow optimization available: {suggestions[0]}",
            suggested_alternative=" | ".join(suggestions[:3]),
            hive_guidance=f"🐝 GitHub Sync intelligence from {self.name}",
            priority_score=self.priority
        )
    
    def analyze(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze synchronization-related operations."""
        if not self._is_sync_operation(tool_name, params, context):
            return {"priority": 0, "suggestions": []}
        
        analysis = {
            "priority": self.priority,
            "analyzer": self.name,
            "suggestions": [],
            "coordination_needed": False,
            "github_context": True,
            "sync_insights": {}
        }
        
        # Multi-repository pattern analysis
        sync_insights = self._analyze_sync_patterns(tool_name, params, context)
        analysis["sync_insights"] = sync_insights
        analysis["suggestions"].extend(sync_insights.get("suggestions", []))
        
        # Repository relationship analysis
        relationship_suggestions = self._analyze_repo_relationships(tool_name, params)
        analysis["suggestions"].extend(relationship_suggestions)
        
        # Cross-repository workflow coordination
        workflow_suggestions = self._analyze_cross_repo_workflows(tool_name, params, context)
        analysis["suggestions"].extend(workflow_suggestions)
        
        # Dependency synchronization analysis
        dependency_suggestions = self._analyze_dependency_sync(params, context)
        analysis["suggestions"].extend(dependency_suggestions)
        
        # Always coordinate for sync operations due to complexity
        analysis["coordination_needed"] = True
        
        return analysis
    
    def _is_sync_operation(self, tool_name: str, params: Dict[str, Any], 
                          context: Dict[str, Any]) -> bool:
        """Check if operation involves synchronization."""
        # Check tool name for sync patterns
        if any(pattern in tool_name.lower() for pattern in self.sync_patterns):
            return True
        
        # Check parameters for sync-related content
        param_text = str(params).lower()
        if any(pattern in param_text for pattern in self.sync_patterns):
            return True
        
        # Check for multi-repository operations in context
        if self._detect_multi_repo_context(context):
            return True
        
        # Check for fork operations
        if "fork" in tool_name.lower():
            return True
        
        # Check for branch operations that might be sync-related
        if "branch" in tool_name.lower() and any(indicator in param_text 
                                                for indicators in self.relationship_indicators.values()
                                                for indicator in indicators):
            return True
        
        return False
    
    def _analyze_sync_patterns(self, tool_name: str, params: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze synchronization patterns and strategies."""
        insights = {
            "sync_type": "unknown",
            "complexity": "medium",
            "repositories_involved": 1,
            "suggestions": []
        }
        
        # Detect sync type
        if "fork" in tool_name.lower():
            insights["sync_type"] = "fork"
            insights["suggestions"].append("🍴 Fork operation - set up upstream remote")
            insights["suggestions"].append("🔄 Plan regular sync strategy with upstream")
        
        # Multi-repository detection
        repo_count = self._count_repositories_in_context(context)
        insights["repositories_involved"] = repo_count
        
        if repo_count > 1:
            insights["complexity"] = "high"
            insights["suggestions"].append("🐝 Multi-repo operation - use swarm coordination")
            insights["suggestions"].append("⚡ Consider parallel processing for efficiency")
        
        # Cross-repository patterns
        if self._detect_cross_repo_patterns(params, context):
            insights["sync_type"] = "cross-repo"
            insights["suggestions"].append("🔗 Cross-repo sync detected - ensure dependency order")
            insights["suggestions"].append("📋 Document sync process for team knowledge")
        
        return insights
    
    def _analyze_repo_relationships(self, tool_name: str, params: Dict[str, Any]) -> List[str]:
        """Analyze repository relationships and suggest sync strategies."""
        suggestions = []
        
        # Repository ownership analysis
        owner = params.get("owner", "")
        repo = params.get("repo", "")
        
        # Fork relationship analysis
        if "fork" in tool_name.lower():
            suggestions.append("🎯 Set up upstream tracking: git remote add upstream <original-repo>")
            suggestions.append("🔄 Create sync workflow: fetch upstream, merge, push")
            suggestions.append("📅 Schedule regular upstream sync (weekly/monthly)")
        
        # Organization vs personal repo patterns
        if owner and len(owner.split("-")) > 1:  # Likely organization
            suggestions.append("🏢 Organization repository - coordinate with team")
            suggestions.append("📋 Ensure proper access controls and branch protection")
        
        # Repository naming patterns that suggest relationships
        if repo:
            if "template" in repo.lower():
                suggestions.append("📝 Template repository - maintain generic examples")
            elif "fork" in repo.lower() or "clone" in repo.lower():
                suggestions.append("🔄 Fork/clone repository - track upstream changes")
            elif any(suffix in repo.lower() for suffix in ["-api", "-client", "-server"]):
                suggestions.append("🔗 Component repository - sync with related components")
        
        return suggestions
    
    def _analyze_cross_repo_workflows(self, tool_name: str, params: Dict[str, Any],
                                     context: Dict[str, Any]) -> List[str]:
        """Analyze cross-repository workflow coordination needs."""
        suggestions = []
        
        # Multi-repository operation detection
        if self._detect_multi_repo_context(context):
            suggestions.append("🐝 Multi-repo workflow - use GitHub swarm coordination")
            suggestions.append("📊 Track changes across all repositories")
            suggestions.append("🔄 Implement atomic operations where possible")
        
        # Dependency chain analysis
        if self._detect_dependency_chain(context):
            suggestions.append("📦 Dependency chain detected - respect build order")
            suggestions.append("🧪 Run integration tests after multi-repo changes")
            suggestions.append("📅 Coordinate release timing across dependent repos")
        
        # Workflow automation opportunities
        suggestions.append("🤖 Consider GitHub Actions for cross-repo automation")
        suggestions.append("🔄 Set up repository dispatch for workflow triggers")
        
        # Conflict prevention
        suggestions.append("🛡️ Implement change coordination to prevent conflicts")
        suggestions.append("📋 Use change management process for multi-repo updates")
        
        return suggestions
    
    def _analyze_dependency_sync(self, params: Dict[str, Any], 
                                context: Dict[str, Any]) -> List[str]:
        """Analyze dependency synchronization needs."""
        suggestions = []
        
        # Package/dependency file updates
        dependency_files = ["package.json", "requirements.txt", "Cargo.toml", 
                          "pom.xml", "build.gradle", "pyproject.toml"]
        
        file_path = params.get("path", "")
        if any(dep_file in file_path for dep_file in dependency_files):
            suggestions.append("📦 Dependency update detected - sync across related repos")
            suggestions.append("🧪 Run comprehensive tests after dependency changes")
            suggestions.append("🔍 Check for breaking changes in updated dependencies")
        
        # Version synchronization
        content = params.get("content", "")
        if content and any(version_indicator in content.lower() 
                          for version_indicator in ["version", "tag", "commit"]):
            suggestions.append("🏷️ Version update - ensure consistency across repositories")
            suggestions.append("📋 Update changelog and release notes accordingly")
        
        # Security dependency updates
        if "security" in str(params).lower() or "vulnerability" in str(params).lower():
            suggestions.append("🚨 Security update - prioritize sync across all repositories")
            suggestions.append("⚡ Consider emergency deployment process")
        
        return suggestions
    
    def _detect_multi_repo_context(self, context: Dict[str, Any]) -> bool:
        """Detect if operation involves multiple repositories."""
        recent_tools = context.get("recent_tools", [])
        
        # Look for different repository references
        repos_mentioned = set()
        for tool in recent_tools:
            if "github" in tool.lower():
                repos_mentioned.add(tool)
        
        return len(repos_mentioned) > 1
    
    def _count_repositories_in_context(self, context: Dict[str, Any]) -> int:
        """Count repositories involved in current context."""
        recent_tools = context.get("recent_tools", [])
        
        # Extract unique repository operations
        repo_operations = set()
        for tool in recent_tools:
            if any(github_op in tool.lower() for github_op in ["github", "repo", "fork"]):
                repo_operations.add(tool)
        
        return max(1, len(repo_operations))
    
    def _detect_cross_repo_patterns(self, params: Dict[str, Any], 
                                   context: Dict[str, Any]) -> bool:
        """Detect cross-repository operation patterns."""
        # Check for references to multiple repositories in parameters
        param_text = str(params).lower()
        cross_repo_indicators = ["upstream", "origin", "fork", "template", "mirror"]
        
        return any(indicator in param_text for indicator in cross_repo_indicators)
    
    def _detect_dependency_chain(self, context: Dict[str, Any]) -> bool:
        """Detect dependency relationships between repositories."""
        recent_tools = context.get("recent_tools", [])
        
        # Look for package management or dependency-related operations
        dependency_tools = sum(1 for tool in recent_tools 
                             if any(dep_keyword in tool.lower() 
                                   for dep_keyword in ["package", "dependency", "install", "update"]))
        
        return dependency_tools > 1
    
    def get_pattern_suggestions(self) -> List[str]:
        """Get general synchronization pattern suggestions."""
        return [
            "🔄 Establish regular sync schedule with upstream repositories",
            "🐝 Use GitHub swarm coordination for complex multi-repo operations",
            "📦 Implement dependency version synchronization across repositories",
            "🤖 Automate cross-repository workflows with GitHub Actions",
            "🔗 Set up proper remote tracking for fork relationships",
            "📋 Document synchronization processes for team knowledge",
            "🧪 Run integration tests after multi-repository changes",
            "🛡️ Implement change coordination to prevent conflicts",
            "📅 Coordinate release timing across dependent repositories",
            "⚡ Use parallel processing for efficient multi-repo operations"
        ]