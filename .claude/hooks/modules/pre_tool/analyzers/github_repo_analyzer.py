"""GitHub Repository Analyzer - Claude Flow Integration Phase 2

Comprehensive repository health monitoring, maintenance suggestions,
and workflow optimization for GitHub repository operations.
"""

from typing import Dict, Any, List, Optional
from ..core.workflow_validator import HiveWorkflowValidator, ValidationResult, ValidationSeverity


class GitHubRepoAnalyzer(HiveWorkflowValidator):
    """Analyzes GitHub repository operations for health and optimization."""
    
    def __init__(self, priority: int = 805):
        super().__init__(priority)
        self.name = "GitHubRepoAnalyzer"
        self.repo_patterns = [
            "repository", "repo", "fork", "clone", "settings",
            "branch", "protection", "webhook", "collaborator",
            "topics", "description", "readme", "license"
        ]
        
        # Repository health indicators
        self.health_files = ["README.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md", 
                           "CODE_OF_CONDUCT.md", ".gitignore", "CHANGELOG.md"]
        self.config_files = [".github/workflows/", ".github/ISSUE_TEMPLATE/", 
                           ".github/PULL_REQUEST_TEMPLATE/", "pyproject.toml", 
                           "package.json", "Dockerfile"]
    
    def get_validator_name(self) -> str:
        """Get name of this validator."""
        return "github_repo_analyzer"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context) -> Optional[ValidationResult]:
        """Validate repository workflow and provide guidance."""
        
        analysis = self.analyze(tool_name, tool_input, {"recent_tools": []})
        
        if analysis["priority"] == 0:
            return None
        
        suggestions = analysis.get("suggestions", [])
        if not suggestions:
            return None
        
        return ValidationResult(
            severity=ValidationSeverity.SUGGEST,
            violation_type=None,
            message=f"Repository workflow optimization available: {suggestions[0]}",
            suggested_alternative=" | ".join(suggestions[:3]),
            hive_guidance=f"🐝 GitHub Repository intelligence from {self.name}",
            priority_score=self.priority
        )
    
    def analyze(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze repository-related operations."""
        if not self._is_repo_operation(tool_name, params):
            return {"priority": 0, "suggestions": []}
        
        analysis = {
            "priority": self.priority,
            "analyzer": self.name,
            "suggestions": [],
            "coordination_needed": False,
            "github_context": True,
            "repo_insights": {}
        }
        
        # Repository health analysis
        health_insights = self._analyze_repo_health(tool_name, params, context)
        analysis["repo_insights"] = health_insights
        analysis["suggestions"].extend(health_insights.get("suggestions", []))
        
        # Branch and protection analysis
        branch_suggestions = self._analyze_branch_strategy(tool_name, params)
        analysis["suggestions"].extend(branch_suggestions)
        
        # Workflow and automation analysis  
        workflow_suggestions = self._analyze_repo_automation(tool_name, params, context)
        analysis["suggestions"].extend(workflow_suggestions)
        
        # Collaboration and maintenance analysis
        maintenance_suggestions = self._analyze_repo_maintenance(params)
        analysis["suggestions"].extend(maintenance_suggestions)
        
        # Set coordination flag for complex repository operations
        analysis["coordination_needed"] = len(analysis["suggestions"]) > 4
        
        return analysis
    
    def _is_repo_operation(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Check if operation is repository-related."""
        # Check tool name
        if any(pattern in tool_name.lower() for pattern in self.repo_patterns):
            return True
        
        # Check parameters for repo-related content
        param_text = str(params).lower()
        if any(pattern in param_text for pattern in self.repo_patterns):
            return True
        
        # Check for GitHub repository tools
        github_repo_tools = [
            "mcp__github__create_repository",
            "mcp__github__fork_repository", 
            "mcp__github__search_repositories",
            "mcp__github__get_file_contents",
            "mcp__github__create_or_update_file",
            "mcp__github__delete_file",
            "mcp__github__list_branches",
            "mcp__github__create_branch"
        ]
        
        return tool_name in github_repo_tools
    
    def _analyze_repo_health(self, tool_name: str, params: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze repository health and completeness."""
        insights = {
            "health_score": 0.5,
            "missing_files": [],
            "configuration_gaps": [],
            "suggestions": []
        }
        
        # Repository creation analysis
        if "create_repository" in tool_name:
            insights["suggestions"].append("📋 Initialize with README, LICENSE, and .gitignore")
            insights["suggestions"].append("🔧 Set up branch protection rules for main branch")
            insights["suggestions"].append("📝 Add repository description and topics")
            
            # Check initialization parameters
            auto_init = params.get("autoInit", False)
            if not auto_init:
                insights["suggestions"].append("🚀 Consider auto-initialization for faster setup")
        
        # File operations analysis
        if "create_or_update_file" in tool_name or "delete_file" in tool_name:
            file_path = params.get("path", "")
            
            # Check for important file updates
            if file_path.lower() in [f.lower() for f in self.health_files]:
                insights["suggestions"].append(f"📄 Updating {file_path} - ensure content quality")
            
            # Configuration file updates
            if any(config in file_path for config in self.config_files):
                insights["suggestions"].append("⚙️ Configuration change - test thoroughly")
        
        # Repository settings analysis
        private = params.get("private", False)
        if private:
            insights["suggestions"].append("🔒 Private repository - ensure proper collaborator access")
        else:
            insights["suggestions"].append("🌍 Public repository - review security and documentation")
        
        return insights
    
    def _analyze_branch_strategy(self, tool_name: str, params: Dict[str, Any]) -> List[str]:
        """Analyze branching strategy and protection rules."""
        suggestions = []
        
        # Branch creation analysis
        if "create_branch" in tool_name:
            branch_name = params.get("branch", "")
            from_branch = params.get("from_branch", "main")
            
            # Branch naming conventions
            if branch_name:
                if "/" not in branch_name:
                    suggestions.append("🌿 Consider branch naming: feature/name, fix/name, etc.")
                
                if "main" in branch_name.lower() or "master" in branch_name.lower():
                    suggestions.append("⚠️ Avoid creating branches similar to main branch names")
            
            # Source branch analysis
            if from_branch != "main":
                suggestions.append(f"🔄 Branching from {from_branch} - ensure it's up to date")
        
        # Branch listing and management
        if "list_branches" in tool_name:
            suggestions.append("🧹 Review stale branches - consider cleanup automation")
            suggestions.append("🔒 Ensure main branch has protection rules")
        
        # General branch strategy recommendations
        suggestions.append("📋 Implement consistent branch naming strategy")
        suggestions.append("🛡️ Set up branch protection rules with required reviews")
        
        return suggestions
    
    def _analyze_repo_automation(self, tool_name: str, params: Dict[str, Any],
                                context: Dict[str, Any]) -> List[str]:
        """Analyze repository automation opportunities."""
        suggestions = []
        
        # Workflow automation opportunities
        if self._detect_repetitive_operations(context):
            suggestions.append("🤖 Repetitive operations detected - consider GitHub Actions")
            suggestions.append("⚡ Automate file updates with workflows")
        
        # File pattern analysis for automation
        if "create_or_update_file" in tool_name:
            file_path = params.get("path", "")
            
            # Documentation automation
            if "README" in file_path or ".md" in file_path:
                suggestions.append("📚 Consider auto-generating docs from code comments")
            
            # Configuration automation
            if "package.json" in file_path or "pyproject.toml" in file_path:
                suggestions.append("📦 Automate dependency updates with Dependabot")
            
            # CI/CD file updates
            if ".github/workflows" in file_path:
                suggestions.append("🔧 Test workflow changes in separate branch first")
        
        # Security automation
        suggestions.append("🔐 Enable Dependabot security updates")
        suggestions.append("🔍 Set up CodeQL code scanning")
        
        return suggestions
    
    def _analyze_repo_maintenance(self, params: Dict[str, Any]) -> List[str]:
        """Analyze repository maintenance needs."""
        suggestions = []
        
        # Repository description and metadata
        description = params.get("description", "")
        if not description or len(description) < 20:
            suggestions.append("📝 Add comprehensive repository description")
        
        # Topics and discoverability
        topics = params.get("topics", [])
        if not topics:
            suggestions.append("🏷️ Add relevant topics for better discoverability")
        
        # Collaboration setup
        suggestions.append("👥 Set up CONTRIBUTING.md for contributor guidelines")
        suggestions.append("📋 Create issue templates for better bug reports")
        suggestions.append("🔄 Add pull request template for consistent reviews")
        
        # Security and compliance
        suggestions.append("🛡️ Add SECURITY.md with vulnerability reporting process")
        suggestions.append("⚖️ Ensure appropriate LICENSE file is present")
        suggestions.append("📜 Consider adding CODE_OF_CONDUCT.md")
        
        # Maintenance automation
        suggestions.append("🧹 Set up stale issue/PR management")
        suggestions.append("📊 Enable repository insights and analytics")
        
        return suggestions
    
    def _detect_repetitive_operations(self, context: Dict[str, Any]) -> bool:
        """Detect repetitive repository operations that could be automated."""
        recent_tools = context.get("recent_tools", [])
        
        # Count file operations
        file_operations = sum(1 for tool in recent_tools 
                             if any(op in tool.lower() 
                                   for op in ["create_or_update_file", "delete_file"]))
        
        return file_operations > 3
    
    def get_pattern_suggestions(self) -> List[str]:
        """Get general repository management pattern suggestions."""
        return [
            "📋 Maintain comprehensive README with setup instructions",
            "🔒 Implement branch protection rules for main branches",
            "🤖 Automate repetitive tasks with GitHub Actions", 
            "📝 Use templates for issues and pull requests",
            "🏷️ Add descriptive topics and tags for discoverability",
            "🔐 Enable security features: Dependabot, CodeQL, secret scanning",
            "👥 Provide clear contribution guidelines and code of conduct",
            "📊 Monitor repository health with insights and analytics",
            "🧹 Regularly clean up stale branches and issues",
            "📦 Keep dependencies updated and secure"
        ]