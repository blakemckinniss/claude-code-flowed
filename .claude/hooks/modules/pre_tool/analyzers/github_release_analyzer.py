"""GitHub Release Analyzer - Claude Flow Integration Phase 2

Advanced release coordination with changelog automation, version management,
and deployment workflow optimization for GitHub releases.
"""

from typing import Dict, Any, List, Optional
import re
from ..core.workflow_validator import HiveWorkflowValidator, ValidationResult, ValidationSeverity


class GitHubReleaseAnalyzer(HiveWorkflowValidator):
    """Analyzes GitHub release operations for workflow optimization."""
    
    def __init__(self, priority: int = 810):
        super().__init__(priority)
        self.name = "GitHubReleaseAnalyzer"
        self.release_patterns = [
            "release", "tag", "version", "deploy", "publish",
            "changelog", "notes", "milestone", "semver"
        ]
        
        # Version patterns
        self.semver_pattern = re.compile(r'v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?')
        self.version_types = {
            "major": "Breaking changes or major new features",
            "minor": "New features, backward compatible", 
            "patch": "Bug fixes, backward compatible",
            "prerelease": "Alpha/beta/rc versions"
        }
    
    def get_validator_name(self) -> str:
        """Get name of this validator."""
        return "github_release_analyzer"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context) -> Optional[ValidationResult]:
        """Validate release workflow and provide guidance."""
        
        analysis = self.analyze(tool_name, tool_input, {"recent_tools": []})
        
        if analysis["priority"] == 0:
            return None
        
        suggestions = analysis.get("suggestions", [])
        if not suggestions:
            return None
        
        return ValidationResult(
            severity=ValidationSeverity.SUGGEST,
            violation_type=None,
            message=f"Release workflow optimization available: {suggestions[0]}",
            suggested_alternative=" | ".join(suggestions[:3]),
            hive_guidance=f"🐝 GitHub Release intelligence from {self.name}",
            priority_score=self.priority
        )
    
    def analyze(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze release-related operations."""
        if not self._is_release_operation(tool_name, params):
            return {"priority": 0, "suggestions": []}
        
        analysis = {
            "priority": self.priority,
            "analyzer": self.name,
            "suggestions": [],
            "coordination_needed": False,
            "github_context": True,
            "release_insights": {}
        }
        
        # Version analysis and validation
        version_insights = self._analyze_version_strategy(tool_name, params)
        analysis["release_insights"] = version_insights
        analysis["suggestions"].extend(version_insights.get("suggestions", []))
        
        # Changelog and release notes analysis
        changelog_suggestions = self._analyze_changelog_quality(params)
        analysis["suggestions"].extend(changelog_suggestions)
        
        # Release workflow coordination
        workflow_suggestions = self._analyze_release_workflow(tool_name, params, context)
        analysis["suggestions"].extend(workflow_suggestions)
        
        # Deployment and distribution analysis
        deployment_suggestions = self._analyze_deployment_strategy(params)
        analysis["suggestions"].extend(deployment_suggestions)
        
        # Set coordination flag for complex releases
        analysis["coordination_needed"] = len(analysis["suggestions"]) > 3
        
        return analysis
    
    def _is_release_operation(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """Check if operation is release-related."""
        # Check tool name
        if any(pattern in tool_name.lower() for pattern in self.release_patterns):
            return True
        
        # Check parameters for release-related content
        param_text = str(params).lower()
        if any(pattern in param_text for pattern in self.release_patterns):
            return True
        
        # Check for version-like patterns
        if self.semver_pattern.search(str(params)):
            return True
        
        # Check for GitHub release tools (when they exist)
        github_release_tools = [
            "mcp__github__create_release",
            "mcp__github__get_release",
            "mcp__github__update_release",
            "mcp__github__list_releases",
            "mcp__github__delete_release"
        ]
        
        return tool_name in github_release_tools
    
    def _analyze_version_strategy(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze version numbering and semantic versioning compliance."""
        insights = {
            "version_detected": None,
            "version_type": "unknown",
            "semver_compliant": False,
            "suggestions": []
        }
        
        # Extract version information
        version_str = self._extract_version(params)
        if version_str:
            insights["version_detected"] = version_str
            
            # Check semantic versioning compliance
            semver_match = self.semver_pattern.match(version_str)
            if semver_match:
                insights["semver_compliant"] = True
                major, minor, patch, prerelease = semver_match.groups()
                
                # Determine version type
                if prerelease:
                    insights["version_type"] = "prerelease"
                    insights["suggestions"].append(f"🧪 Prerelease version detected: {prerelease}")
                elif major != "0":
                    if minor == "0" and patch == "0":
                        insights["version_type"] = "major"
                        insights["suggestions"].append("🚀 Major release - ensure breaking changes documented")
                    elif patch == "0":
                        insights["version_type"] = "minor"
                        insights["suggestions"].append("✨ Minor release - highlight new features")
                    else:
                        insights["version_type"] = "patch"
                        insights["suggestions"].append("🔧 Patch release - focus on bug fixes")
            else:
                insights["suggestions"].append("📏 Consider using semantic versioning (semver.org)")
        
        # Version strategy recommendations
        if "create" in tool_name:
            insights["suggestions"].append("📋 Verify version follows project versioning strategy")
            insights["suggestions"].append("🏷️ Ensure corresponding git tag is created")
        
        return insights
    
    def _extract_version(self, params: Dict[str, Any]) -> Optional[str]:
        """Extract version string from parameters."""
        # Common version parameter names
        version_keys = ["version", "tag_name", "name", "title"]
        
        for key in version_keys:
            if key in params:
                value = str(params[key])
                if self.semver_pattern.search(value):
                    return value
        
        # Search in body/description
        body = params.get("body", "")
        if body:
            match = self.semver_pattern.search(body)
            if match:
                return match.group(0)
        
        return None
    
    def _analyze_changelog_quality(self, params: Dict[str, Any]) -> List[str]:
        """Analyze changelog and release notes quality."""
        suggestions = []
        
        body = params.get("body", "")
        notes = params.get("release_notes", "")
        changelog = f"{body} {notes}".strip()
        
        if not changelog or len(changelog) < 50:
            suggestions.append("📝 Add comprehensive release notes and changelog")
            suggestions.append("📋 Include: Features, Bug Fixes, Breaking Changes, Dependencies")
            return suggestions
        
        # Analyze changelog structure
        changelog_lower = changelog.lower()
        
        # Check for standard sections
        standard_sections = {
            "features": ["feature", "new", "add", "implement"],
            "fixes": ["fix", "bug", "issue", "resolve"],
            "breaking": ["breaking", "breaking change", "incompatible"],
            "dependencies": ["dependency", "upgrade", "update", "bump"]
        }
        
        missing_sections = []
        for section, keywords in standard_sections.items():
            if not any(keyword in changelog_lower for keyword in keywords):
                missing_sections.append(section)
        
        if missing_sections:
            suggestions.append(f"📋 Consider adding sections: {', '.join(missing_sections)}")
        
        # Check for links and references
        if "http" not in changelog and "#" not in changelog:
            suggestions.append("🔗 Add links to issues/PRs for better traceability")
        
        # Check for contributor acknowledgment
        if "thank" not in changelog_lower and "contributor" not in changelog_lower:
            suggestions.append("👥 Consider acknowledging contributors")
        
        # Migration guide for breaking changes
        if any(keyword in changelog_lower for keyword in ["breaking", "incompatible"]):
            if "migration" not in changelog_lower and "upgrade" not in changelog_lower:
                suggestions.append("📚 Add migration guide for breaking changes")
        
        return suggestions
    
    def _analyze_release_workflow(self, tool_name: str, params: Dict[str, Any],
                                 context: Dict[str, Any]) -> List[str]:
        """Analyze release workflow and coordination needs."""
        suggestions = []
        
        # Pre-release checks
        if "create" in tool_name:
            suggestions.append("✅ Verify all CI/CD checks pass before release")
            suggestions.append("🧪 Run comprehensive test suite on release branch")
            suggestions.append("📊 Update version numbers in all relevant files")
        
        # Release coordination
        if self._detect_multi_repo_release(context):
            suggestions.append("🐝 Multi-repo release detected - use swarm coordination")
            suggestions.append("📋 Coordinate release timing across repositories")
        
        # Automation opportunities
        suggestions.append("🤖 Consider automating release with GitHub Actions")
        suggestions.append("📦 Auto-generate changelog from commit messages")
        
        # Post-release activities
        if params.get("published", True):
            suggestions.append("📢 Plan release announcement and communication")
            suggestions.append("📈 Monitor deployment metrics and user feedback")
        
        return suggestions
    
    def _analyze_deployment_strategy(self, params: Dict[str, Any]) -> List[str]:
        """Analyze deployment and distribution strategy."""
        suggestions = []
        
        # Draft vs published releases
        is_draft = params.get("draft", False)
        is_prerelease = params.get("prerelease", False)
        
        if is_draft:
            suggestions.append("📝 Draft release - perfect for review before publishing")
        
        if is_prerelease:
            suggestions.append("🧪 Prerelease - ensure clear communication to users")
            suggestions.append("⚠️ Add warnings about stability in release notes")
        
        # Asset management
        assets = params.get("assets", [])
        if not assets and not is_draft:
            suggestions.append("📦 Consider adding release assets (binaries, docs)")
        
        # Distribution channels
        suggestions.append("📡 Plan distribution: package managers, containers, etc.")
        
        # Security considerations
        if assets:
            suggestions.append("🔒 Generate checksums for release assets")
            suggestions.append("✍️ Consider signing releases for authenticity")
        
        return suggestions
    
    def _detect_multi_repo_release(self, context: Dict[str, Any]) -> bool:
        """Detect if this is part of a multi-repository release."""
        recent_tools = context.get("recent_tools", [])
        
        release_tool_count = sum(1 for tool in recent_tools 
                                if any(pattern in tool.lower() 
                                      for pattern in self.release_patterns))
        
        return release_tool_count > 2
    
    def get_pattern_suggestions(self) -> List[str]:
        """Get general release management pattern suggestions."""
        return [
            "📏 Follow semantic versioning (semver.org) for version numbers",
            "📝 Maintain comprehensive changelog with each release",
            "🏷️ Use consistent git tagging strategy",
            "✅ Implement pre-release validation checklist",
            "🤖 Automate release process with CI/CD",
            "📦 Provide release assets for different platforms",
            "🔒 Sign releases and provide checksums for security",
            "📢 Plan release communication and documentation updates",
            "🐝 Use swarm coordination for complex multi-repo releases"
        ]