"""GitHub patterns analyzer."""

from ..core import Analyzer


class GitHubAnalyzer(Analyzer):
    """Analyzes GitHub-related patterns."""
    
    def get_name(self) -> str:
        return "github"
    
    def _initialize_patterns(self) -> None:
        """Initialize GitHub patterns."""
        self.add_pattern(
            r"(github|pr|pull request|merge|branch)",
            """
🔗 GITHUB PR WORKFLOW DETECTED
• Use pr-manager for pull request management
• Deploy code-review-swarm for automated review
• Consider github-modes for batch operations
• Enable workflow-automation for CI/CD""",
            {
                "category": "github-pr",
                "suggested_agents": ["pr-manager", "code-review-swarm", "github-modes"],
                "workflow": "pull-request"
            }
        )
        
        self.add_pattern(
            r"(issue|bug|ticket|task|story)",
            """
🐛 ISSUE MANAGEMENT DETECTED
• Deploy issue-tracker for issue triage
• Use swarm-issue for issue-to-task conversion
• Consider project-board-sync for visualization
• Enable github-modes for batch operations""",
            {
                "category": "github-issue",
                "suggested_agents": ["issue-tracker", "swarm-issue", "project-board-sync"],
                "workflow": "issue-management"
            }
        )
        
        self.add_pattern(
            r"(release|deploy|version|tag|publish)",
            """
🚀 RELEASE MANAGEMENT DETECTED
• Use release-manager for coordination
• Deploy release-swarm for automation
• Consider workflow-automation for CI/CD
• Enable multi-repo-swarm for dependencies""",
            {
                "category": "github-release",
                "suggested_agents": ["release-manager", "release-swarm", "workflow-automation"],
                "workflow": "release"
            }
        )
        
        self.add_pattern(
            r"(repository|repo|clone|fork|sync)",
            """
📦 REPOSITORY MANAGEMENT DETECTED
• Use repo-architect for structure optimization
• Deploy sync-coordinator for multi-repo sync
• Consider multi-repo-swarm for organization-wide tasks
• Enable github-modes for comprehensive integration""",
            {
                "category": "github-repo",
                "suggested_agents": ["repo-architect", "sync-coordinator", "multi-repo-swarm"],
                "workflow": "repository"
            }
        )