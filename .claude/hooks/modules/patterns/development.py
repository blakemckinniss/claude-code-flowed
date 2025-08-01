"""Development patterns analyzer."""

from ..core import Analyzer


class DevelopmentAnalyzer(Analyzer):
    """Analyzes development-related patterns."""
    
    def get_name(self) -> str:
        return "development"
    
    def _initialize_patterns(self) -> None:
        """Initialize development patterns."""
        self.add_pattern(
            r"(build|create|implement|develop|code|feature|function|class|module|component)",
            """
🚀 ZEN-ORCHESTRATED DEVELOPMENT WORKFLOW
• 🧠 START: mcp__zen__planner for systematic SPARC approach
• 📁 PREPARE: mcp__filesystem__create_directory + mcp__filesystem__list_directory  
• 🤖 COORDINATE: mcp__claude-flow__swarm_init with ZEN's topology
• 🔧 EXECUTE: Claude Code with ZEN's orchestrated plan
• ✅ ZEN → Filesystem → Flow → Code (Never skip ZEN!)""",
            {
                "category": "development", 
                "zen_orchestrated": True,
                "methodology": "ZEN_SPARC",
                "workflow": "zen_first"
            }
        )
        
        self.add_pattern(
            r"(api|endpoint|rest|graphql|backend|server)",
            """
🌐 API DEVELOPMENT DETECTED
• Use backend-dev agent for API implementation
• Consider api-docs agent for OpenAPI documentation
• Deploy security-manager for authentication
• Use performance-benchmarker for API testing""",
            {
                "category": "api",
                "suggested_agents": ["backend-dev", "api-docs", "security-manager"],
                "focus": "API development"
            }
        )
        
        self.add_pattern(
            r"(react|vue|angular|frontend|ui|ux|mobile|native)",
            """
📱 FRONTEND/MOBILE DEVELOPMENT DETECTED
• Deploy mobile-dev agent for React Native
• Use base-template-generator for boilerplate
• Include production-validator for UI testing
• Consider performance-benchmarker for optimization""",
            {
                "category": "frontend",
                "suggested_agents": ["mobile-dev", "base-template-generator", "production-validator"],
                "focus": "Frontend/Mobile"
            }
        )
        
        self.add_pattern(
            r"(database|sql|nosql|schema|migration|model)",
            """
🗄️ DATABASE DESIGN DETECTED
• Use system-architect for schema design
• Deploy migration-planner for safe migrations
• Consider performance-benchmarker for queries
• Include tester for data integrity tests""",
            {
                "category": "database",
                "suggested_agents": ["system-architect", "migration-planner", "tester"],
                "focus": "Database design"
            }
        )