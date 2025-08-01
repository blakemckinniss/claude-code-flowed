"""Testing patterns analyzer."""

from ..core import Analyzer


class TestingAnalyzer(Analyzer):
    """Analyzes testing-related patterns."""
    
    def get_name(self) -> str:
        return "testing"
    
    def _initialize_patterns(self) -> None:
        """Initialize testing patterns."""
        self.add_pattern(
            r"(test|tdd|testing|quality|validation|qa)",
            """
🧪 TESTING FOCUS DETECTED
• Deploy TDD swarm: tdd-london-swarm, tester, production-validator
• Use SPARC TDD workflow for systematic testing
• Generate tests in parallel for efficiency
• Consider code-analyzer for coverage analysis""",
            {
                "category": "testing",
                "suggested_agents": ["tdd-london-swarm", "tester", "production-validator", "code-analyzer"],
                "methodology": "TDD"
            }
        )
        
        self.add_pattern(
            r"(unit test|mock|stub|spy|fixture)",
            """
🎯 UNIT TESTING DETECTED
• Use tdd-london-swarm for mock-driven development
• Deploy tester for comprehensive test generation
• Consider sparc-coder for TDD implementation
• Enable parallel test generation for speed""",
            {
                "category": "unit-testing",
                "suggested_agents": ["tdd-london-swarm", "tester", "sparc-coder"],
                "focus": "unit-tests"
            }
        )
        
        self.add_pattern(
            r"(integration|e2e|end.to.end|acceptance)",
            """
🔗 INTEGRATION TESTING DETECTED
• Deploy production-validator for real scenarios
• Use tester for comprehensive coverage
• Consider performance-benchmarker for load tests
• Enable swarm coordination for complex flows""",
            {
                "category": "integration-testing",
                "suggested_agents": ["production-validator", "tester", "performance-benchmarker"],
                "focus": "integration"
            }
        )
        
        self.add_pattern(
            r"(coverage|quality|lint|format|static analysis)",
            """
📊 CODE QUALITY DETECTED
• Use code-analyzer for deep analysis
• Deploy reviewer for quality assurance
• Consider code-review-swarm for automation
• Enable metrics tracking for improvement""",
            {
                "category": "code-quality",
                "suggested_agents": ["code-analyzer", "reviewer", "code-review-swarm"],
                "focus": "quality"
            }
        )