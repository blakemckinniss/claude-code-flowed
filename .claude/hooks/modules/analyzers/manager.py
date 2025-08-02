"""Manager for coordinating multiple analyzers."""

from typing import List, Dict, Type, Optional
from ..core import Analyzer, PatternMatch
from ..patterns import (
    HiveOrchestrationAnalyzer,
    MCPOrchestrationAnalyzer,
    DevelopmentAnalyzer,
    GitHubAnalyzer,
    TestingAnalyzer,
    PerformanceAnalyzer,
    SwarmAnalyzer,
    CommandPatternsAnalyzer
)


class AnalyzerManager:
    """Manages multiple analyzers and coordinates analysis."""
    
    # Registry of available analyzers
    ANALYZER_REGISTRY: Dict[str, Type[Analyzer]] = {
        "command_patterns": CommandPatternsAnalyzer,
        "hive_orchestration": HiveOrchestrationAnalyzer,
        "mcp_orchestration": MCPOrchestrationAnalyzer,
        "development": DevelopmentAnalyzer,
        "github": GitHubAnalyzer,
        "testing": TestingAnalyzer,
        "performance": PerformanceAnalyzer,
        "swarm": SwarmAnalyzer
    }
    
    def __init__(self, enabled_analyzers: Optional[List[str]] = None):
        self.analyzers: List[Analyzer] = []
        
        if enabled_analyzers is None:
            enabled_analyzers = list(self.ANALYZER_REGISTRY.keys())
        
        self._initialize_analyzers(enabled_analyzers)
    
    def _initialize_analyzers(self, enabled_names: List[str]) -> None:
        """Initialize enabled analyzers."""
        for name in enabled_names:
            if name in self.ANALYZER_REGISTRY:
                analyzer_class = self.ANALYZER_REGISTRY[name]
                analyzer = analyzer_class(priority=self._get_priority(name))
                self.analyzers.append(analyzer)
    
    def _get_priority(self, analyzer_name: str) -> int:
        """Get priority for an analyzer."""
        # Higher priority for more specific analyzers
        priorities = {
            "command_patterns": 2000,     # SUPREME - Direct actionable commands
            "hive_orchestration": 1500,   # Queen ZEN commands the hive
            "mcp_orchestration": 1000,    # MCP ZEN is master orchestrator  
            "swarm": 100,                 # Coordination is critical
            "performance": 80,
            "testing": 70,
            "github": 60,
            "development": 50             # Lowest - most general
        }
        return priorities.get(analyzer_name, 0)
    
    def analyze(self, prompt: str) -> List[PatternMatch]:
        """Analyze prompt with all enabled analyzers."""
        all_matches = []
        
        for analyzer in self.analyzers:
            matches = analyzer.analyze(prompt)
            all_matches.extend(matches)
        
        return all_matches
    
    def register_analyzer(self, name: str, analyzer_class: Type[Analyzer]) -> None:
        """Register a new analyzer type."""
        self.ANALYZER_REGISTRY[name] = analyzer_class
    
    def add_analyzer(self, analyzer: Analyzer) -> None:
        """Add an analyzer instance."""
        self.analyzers.append(analyzer)
    
    def get_analyzer_names(self) -> List[str]:
        """Get names of all registered analyzers."""
        return list(self.ANALYZER_REGISTRY.keys())
    
    def get_enabled_analyzer_names(self) -> List[str]:
        """Get names of enabled analyzers."""
        return [analyzer.get_name() for analyzer in self.analyzers]