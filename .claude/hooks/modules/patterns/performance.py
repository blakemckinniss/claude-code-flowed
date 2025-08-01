"""Performance patterns analyzer."""

from ..core import Analyzer


class PerformanceAnalyzer(Analyzer):
    """Analyzes performance-related patterns."""
    
    def get_name(self) -> str:
        return "performance"
    
    def _initialize_patterns(self) -> None:
        """Initialize performance patterns."""
        self.add_pattern(
            r"(performance|optimize|bottleneck|slow|speed|fast|efficient)",
            """
⚡ PERFORMANCE OPTIMIZATION DETECTED
• Use performance-benchmarker for baseline metrics
• Deploy perf-analyzer for bottleneck identification
• Consider adaptive-coordinator for optimization
• Enable parallel execution strategies""",
            {
                "category": "performance",
                "suggested_agents": ["performance-benchmarker", "perf-analyzer", "adaptive-coordinator"],
                "focus": "optimization"
            }
        )
        
        self.add_pattern(
            r"(memory|leak|usage|consumption|resource)",
            """
💾 MEMORY OPTIMIZATION DETECTED
• Use memory-coordinator for management
• Deploy swarm-memory-manager for distribution
• Consider perf-analyzer for leak detection
• Monitor with performance-benchmarker""",
            {
                "category": "memory",
                "suggested_agents": ["memory-coordinator", "swarm-memory-manager", "perf-analyzer"],
                "focus": "memory-optimization"
            }
        )
        
        self.add_pattern(
            r"(parallel|concurrent|async|thread|process)",
            """
🔄 CONCURRENCY OPTIMIZATION DETECTED
• Deploy mesh-coordinator for parallel execution
• Use task-orchestrator for workflow optimization
• Consider adaptive-coordinator for dynamic scaling
• Enable batchtools for 300% gains""",
            {
                "category": "concurrency",
                "suggested_agents": ["mesh-coordinator", "task-orchestrator", "adaptive-coordinator"],
                "focus": "parallelization"
            }
        )
        
        self.add_pattern(
            r"(scale|scalability|load|stress|benchmark)",
            """
📈 SCALABILITY ANALYSIS DETECTED
• Use performance-benchmarker for load testing
• Deploy hierarchical-coordinator for scaling
• Consider byzantine-coordinator for resilience
• Enable distributed consensus protocols""",
            {
                "category": "scalability",
                "suggested_agents": ["performance-benchmarker", "hierarchical-coordinator", "byzantine-coordinator"],
                "focus": "scalability"
            }
        )