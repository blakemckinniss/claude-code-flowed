"""Post-tool analysis manager - coordinates all drift detection analyzers."""

import json
import os
import sys
from typing import List, Dict, Any, Optional, Type

# Path setup handled by centralized resolver when importing this module
from .core import (
    DriftAnalyzer, 
    DriftEvidence,
    NonBlockingGuidanceProvider,
    GuidanceOutputHandler, 
    ContextualGuidanceEnhancer
)
from .analyzers import (
    ZenBypassAnalyzer,
    FlowCoordinationAnalyzer,
    NativeToolOveruseAnalyzer,
    WorkflowPatternAnalyzer,
    BatchingOpportunityAnalyzer,
    MemoryCoordinationAnalyzer
)


class PostToolAnalysisConfig:
    """Configuration for post-tool analysis system."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default config path relative to hooks directory
            hooks_dir = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(hooks_dir, "post_tool_config.json")
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load post-tool config: {e}")
        
        # Return default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "enabled_analyzers": [
                "zen_bypass_analyzer",
                "flow_coordination_analyzer", 
                "native_overuse_analyzer",
                "workflow_pattern_analyzer",
                "batching_opportunity_analyzer",
                "memory_coordination_analyzer"
            ],
            "guidance_settings": {
                "max_guidance_frequency": 5,  # Max once every 5 tools
                "escalation_threshold": 3,    # Escalate after 3 ignored guidances
                "emergency_threshold": 5      # Emergency mode after 5 ignored
            },
            "drift_sensitivity": {
                "zen_bypass": "normal",       # normal, strict, lenient
                "native_overuse": "normal",
                "workflow_fragmentation": "lenient"
            },
            "blocking_thresholds": {
                "critical_only": False,       # Only block on critical severity
                "emergency_escalation": True  # Block on emergency escalation
            }
        }
    
    def is_analyzer_enabled(self, analyzer_name: str) -> bool:
        """Check if an analyzer is enabled."""
        return analyzer_name in self._config.get("enabled_analyzers", [])
    
    def get_guidance_settings(self) -> Dict[str, Any]:
        """Get guidance settings."""
        return self._config.get("guidance_settings", {})
    
    def get_drift_sensitivity(self, drift_type: str) -> str:
        """Get sensitivity setting for a drift type."""
        return self._config.get("drift_sensitivity", {}).get(drift_type, "normal")
    
    def get_blocking_settings(self) -> Dict[str, Any]:
        """Get blocking behavior settings."""
        return self._config.get("blocking_thresholds", {})


class PostToolAnalysisManager:
    """Coordinates all post-tool drift detection and guidance."""
    
    # Registry of available analyzers
    ANALYZER_REGISTRY: Dict[str, Type[DriftAnalyzer]] = {
        "zen_bypass_analyzer": ZenBypassAnalyzer,
        "flow_coordination_analyzer": FlowCoordinationAnalyzer,
        "native_overuse_analyzer": NativeToolOveruseAnalyzer,
        "workflow_pattern_analyzer": WorkflowPatternAnalyzer,
        "batching_opportunity_analyzer": BatchingOpportunityAnalyzer,
        "memory_coordination_analyzer": MemoryCoordinationAnalyzer
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = PostToolAnalysisConfig(config_path)
        self.analyzers: List[DriftAnalyzer] = []
        self.guidance_provider = NonBlockingGuidanceProvider()
        self.guidance_enhancer = ContextualGuidanceEnhancer()
        self.tool_count = 0
        
        self._initialize_analyzers()
    
    def _initialize_analyzers(self) -> None:
        """Initialize enabled analyzers."""
        for analyzer_name, analyzer_class in self.ANALYZER_REGISTRY.items():
            if self.config.is_analyzer_enabled(analyzer_name):
                priority = self._get_analyzer_priority(analyzer_name)
                analyzer = analyzer_class(priority=priority)
                self.analyzers.append(analyzer)
        
        # Sort by priority (highest first)
        self.analyzers.sort(key=lambda a: a.priority, reverse=True)
    
    def _get_analyzer_priority(self, analyzer_name: str) -> int:
        """Get priority for an analyzer."""
        priorities = {
            "zen_bypass_analyzer": 1000,          # Highest - Queen ZEN is supreme
            "flow_coordination_analyzer": 800,    # High - Flow coordination critical
            "workflow_pattern_analyzer": 700,     # High - Pattern optimization important
            "memory_coordination_analyzer": 600,  # Medium-High - Memory coordination
            "native_overuse_analyzer": 600,       # Medium-High - Native tool balance
            "batching_opportunity_analyzer": 500  # Medium - Efficiency optimization
        }
        return priorities.get(analyzer_name, 500)
    
    def analyze_tool_usage(self, tool_name: str, tool_input: Dict[str, Any], 
                          tool_response: Dict[str, Any]) -> None:
        """Analyze a tool usage for drift and provide guidance if needed."""
        self.tool_count += 1
        
        # Collect drift evidence from all analyzers
        all_evidence: List[DriftEvidence] = []
        
        for analyzer in self.analyzers:
            try:
                evidence = analyzer.analyze_drift(tool_name, tool_input, tool_response)
                if evidence:
                    all_evidence.append(evidence)
            except Exception as e:
                print(f"Warning: Analyzer {analyzer.get_analyzer_name()} failed: {e}")
        
        # Provide guidance if warranted
        guidance_info = self.guidance_provider.provide_guidance(all_evidence, self.tool_count)
        
        if guidance_info:
            # Enhance guidance with context
            enhanced_message = self.guidance_enhancer.enhance_guidance(
                guidance_info["message"],
                tool_name,
                tool_input,
                all_evidence[0].tool_sequence if all_evidence else []
            )
            guidance_info["message"] = enhanced_message
            
            # Handle guidance output and exit
            GuidanceOutputHandler.handle_guidance_output(guidance_info)
        else:
            # No guidance needed
            GuidanceOutputHandler.handle_no_guidance()
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get status of all analyzers for debugging."""
        return {
            "total_tools_processed": self.tool_count,
            "active_analyzers": [a.get_analyzer_name() for a in self.analyzers],
            "analyzer_priorities": {a.get_analyzer_name(): a.priority for a in self.analyzers},
            "config_path": self.config.config_path
        }


class DebugAnalysisReporter:
    """Provides detailed analysis reports for debugging."""
    
    def __init__(self, manager: PostToolAnalysisManager):
        self.manager = manager
    
    def generate_debug_report(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Generate detailed debug report for troubleshooting."""
        report_parts = [
            "=== POST-TOOL ANALYSIS DEBUG REPORT ===",
            f"Tool: {tool_name}",
            f"Total Tools Processed: {self.manager.tool_count}",
            f"Active Analyzers: {len(self.manager.analyzers)}",
            ""
        ]
        
        # Analyzer details
        for analyzer in self.manager.analyzers:
            report_parts.extend([
                f"Analyzer: {analyzer.get_analyzer_name()}",
                f"  Priority: {analyzer.priority}",
                f"  MCP Ratio: {analyzer.get_mcp_ratio():.2f}",
                f"  Recent Tools: {' → '.join(analyzer.get_recent_tools(3))}",
                f"  Has ZEN Coordination: {analyzer.has_zen_coordination()}",
                f"  Has Flow Coordination: {analyzer.has_flow_coordination()}",
                ""
            ])
        
        return "\n".join(report_parts)
    
    def log_debug_info(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """Log debug information if debugging is enabled."""
        debug_env = os.environ.get("CLAUDE_HOOKS_DEBUG", "").lower()
        if debug_env in ["true", "1", "on"]:
            report = self.generate_debug_report(tool_name, tool_input)
            print(f"DEBUG: {report}", file=sys.stderr)