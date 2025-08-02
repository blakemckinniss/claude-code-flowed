"""Post-tool analysis system for Claude Code hook integration.

This modular system provides two complementary approaches:

1. Legacy drift detection system - Detects when Claude Code drifts away from the proper
   Queen ZEN → Flow Workers → Storage Workers → Execution Drones workflow
   
2. New analyzer dispatch system - Extensible framework for tool-specific analyzers
   with async execution support and intelligent pattern matching
"""

# New analyzer dispatch system
from .analyzer_dispatch import (
    AnalyzerDispatcher,
    ToolAnalyzer,
    AnalyzerResult,
    AnalysisResult,
    ToolMatcher,
    AnalyzerRegistry,
)

# Legacy drift detection system
from .manager import PostToolAnalysisManager, PostToolAnalysisConfig, DebugAnalysisReporter
from .core import (
    DriftSeverity,
    DriftType,
    DriftEvidence,
    NonBlockingGuidanceProvider,
    GuidanceOutputHandler
)

__version__ = "2.0.0"

__all__ = [
    # New analyzer dispatch system
    'AnalyzerDispatcher',
    'ToolAnalyzer', 
    'AnalyzerResult',
    'AnalysisResult',
    'ToolMatcher',
    'AnalyzerRegistry',
    
    # Legacy drift detection system
    'DebugAnalysisReporter',
    'DriftEvidence',
    'DriftSeverity',
    'DriftType',
    'GuidanceOutputHandler',
    'NonBlockingGuidanceProvider',
    'PostToolAnalysisConfig',
    'PostToolAnalysisManager'
]