"""Post-tool analysis system for Claude Code hook integration.

This modular system detects when Claude Code drifts away from the proper
Queen ZEN → Flow Workers → Storage Workers → Execution Drones workflow
and provides strategic guidance to restore hive coordination.
"""

from .manager import PostToolAnalysisManager, PostToolAnalysisConfig, DebugAnalysisReporter
from .core import (
    DriftSeverity,
    DriftType,
    DriftEvidence,
    NonBlockingGuidanceProvider,
    GuidanceOutputHandler
)

__version__ = "1.0.0"

__all__ = [
    'PostToolAnalysisManager',
    'PostToolAnalysisConfig', 
    'DebugAnalysisReporter',
    'DriftSeverity',
    'DriftType',
    'DriftEvidence',
    'NonBlockingGuidanceProvider',
    'GuidanceOutputHandler'
]