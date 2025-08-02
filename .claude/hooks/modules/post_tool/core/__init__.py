"""Core modules for the Universal Tool Feedback System.

This package provides the foundational components for the modular analyzer architecture:

- tool_analyzer_base: Protocol interfaces and base classes for analyzers
- analyzer_registry: Dynamic analyzer registration and coordination system
- hook_integration: Seamless PostToolUse hook integration layer
- performance_optimizer: High-performance caching and async execution
- system_integration: Complete system coordinator and entry point

The system is designed for:
- Sub-100ms stderr feedback generation
- Non-blocking async-first execution
- Intelligent caching with LRU eviction
- Priority-based analyzer execution
- Backward compatibility with existing hooks
"""

from .tool_analyzer_base import (
    ToolAnalyzer,
    BaseToolAnalyzer,
    ToolContext,
    FeedbackResult,
    FeedbackSeverity,
    ToolCategory,
    AnalyzerMetrics,
    AsyncAnalyzerPool,
    AnalyzerConfiguration
)

from .analyzer_registry import (
    AnalyzerRegistry,
    get_global_registry
)

from .hook_integration import (
    PostToolHookIntegrator,
    BackwardCompatibilityLayer,
    analyze_tool_for_hook_sync
)

from .performance_optimizer import (
    PerformanceOptimizer,
    HighPerformanceCache,
    AsyncExecutionPool,
    ResourceMonitor,
    get_global_optimizer
)

from .system_integration import (
    UniversalToolFeedbackSystem,
    get_global_system,
    initialize_global_system,
    shutdown_global_system,
    analyze_tool_with_universal_system,
    analyze_tool_with_universal_system_sync,
    LegacyCompatibilityWrapper,
    get_system_diagnostics,
    update_system_config,
    get_system_config,
    run_performance_test
)

from .drift_detector import (
    DriftAnalyzer,
    DriftEvidence,
    DriftSeverity,
    DriftType,
    HiveWorkflowValidator,
    DriftGuidanceGenerator
)

from .guidance_system import (
    NonBlockingGuidanceProvider,
    GuidanceOutputHandler,
    ContextualGuidanceEnhancer
)

__all__ = [
    # Core interfaces and base classes
    "ToolAnalyzer",
    "BaseToolAnalyzer", 
    "ToolContext",
    "FeedbackResult",
    "FeedbackSeverity",
    "ToolCategory",
    "AnalyzerMetrics",
    "AsyncAnalyzerPool",
    "AnalyzerConfiguration",
    
    # Registry system
    "AnalyzerRegistry",
    "get_global_registry",
    
    # Hook integration
    "PostToolHookIntegrator",
    "BackwardCompatibilityLayer",
    "analyze_tool_for_hook_sync",
    
    # Performance optimization
    "PerformanceOptimizer",
    "HighPerformanceCache", 
    "AsyncExecutionPool",
    "ResourceMonitor",
    "get_global_optimizer",
    
    # System integration
    "UniversalToolFeedbackSystem",
    "get_global_system",
    "initialize_global_system",
    "shutdown_global_system",
    "analyze_tool_with_universal_system",
    "analyze_tool_with_universal_system_sync",
    "LegacyCompatibilityWrapper",
    "get_system_diagnostics",
    "update_system_config",
    "get_system_config",
    "run_performance_test",
    
    # Drift detection
    "DriftAnalyzer",
    "DriftEvidence", 
    "DriftSeverity",
    "DriftType",
    "HiveWorkflowValidator",
    "DriftGuidanceGenerator",
    
    # Guidance system
    "NonBlockingGuidanceProvider",
    "GuidanceOutputHandler",
    "ContextualGuidanceEnhancer"
]

# System version and metadata
__version__ = "1.0.0"
__author__ = "Claude Code Intelligence System"
__description__ = "Universal Tool Feedback System - Modular stderr exit(2) architecture"