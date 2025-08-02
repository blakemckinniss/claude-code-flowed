"""Specialized analyzers for tool-specific feedback generation.

This package contains specialized analyzers that implement the ToolAnalyzer protocol
for specific categories of tools and operations:

- file_operations_analyzer: File system operations (Write, Edit, MultiEdit, etc.)
- mcp_coordination_analyzer: MCP tool hierarchy and workflow coordination
- execution_safety_analyzer: Command execution security and safety validation

Each analyzer is designed for:
- High-performance analysis (<100ms target)
- Async-first execution with sync compatibility
- Intelligent caching and memoization
- Priority-based feedback generation
- Comprehensive error handling and fallbacks
"""

from .file_operations_analyzer import FileOperationsAnalyzer
from .mcp_coordination_analyzer import MCPCoordinationAnalyzer, MCPParameterValidator
from .execution_safety_analyzer import ExecutionSafetyAnalyzer, PackageManagerAnalyzer

__all__ = [
    "FileOperationsAnalyzer",
    "MCPCoordinationAnalyzer", 
    "MCPParameterValidator",
    "ExecutionSafetyAnalyzer",
    "PackageManagerAnalyzer"
]

# Auto-registration support
AVAILABLE_ANALYZERS = [
    FileOperationsAnalyzer,
    MCPCoordinationAnalyzer,
    MCPParameterValidator, 
    ExecutionSafetyAnalyzer,
    PackageManagerAnalyzer
]

def get_available_analyzers():
    """Get list of all available specialized analyzers."""
    return AVAILABLE_ANALYZERS.copy()

def create_default_analyzers():
    """Create instances of all analyzers with default configurations."""
    return [
        FileOperationsAnalyzer(priority=800),
        MCPCoordinationAnalyzer(priority=900),
        MCPParameterValidator(priority=700),
        ExecutionSafetyAnalyzer(priority=950),
        PackageManagerAnalyzer(priority=750)
    ]