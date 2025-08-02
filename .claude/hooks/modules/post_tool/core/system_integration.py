"""System Integration Layer for Universal Tool Feedback System.

This module provides the complete integration between the PostToolUse hook
and the new modular analyzer architecture, ensuring seamless operation
with the existing Claude Hook → ZEN → Claude Flow ecosystem.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .tool_analyzer_base import ToolContext, FeedbackResult, FeedbackSeverity
from .analyzer_registry import get_global_registry, AnalyzerRegistry
from .hook_integration import PostToolHookIntegrator
from .performance_optimizer import get_global_optimizer, PerformanceOptimizer

# Import specialized analyzers for auto-registration
from ..analyzers.specialized.file_operations_analyzer import FileOperationsAnalyzer
from ..analyzers.specialized.mcp_coordination_analyzer import (
    MCPCoordinationAnalyzer, MCPParameterValidator
)
from ..analyzers.specialized.execution_safety_analyzer import (
    ExecutionSafetyAnalyzer, PackageManagerAnalyzer
)


class UniversalToolFeedbackSystem:
    """Complete universal tool feedback system coordinator."""
    
    def __init__(self):
        """Initialize the universal feedback system."""
        self.registry = get_global_registry()
        self.optimizer = get_global_optimizer()
        self.integrator = PostToolHookIntegrator(self.registry)
        self.initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # System configuration
        self.config = {
            "performance_target_ms": 100,
            "enable_caching": True,
            "enable_parallel_execution": True,
            "max_concurrent_analyzers": 4,
            "cache_ttl_seconds": 300,
            "enable_performance_monitoring": True
        }
    
    async def initialize(self) -> None:
        """Initialize the feedback system with all analyzers."""
        if self.initialized:
            return
        
        async with self._initialization_lock:
            if self.initialized:
                return
            
            try:
                # Register all built-in analyzers
                await self._register_builtin_analyzers()
                
                # Initialize performance monitoring
                if self.config["enable_performance_monitoring"]:
                    self._setup_performance_monitoring()
                
                self.initialized = True
                print("🚀 Universal Tool Feedback System initialized", file=sys.stderr)
                
            except Exception as e:
                print(f"Error initializing feedback system: {e}", file=sys.stderr)
                raise
    
    async def _register_builtin_analyzers(self) -> None:
        """Register all built-in analyzers."""
        # File operations analyzer (high priority)
        file_analyzer = FileOperationsAnalyzer(priority=800)
        self.registry.register_analyzer(file_analyzer, replace_existing=True)
        
        # MCP coordination analyzer (highest priority)
        mcp_analyzer = MCPCoordinationAnalyzer(priority=900)
        self.registry.register_analyzer(mcp_analyzer, replace_existing=True)
        
        # MCP parameter validator
        param_validator = MCPParameterValidator(priority=700)
        self.registry.register_analyzer(param_validator, replace_existing=True)
        
        # Execution safety analyzer (very high priority)
        exec_analyzer = ExecutionSafetyAnalyzer(priority=950)
        self.registry.register_analyzer(exec_analyzer, replace_existing=True)
        
        # Package manager analyzer
        pkg_analyzer = PackageManagerAnalyzer(priority=750)
        self.registry.register_analyzer(pkg_analyzer, replace_existing=True)
        
        print(f"✅ Registered {len(self.registry._analyzers)} built-in analyzers", file=sys.stderr)
    
    def _setup_performance_monitoring(self) -> None:
        """Setup performance monitoring and optimization."""
        from .performance_optimizer import ResourceMonitor
        
        # Start resource monitoring
        monitor = ResourceMonitor(self.optimizer)
        monitor.start_monitoring(interval=60.0)  # Check every minute
    
    async def process_tool_usage(
        self, 
        tool_name: str, 
        tool_input: Dict[str, Any], 
        tool_response: Dict[str, Any],
        session_context: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """Process tool usage through the complete feedback system.
        
        Args:
            tool_name: Name of the tool used
            tool_input: Input parameters passed to tool
            tool_response: Response received from tool
            session_context: Optional session context information
            
        Returns:
            Exit code (0=success, 1=error, 2=guidance) or None for no action
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
        
        # Use the hook integrator for processing
        return await self.integrator.process_tool_usage(
            tool_name, tool_input, tool_response, session_context
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and performance metrics."""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "initialized": self.initialized,
            "config": self.config,
            "registry_info": self.registry.get_registry_info(),
            "performance_report": self.optimizer.get_performance_report(),
            "integration_stats": self.integrator.get_integration_stats()
        }
    
    def shutdown(self) -> None:
        """Shutdown the feedback system and cleanup resources."""
        if self.optimizer:
            self.optimizer.shutdown()
        self.initialized = False


# Global system instance
_global_system: Optional[UniversalToolFeedbackSystem] = None


def get_global_system() -> UniversalToolFeedbackSystem:
    """Get or create the global feedback system."""
    global _global_system
    
    if _global_system is None:
        _global_system = UniversalToolFeedbackSystem()
    
    return _global_system


async def initialize_global_system() -> None:
    """Initialize the global feedback system."""
    system = get_global_system()
    await system.initialize()


def shutdown_global_system() -> None:
    """Shutdown the global feedback system."""
    global _global_system
    
    if _global_system is not None:
        _global_system.shutdown()
        _global_system = None


# Main integration function for PostToolUse hook
async def analyze_tool_with_universal_system(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_response: Dict[str, Any],
    session_context: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """Main entry point for PostToolUse hook integration.
    
    This function should be called from the PostToolUse hook to get
    universal tool analysis with the complete modular architecture.
    
    Args:
        tool_name: Name of the tool used
        tool_input: Input parameters passed to tool
        tool_response: Response received from tool
        session_context: Optional session context information
        
    Returns:
        Exit code (0=success, 1=error, 2=guidance) or None for no action
    """
    try:
        system = get_global_system()
        return await system.process_tool_usage(
            tool_name, tool_input, tool_response, session_context
        )
    except Exception as e:
        print(f"Warning: Universal system error: {e}", file=sys.stderr)
        return None  # Fall back to existing hook behavior


def analyze_tool_with_universal_system_sync(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_response: Dict[str, Any],
    session_context: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """Synchronous wrapper for PostToolUse hook integration.
    
    This function can be directly called from the existing PostToolUse hook
    without requiring async/await syntax changes in the main hook file.
    """
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async analysis
        return loop.run_until_complete(
            analyze_tool_with_universal_system(
                tool_name, tool_input, tool_response, session_context
            )
        )
    
    except Exception as e:
        print(f"Warning: Universal system sync error: {e}", file=sys.stderr)
        return None  # Fall back to existing behavior


class LegacyCompatibilityWrapper:
    """Wrapper to maintain compatibility with existing PostToolUse patterns."""
    
    @staticmethod
    def check_ruff_integration(tool_name: str, tool_input: Dict[str, Any]) -> Optional[int]:
        """Legacy Ruff integration check - now handled by FileOperationsAnalyzer."""
        # This is now handled by the FileOperationsAnalyzer
        # Return None to indicate the new system should handle it
        return None
    
    @staticmethod
    def check_hook_violations(tool_name: str, tool_input: Dict[str, Any]) -> Optional[int]:
        """Legacy hook violation check - now handled by FileOperationsAnalyzer."""
        # This is now handled by the FileOperationsAnalyzer  
        # Return None to indicate the new system should handle it
        return None
    
    @staticmethod
    def check_workflow_patterns(tool_name: str, tool_input: Dict[str, Any]) -> Optional[int]:
        """Legacy workflow pattern check - now handled by MCPCoordinationAnalyzer."""
        # This is now handled by the MCPCoordinationAnalyzer
        # Return None to indicate the new system should handle it
        return None


def get_system_diagnostics() -> Dict[str, Any]:
    """Get comprehensive system diagnostics for debugging."""
    try:
        system = get_global_system()
        status = system.get_system_status()
        
        return {
            "system_status": status,
            "environment": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "hook_path": __file__
            },
            "performance_summary": {
                "target_ms": system.config["performance_target_ms"],
                "caching_enabled": system.config["enable_caching"],
                "parallel_execution": system.config["enable_parallel_execution"]
            }
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "system_status": "error",
            "fallback_available": True
        }


# Configuration management
def update_system_config(config_updates: Dict[str, Any]) -> None:
    """Update system configuration."""
    system = get_global_system()
    system.config.update(config_updates)
    
    # Apply configuration changes
    if "max_concurrent_analyzers" in config_updates:
        system.optimizer.execution_pool.max_workers = config_updates["max_concurrent_analyzers"]
    
    if "cache_ttl_seconds" in config_updates:
        system.optimizer.cache.default_ttl = config_updates["cache_ttl_seconds"]


def get_system_config() -> Dict[str, Any]:
    """Get current system configuration."""
    system = get_global_system()
    return system.config.copy()


# Development and testing utilities
def run_performance_test(tool_name: str = "Read", iterations: int = 100) -> Dict[str, Any]:
    """Run performance test on the universal system."""
    import random
    
    async def test_runner():
        system = get_global_system()
        await system.initialize()
        
        results = []
        
        for i in range(iterations):
            # Create test context
            context = ToolContext(
                tool_name=tool_name,
                tool_input={"file_path": f"test_file_{i}.py"},
                tool_response={"success": True},
                execution_time=random.uniform(0.01, 0.1)
            )
            
            start_time = time.time()
            
            # Analyze with system
            analyzers = system.registry.get_analyzers_for_tool(tool_name)
            await system.optimizer.execute_with_optimization(analyzers, context)
            
            duration = time.time() - start_time
            results.append(duration)
        
        return results
    
    # Run test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        durations = loop.run_until_complete(test_runner())
        
        return {
            "iterations": iterations,
            "average_duration_ms": (sum(durations) / len(durations)) * 1000,
            "min_duration_ms": min(durations) * 1000,
            "max_duration_ms": max(durations) * 1000,
            "target_met": (sum(durations) / len(durations)) * 1000 < 100,
            "success_rate": 100.0  # All successful in test
        }
    finally:
        loop.close()