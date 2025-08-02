"""PostToolUse Hook Integration Layer.

This module provides seamless integration between the PostToolUse hook
and the new universal tool analyzer system, maintaining backward 
compatibility while enabling the new modular architecture.

Key Features:
- Non-breaking integration with existing PostToolUse hook
- Maintains existing zen_consultant.py patterns
- Extends functionality without disrupting current workflows
- Performance-optimized with <100ms target execution time
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from .tool_analyzer_base import ToolContext, FeedbackResult, FeedbackSeverity
from .analyzer_registry import get_global_registry, AnalyzerRegistry
from ..core.drift_detector import DriftEvidence


class PostToolHookIntegrator:
    """Integration layer between PostToolUse hook and analyzer system."""
    
    def __init__(self, registry: Optional[AnalyzerRegistry] = None):
        """Initialize hook integrator.
        
        Args:
            registry: Optional analyzer registry (uses global if None)
        """
        self.registry = registry or get_global_registry()
        self.integration_stats = {
            "total_integrations": 0,
            "successful_integrations": 0,
            "fallback_count": 0,
            "average_execution_time": 0.0
        }
    
    async def process_tool_usage(
        self, 
        tool_name: str, 
        tool_input: Dict[str, Any], 
        tool_response: Dict[str, Any],
        session_context: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """Process tool usage through the analyzer system.
        
        Args:
            tool_name: Name of the tool used
            tool_input: Input parameters passed to tool
            tool_response: Response received from tool
            session_context: Optional session context information
            
        Returns:
            Exit code (0=success, 1=error, 2=guidance) or None for no action
        """
        start_time = time.time()
        
        try:
            # Create tool context
            context = self._create_tool_context(
                tool_name, tool_input, tool_response, session_context
            )
            
            # Analyze through registry
            results = await self.registry.analyze_tool(context)
            
            # Process results and determine exit code
            exit_code = self._process_analysis_results(results, context)
            
            # Update statistics
            execution_time = time.time() - start_time
            self._update_integration_stats(execution_time, exit_code is not None)
            
            return exit_code
        
        except Exception as e:
            print(f"Warning: Hook integration error: {e}", file=sys.stderr)
            self.integration_stats["fallback_count"] += 1
            return None  # Fall back to existing hook behavior
    
    def _create_tool_context(
        self, 
        tool_name: str, 
        tool_input: Dict[str, Any], 
        tool_response: Dict[str, Any],
        session_context: Optional[Dict[str, Any]]
    ) -> ToolContext:
        """Create tool context from hook input data."""
        # Extract workflow history from session context if available
        workflow_history = []
        if session_context:
            workflow_history = session_context.get("recent_tools", [])
        
        # Calculate execution time if available
        execution_time = 0.0
        if "start_time" in tool_response:
            execution_time = time.time() - tool_response["start_time"]
        elif "duration" in tool_response:
            execution_time = tool_response["duration"]
        
        return ToolContext(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
            execution_time=execution_time,
            success=tool_response.get("success", True),
            session_context=session_context,
            workflow_history=workflow_history
        )
    
    def _process_analysis_results(
        self, 
        results: List[FeedbackResult], 
        context: ToolContext
    ) -> Optional[int]:
        """Process analysis results and determine appropriate exit code."""
        if not results:
            return None  # No feedback needed
        
        # Get highest priority result
        priority_result = self.registry.get_highest_priority_result(results)
        if not priority_result:
            return None
        
        # Handle based on severity
        if priority_result.severity == FeedbackSeverity.INFO:
            # Informational only - no action needed
            return None
        
        elif priority_result.severity == FeedbackSeverity.WARNING:
            # Output guidance to stderr and exit with code 2
            self._output_guidance_message(priority_result, context)
            return 2
        
        else:  # ERROR or CRITICAL
            # Output error message to stderr and exit with code 1
            self._output_error_message(priority_result, context)
            return 1
    
    def _output_guidance_message(self, result: FeedbackResult, context: ToolContext):
        """Output guidance message to stderr."""
        lines = [
            f"\n{'='*70}",
            f"🎯 TOOL USAGE GUIDANCE - {result.analyzer_name.upper()}",
            f"Tool: {context.tool_name}",
            f"{'='*70}",
            f"💡 {result.message}",
        ]
        
        # Add suggestions if available
        if result.suggestions:
            lines.append("\n🛠️ SUGGESTIONS:")
            for i, suggestion in enumerate(result.suggestions[:3], 1):
                lines.append(f"  {i}. {suggestion}")
        
        # Add performance info if available
        if result.performance_impact:
            lines.append(f"\n⚡ Analysis time: {result.performance_impact:.3f}s")
        
        lines.append(f"{'='*70}\n")
        
        print("\n".join(lines), file=sys.stderr)
    
    def _output_error_message(self, result: FeedbackResult, context: ToolContext):
        """Output error message to stderr."""
        lines = [
            f"\n{'='*70}",
            f"🚨 TOOL USAGE ERROR - {result.analyzer_name.upper()}",
            f"Tool: {context.tool_name}",
            f"{'='*70}",
            f"❌ {result.message}",
        ]
        
        # Add suggestions for resolution
        if result.suggestions:
            lines.append("\n🔧 RESOLUTION STEPS:")
            for i, suggestion in enumerate(result.suggestions[:3], 1):
                lines.append(f"  {i}. {suggestion}")
        
        lines.append(f"{'='*70}\n")
        
        print("\n".join(lines), file=sys.stderr)
    
    def _update_integration_stats(self, execution_time: float, successful: bool):
        """Update integration performance statistics."""
        self.integration_stats["total_integrations"] += 1
        
        if successful:
            self.integration_stats["successful_integrations"] += 1
        
        # Update average execution time
        prev_avg = self.integration_stats["average_execution_time"]
        total_count = self.integration_stats["total_integrations"]
        self.integration_stats["average_execution_time"] = (
            (prev_avg * (total_count - 1) + execution_time) / total_count
        )
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics."""
        return {
            **self.integration_stats,
            "registry_info": self.registry.get_registry_info()
        }


class BackwardCompatibilityLayer:
    """Ensures backward compatibility with existing PostToolUse patterns."""
    
    @staticmethod
    def convert_drift_evidence_to_feedback(evidence: DriftEvidence) -> FeedbackResult:
        """Convert existing DriftEvidence to new FeedbackResult format."""
        severity_mapping = {
            evidence.severity.NONE: FeedbackSeverity.INFO,
            evidence.severity.MINOR: FeedbackSeverity.INFO,
            evidence.severity.MODERATE: FeedbackSeverity.WARNING,
            evidence.severity.MAJOR: FeedbackSeverity.WARNING,
            evidence.severity.CRITICAL: FeedbackSeverity.ERROR
        }
        
        return FeedbackResult(
            severity=severity_mapping.get(evidence.severity, FeedbackSeverity.INFO),
            message=evidence.evidence_details,
            suggestions=[evidence.correction_guidance] if evidence.correction_guidance else [],
            metadata={
                "drift_type": evidence.drift_type.value,
                "tool_sequence": evidence.tool_sequence,
                "missing_tools": evidence.missing_tools,
                "priority_score": evidence.priority_score
            },
            analyzer_name="legacy_drift_detector"
        )
    
    @staticmethod
    def create_legacy_workflow_context(
        tool_name: str, 
        tool_input: Dict[str, Any], 
        tool_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create context compatible with existing workflow analyzers."""
        return {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "success": tool_response.get("success", True),
            "timestamp": time.time()
        }


def create_hook_integrator() -> PostToolHookIntegrator:
    """Factory function to create hook integrator with proper initialization."""
    # Get global registry and ensure it's initialized
    registry = get_global_registry()
    
    # Create integrator
    integrator = PostToolHookIntegrator(registry)
    
    return integrator


# Convenience function for direct integration into existing PostToolUse hook
async def analyze_tool_for_hook(
    tool_name: str, 
    tool_input: Dict[str, Any], 
    tool_response: Dict[str, Any],
    session_context: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """Convenience function for direct integration into PostToolUse hook.
    
    Args:
        tool_name: Name of the tool used
        tool_input: Input parameters passed to tool
        tool_response: Response received from tool
        session_context: Optional session context information
        
    Returns:
        Exit code (0=success, 1=error, 2=guidance) or None for no action
    """
    integrator = create_hook_integrator()
    return await integrator.process_tool_usage(
        tool_name, tool_input, tool_response, session_context
    )


def analyze_tool_for_hook_sync(
    tool_name: str, 
    tool_input: Dict[str, Any], 
    tool_response: Dict[str, Any],
    session_context: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """Synchronous wrapper for hook integration.
    
    This function can be directly called from the existing PostToolUse hook
    without requiring async/await syntax changes.
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
            analyze_tool_for_hook(tool_name, tool_input, tool_response, session_context)
        )
    
    except Exception as e:
        print(f"Warning: Sync hook integration error: {e}", file=sys.stderr)
        return None  # Fall back to existing behavior