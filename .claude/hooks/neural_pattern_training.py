#!/usr/bin/env python3
"""Neural Pattern Training Hook - claude-flow integration.

Post-operation hook that trains the neural pattern validator based on successful operations.
This implements claude-flow's neural learning capabilities to create self-improving workflows.

Called after tool operations to learn from successful patterns and improve future guidance.
"""

import json
import sys
from typing import Dict, Any

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

from modules.pre_tool.analyzers.neural_pattern_validator import (
    NeuralPatternValidator, 
    NeuralPatternStorage
)


class NeuralTrainingCoordinator:
    """Coordinates neural pattern training from successful operations."""
    
    def __init__(self):
        self.pattern_validator = NeuralPatternValidator(learning_enabled=True)
        self.pattern_storage = NeuralPatternStorage()
        self.training_session_id = None
    
    def train_from_operation(self, operation_data: Dict[str, Any]) -> None:
        """Train neural patterns from successful operation data."""
        
        try:
            # Extract operation details
            tool_name = operation_data.get("tool_name", "")
            tool_input = operation_data.get("tool_input", {})
            success_outcome = operation_data.get("success", False)
            optimization_applied = operation_data.get("optimization_applied", "")
            execution_time = operation_data.get("execution_time", 0.0)
            
            if not tool_name or not success_outcome:
                return
            
            # Generate context hash for this operation
            context_hash = self._generate_operation_context_hash(operation_data)
            
            # Extract learned optimization from the operation
            learned_optimization = self._extract_learned_optimization(
                operation_data, optimization_applied
            )
            
            # Train the neural pattern
            self.pattern_validator.learn_from_success(
                tool_name=tool_name,
                tool_input=tool_input,
                context_hash=context_hash,
                optimization_applied=learned_optimization
            )
            
            # Log training success
            self._log_training_event(tool_name, learned_optimization, execution_time)
            
        except Exception as e:
            print(f"Neural training error: {e}", file=sys.stderr)
    
    def _generate_operation_context_hash(self, operation_data: Dict[str, Any]) -> str:
        """Generate context hash for the operation."""
        # Simplified context hash generation
        context_elements = {
            "tool_name": operation_data.get("tool_name", ""),
            "session_stage": operation_data.get("session_stage", "unknown"),
            "previous_tools": operation_data.get("previous_tools", []),
            "zen_active": operation_data.get("zen_coordination_active", False),
            "flow_active": operation_data.get("flow_coordination_active", False)
        }
        
        import hashlib
        context_str = json.dumps(context_elements, sort_keys=True)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]
    
    def _extract_learned_optimization(self, operation_data: Dict[str, Any], 
                                    optimization_applied: str) -> str:
        """Extract learned optimization from successful operation."""
        
        # If explicit optimization was applied, use that
        if optimization_applied:
            return optimization_applied
        
        # Infer optimization from successful operation patterns
        tool_name = operation_data.get("tool_name", "")
        success_factors = operation_data.get("success_factors", [])
        
        optimizations = []
        
        # ZEN coordination optimizations
        if operation_data.get("zen_coordination_active"):
            optimizations.append("Queen ZEN coordination ensures optimal results")
        
        # MCP tool optimizations
        if tool_name.startswith("mcp__"):
            optimizations.append(f"MCP tool {tool_name} provides superior coordination")
        
        # Flow coordination optimizations
        if operation_data.get("flow_coordination_active"):
            optimizations.append("Flow Worker coordination enhances efficiency")
        
        # Performance optimizations
        execution_time = operation_data.get("execution_time", 0.0)
        if execution_time > 0 and execution_time < 1.0:
            optimizations.append("Fast execution achieved through optimal tool selection")
        
        # Success factor optimizations
        for factor in success_factors:
            if "batching" in factor.lower():
                optimizations.append("Batched operations improve efficiency by 300%")
            elif "coordination" in factor.lower():
                optimizations.append("Proper coordination prevents errors and enhances results")
            elif "zen" in factor.lower():
                optimizations.append("Queen ZEN's guidance optimizes complex operations")
        
        # Return the most relevant optimization
        if optimizations:
            return optimizations[0]  # Primary optimization
        else:
            return f"Successful {tool_name} operation - pattern learned"
    
    def _log_training_event(self, tool_name: str, optimization: str, 
                           execution_time: float) -> None:
        """Log neural training event for monitoring."""
        
        training_log = {
            "event": "neural_training",
            "tool_name": tool_name,
            "optimization_learned": optimization,
            "execution_time": execution_time,
            "timestamp": __import__("time").time()
        }
        
        # Log to stderr for Claude Code visibility
        print(f"🧠 Neural Training: Learned pattern for {tool_name}", file=sys.stderr)
        
        # In production, this would also log to structured logging system
        # or claude-flow memory system for cross-session persistence


def main():
    """Main entry point for neural pattern training hook."""
    
    try:
        # Read operation data from stdin (passed by Claude Code after operations)
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            # No training data provided - this is normal for many operations
            sys.exit(0)
        
        # Parse operation data
        try:
            operation_data = json.loads(input_data)
        except json.JSONDecodeError:
            # Invalid JSON - skip training
            sys.exit(0)
        
        # Initialize neural training coordinator
        trainer = NeuralTrainingCoordinator()
        
        # Train from the operation
        trainer.train_from_operation(operation_data)
        
        # Get neural metrics for monitoring
        metrics = trainer.pattern_validator.get_neural_metrics()
        
        # Output training success metrics
        print("🧠 Neural Training Complete:", file=sys.stderr)
        print(f"   Total Patterns: {metrics['total_patterns']}", file=sys.stderr)
        print(f"   High Confidence: {metrics['high_confidence_patterns']}", file=sys.stderr)
        print(f"   Learning Effectiveness: {metrics['neural_effectiveness']:.1f}%", file=sys.stderr)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"Neural training hook error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()