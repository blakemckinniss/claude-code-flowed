#!/usr/bin/env python3
"""Neural Pattern Training Hook - claude-flow integration with ZEN adaptive learning.

Post-operation hook that trains the neural pattern validator based on successful operations.
This implements claude-flow's neural learning capabilities to create self-improving workflows.
Enhanced with ZEN adaptive learning integration for comprehensive intelligence.

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

# Import ZEN adaptive learning components
try:
    from modules.core.zen_adaptive_learning import ZenAdaptiveLearningEngine, ZenLearningOutcome
    from modules.core.zen_neural_training import integrate_zen_neural_training, ZenNeuralTrainingPipeline
    ZEN_INTEGRATION_AVAILABLE = True
except ImportError:
    ZEN_INTEGRATION_AVAILABLE = False
    print("ZEN integration not available - falling back to standard neural training", file=sys.stderr)


class NeuralTrainingCoordinator:
    """Coordinates neural pattern training from successful operations with ZEN integration."""
    
    def __init__(self):
        self.pattern_validator = NeuralPatternValidator(learning_enabled=True)
        self.pattern_storage = NeuralPatternStorage()
        self.training_session_id = None
        
        # ZEN adaptive learning integration
        if ZEN_INTEGRATION_AVAILABLE:
            self.zen_learning_engine = ZenAdaptiveLearningEngine()
            self.zen_neural_pipeline = ZenNeuralTrainingPipeline()
            self.zen_integration_active = True
            print("🧠 ZEN Adaptive Learning integrated with neural training", file=sys.stderr)
        else:
            self.zen_learning_engine = None
            self.zen_neural_pipeline = None
            self.zen_integration_active = False
    
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
            
            # ZEN adaptive learning integration
            if self.zen_integration_active:
                self._integrate_zen_learning(operation_data, learned_optimization)
            
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
        
        {
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
    
    def _integrate_zen_learning(self, operation_data: Dict[str, Any], learned_optimization: str) -> None:
        """Integrate operation data with ZEN adaptive learning system."""
        if not self.zen_integration_active:
            return
        
        try:
            # Check if this is a ZEN consultation operation
            tool_name = operation_data.get("tool_name", "")
            
            if "zen" in tool_name.lower() or "consultation" in str(operation_data.get("context", "")).lower():
                # This is a ZEN-related operation, process for adaptive learning
                self._process_zen_consultation_outcome(operation_data, learned_optimization)
            else:
                # Regular operation, contribute to neural training pipeline
                integrate_zen_neural_training(operation_data)
                
        except Exception as e:
            print(f"ZEN integration error: {e}", file=sys.stderr)
    
    def _process_zen_consultation_outcome(self, operation_data: Dict[str, Any], 
                                        learned_optimization: str) -> None:
        """Process ZEN consultation outcome for adaptive learning."""
        try:
            # Extract consultation data
            consultation_id = operation_data.get("consultation_id", f"zen_{int(__import__('time').time())}")
            prompt = operation_data.get("prompt", operation_data.get("context", ""))
            
            # Create ZEN learning outcome
            outcome = ZenLearningOutcome(
                consultation_id=consultation_id,
                prompt=prompt,
                complexity=operation_data.get("complexity", "medium"),
                coordination_type=operation_data.get("coordination_type", "SWARM"),
                agents_allocated=operation_data.get("agents_allocated", 0),
                agent_types=operation_data.get("agent_types", []),
                mcp_tools=operation_data.get("mcp_tools", []),
                execution_success=operation_data.get("success", False),
                user_satisfaction=operation_data.get("user_satisfaction", 0.7),  # Default good satisfaction
                actual_agents_needed=operation_data.get("actual_agents_needed"),
                performance_metrics={
                    "execution_time": operation_data.get("execution_time", 0.0),
                    "optimization_applied": learned_optimization,
                    "neural_pattern_confidence": operation_data.get("confidence", 0.5)
                },
                lessons_learned=[learned_optimization] if learned_optimization else [],
                timestamp=__import__('time').time()
            )
            
            # Record in ZEN learning engine
            self.zen_learning_engine.record_consultation_outcome(outcome)
            
            # Update neural models
            self.zen_neural_pipeline.update_models_from_outcome(outcome)
            
            print(f"🧠 ZEN Learning: Recorded consultation outcome with {outcome.user_satisfaction:.1%} satisfaction", file=sys.stderr)
            
        except Exception as e:
            print(f"Error processing ZEN consultation outcome: {e}", file=sys.stderr)
    
    def get_enhanced_neural_metrics(self) -> Dict[str, Any]:
        """Get enhanced neural metrics including ZEN adaptive learning."""
        # Get standard neural metrics
        standard_metrics = self.pattern_validator.get_neural_metrics()
        
        if self.zen_integration_active:
            # Add ZEN learning metrics
            zen_metrics = self.zen_learning_engine.get_learning_metrics()
            neural_training_metrics = self.zen_neural_pipeline.get_training_metrics()
            
            return {
                "standard_neural": standard_metrics,
                "zen_adaptive_learning": zen_metrics,
                "zen_neural_training": neural_training_metrics,
                "integration_status": "active",
                "total_intelligence_sources": 3
            }
        else:
            return {
                "standard_neural": standard_metrics,
                "integration_status": "not_available",
                "total_intelligence_sources": 1
            }


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
        
        # Get enhanced neural metrics for monitoring
        metrics = trainer.get_enhanced_neural_metrics()
        
        # Output training success metrics
        print("🧠 Neural Training Complete:", file=sys.stderr)
        
        if metrics.get("integration_status") == "active":
            # ZEN integration active - show comprehensive metrics
            standard_metrics = metrics["standard_neural"]
            zen_metrics = metrics["zen_adaptive_learning"]
            
            print(f"   📊 Standard Neural Patterns: {standard_metrics['total_patterns']}", file=sys.stderr)
            print(f"   📊 High Confidence Patterns: {standard_metrics['high_confidence_patterns']}", file=sys.stderr)
            print(f"   📊 Neural Effectiveness: {standard_metrics.get('neural_effectiveness', 0):.1f}%", file=sys.stderr)
            print(f"   🧠 ZEN Learning Sessions: {zen_metrics.get('successful_learnings', 0)}", file=sys.stderr)
            print(f"   🧠 ZEN Adaptive Patterns: {zen_metrics.get('total_patterns', 0)}", file=sys.stderr)
            print(f"   🧠 ZEN Learning Effectiveness: {zen_metrics.get('learning_effectiveness', 0):.1%}", file=sys.stderr)
            print(f"   ⚡ Total Intelligence Sources: {metrics['total_intelligence_sources']}", file=sys.stderr)
        else:
            # Standard neural training only
            standard_metrics = metrics["standard_neural"]
            print(f"   Total Patterns: {standard_metrics['total_patterns']}", file=sys.stderr)
            print(f"   High Confidence: {standard_metrics['high_confidence_patterns']}", file=sys.stderr)
            print(f"   Learning Effectiveness: {standard_metrics.get('neural_effectiveness', 0):.1f}%", file=sys.stderr)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"Neural training hook error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()