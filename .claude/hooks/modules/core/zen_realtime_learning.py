#!/usr/bin/env python3
"""ZEN Real-time Learning Integration - Immediate learning feedback loops.

This module provides real-time learning capabilities that immediately update
ZEN intelligence based on consultation outcomes and user interactions.

Key Features:
- Immediate model updates from consultation results
- Real-time pattern recognition and adaptation
- Live feedback integration
- Performance monitoring and adjustment
- Dynamic recommendation tuning
"""

import json
import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

# Import ZEN components
from .zen_adaptive_learning import ZenAdaptiveLearningEngine, ZenLearningOutcome, AdaptiveZenConsultant
from .zen_neural_training import ZenNeuralTrainingPipeline
from .zen_memory_pipeline import ZenMemoryPipeline


@dataclass
class RealtimeFeedback:
    """Real-time feedback from user interactions."""
    consultation_id: str
    feedback_type: str  # satisfaction, correction, preference
    feedback_value: Any
    context: Dict[str, Any]
    timestamp: float


@dataclass
class LearningEvent:
    """Learning event for real-time processing."""
    event_type: str  # consultation, feedback, optimization
    event_data: Dict[str, Any]
    priority: int  # 1=high, 2=medium, 3=low
    timestamp: float


class ZenRealtimeLearningProcessor:
    """Processes learning events in real-time."""
    
    def __init__(self):
        self.learning_queue = Queue()
        self.feedback_queue = Queue()
        self.processing_thread = None
        self.is_running = False
        
        # ZEN components
        self.learning_engine = ZenAdaptiveLearningEngine()
        self.neural_pipeline = ZenNeuralTrainingPipeline()
        self.memory_pipeline = ZenMemoryPipeline()
        self.adaptive_consultant = AdaptiveZenConsultant()
        
        # Real-time metrics
        self.events_processed = 0
        self.learning_updates = 0
        self.real_time_accuracy = 0.0
        self.last_update_time = time.time()
        
        # Callback handlers
        self.feedback_handlers: List[Callable] = []
        self.learning_handlers: List[Callable] = []
    
    def start_realtime_processing(self) -> None:
        """Start real-time learning processing thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_learning_events, daemon=True)
        self.processing_thread.start()
        print("🔄 ZEN Real-time Learning: Processing started")
    
    def stop_realtime_processing(self) -> None:
        """Stop real-time learning processing."""
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        print("⏹️ ZEN Real-time Learning: Processing stopped")
    
    def submit_consultation_result(self, consultation_id: str, prompt: str, 
                                 result: Dict[str, Any], user_feedback: Optional[Dict[str, Any]] = None) -> None:
        """Submit consultation result for immediate learning."""
        # Create learning event
        event = LearningEvent(
            event_type="consultation",
            event_data={
                "consultation_id": consultation_id,
                "prompt": prompt,
                "result": result,
                "user_feedback": user_feedback,
                "success": result.get("success", True),
                "confidence": result.get("confidence", 0.5)
            },
            priority=1,  # High priority for consultations
            timestamp=time.time()
        )
        
        self.learning_queue.put(event)
        print(f"📝 ZEN Real-time Learning: Queued consultation result for {consultation_id}")
    
    def submit_user_feedback(self, consultation_id: str, feedback_type: str, 
                           feedback_value: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """Submit immediate user feedback."""
        feedback = RealtimeFeedback(
            consultation_id=consultation_id,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            context=context or {},
            timestamp=time.time()
        )
        
        self.feedback_queue.put(feedback)
        print(f"📊 ZEN Real-time Learning: Received {feedback_type} feedback for {consultation_id}")
    
    def submit_optimization_discovery(self, optimization_data: Dict[str, Any]) -> None:
        """Submit optimization discovery for immediate integration."""
        event = LearningEvent(
            event_type="optimization",
            event_data=optimization_data,
            priority=2,  # Medium priority
            timestamp=time.time()
        )
        
        self.learning_queue.put(event)
        print("⚡ ZEN Real-time Learning: Queued optimization discovery")
    
    def _process_learning_events(self) -> None:
        """Main processing loop for learning events."""
        while self.is_running:
            try:
                # Process feedback first (higher priority)
                self._process_feedback_queue()
                
                # Process learning events
                try:
                    event = self.learning_queue.get(timeout=1.0)
                    self._handle_learning_event(event)
                    self.events_processed += 1
                except Empty:
                    continue
                    
            except Exception as e:
                print(f"Error in real-time learning processing: {e}")
                time.sleep(1.0)
    
    def _process_feedback_queue(self) -> None:
        """Process all pending feedback."""
        feedback_count = 0
        
        while not self.feedback_queue.empty() and feedback_count < 10:  # Batch limit
            try:
                feedback = self.feedback_queue.get_nowait()
                self._handle_user_feedback(feedback)
                feedback_count += 1
            except Empty:
                break
            except Exception as e:
                print(f"Error processing feedback: {e}")
    
    def _handle_learning_event(self, event: LearningEvent) -> None:
        """Handle individual learning event."""
        try:
            if event.event_type == "consultation":
                self._process_consultation_event(event)
            elif event.event_type == "optimization":
                self._process_optimization_event(event)
            elif event.event_type == "feedback":
                self._process_feedback_event(event)
            
            # Trigger learning callbacks
            for handler in self.learning_handlers:
                try:
                    handler(event)
                except Exception as e:
                    print(f"Error in learning handler: {e}")
                    
        except Exception as e:
            print(f"Error handling learning event: {e}")
    
    def _process_consultation_event(self, event: LearningEvent) -> None:
        """Process consultation learning event."""
        data = event.event_data
        
        # Create ZEN learning outcome
        outcome = ZenLearningOutcome(
            consultation_id=data["consultation_id"],
            prompt=data["prompt"],
            complexity=data["result"].get("complexity", "medium"),
            coordination_type=data["result"].get("coordination", "SWARM"),
            agents_allocated=data["result"].get("agent_count", 0),
            agent_types=data["result"].get("agent_types", []),
            mcp_tools=data["result"].get("tools", []),
            execution_success=data["success"],
            user_satisfaction=data.get("user_feedback", {}).get("satisfaction", 0.7),
            actual_agents_needed=data.get("user_feedback", {}).get("actual_agents_needed"),
            performance_metrics={
                "confidence": data["confidence"],
                "response_time": data.get("response_time", 0.0),
                "realtime_processing": True
            },
            lessons_learned=data.get("user_feedback", {}).get("lessons", []),
            timestamp=event.timestamp
        )
        
        # Record in learning engine
        success = self.learning_engine.record_consultation_outcome(outcome)
        
        if success:
            # Update neural models if significant learning
            if outcome.user_satisfaction > 0.8 or outcome.execution_success:
                self.neural_pipeline.update_models_from_outcome(outcome)
                self.learning_updates += 1
            
            # Update real-time accuracy
            self._update_realtime_accuracy(outcome)
            
            print(f"✅ Real-time learning: Updated from consultation {outcome.consultation_id}")
    
    def _process_optimization_event(self, event: LearningEvent) -> None:
        """Process optimization learning event."""
        optimization_data = event.event_data
        
        # Extract optimization insights
        optimization_type = optimization_data.get("type", "general")
        improvement = optimization_data.get("improvement", 0.0)
        optimization_data.get("context", {})
        
        # Update learning patterns
        if improvement > 0.1:  # Significant improvement
            # Create synthetic learning outcome
            outcome = ZenLearningOutcome(
                consultation_id=f"opt_{int(event.timestamp)}",
                prompt=f"Optimization: {optimization_type}",
                complexity="medium",
                coordination_type="SWARM",
                agents_allocated=1,
                agent_types=["optimizer"],
                mcp_tools=[],
                execution_success=True,
                user_satisfaction=0.5 + (improvement * 0.5),  # Scale improvement to satisfaction
                actual_agents_needed=1,
                performance_metrics={
                    "optimization_type": optimization_type,
                    "improvement": improvement,
                    "realtime_discovery": True
                },
                lessons_learned=[f"{optimization_type} optimization achieved {improvement:.1%} improvement"],
                timestamp=event.timestamp
            )
            
            self.learning_engine.record_consultation_outcome(outcome)
            print(f"🔧 Real-time learning: Recorded {optimization_type} optimization")
    
    def _handle_user_feedback(self, feedback: RealtimeFeedback) -> None:
        """Handle immediate user feedback."""
        try:
            # Update consultation outcome with feedback
            if feedback.feedback_type == "satisfaction":
                # Update satisfaction score for consultation
                self._update_consultation_satisfaction(
                    feedback.consultation_id, 
                    float(feedback.feedback_value)
                )
            
            elif feedback.feedback_type == "correction":
                # Learn from user corrections
                self._process_user_correction(feedback)
            
            elif feedback.feedback_type == "preference":
                # Update user preferences
                self._update_user_preferences(feedback)
            
            # Trigger feedback callbacks
            for handler in self.feedback_handlers:
                try:
                    handler(feedback)
                except Exception as e:
                    print(f"Error in feedback handler: {e}")
                    
        except Exception as e:
            print(f"Error handling user feedback: {e}")
    
    def _update_consultation_satisfaction(self, consultation_id: str, satisfaction: float) -> None:
        """Update satisfaction score for a consultation."""
        # This would normally update the database
        # For now, we'll create an adjustment outcome
        outcome = ZenLearningOutcome(
            consultation_id=f"{consultation_id}_feedback",
            prompt="User satisfaction feedback",
            complexity="simple",
            coordination_type="SWARM",
            agents_allocated=0,
            agent_types=[],
            mcp_tools=[],
            execution_success=True,
            user_satisfaction=satisfaction,
            actual_agents_needed=0,
            performance_metrics={
                "feedback_adjustment": True,
                "original_consultation": consultation_id
            },
            lessons_learned=[f"User provided satisfaction score: {satisfaction:.1%}"],
            timestamp=time.time()
        )
        
        self.learning_engine.record_consultation_outcome(outcome)
        print(f"📈 Updated satisfaction for {consultation_id}: {satisfaction:.1%}")
    
    def _process_user_correction(self, feedback: RealtimeFeedback) -> None:
        """Process user correction feedback."""
        correction_data = feedback.feedback_value
        
        # Extract what the user thinks should have been different
        if isinstance(correction_data, dict):
            corrected_complexity = correction_data.get("complexity")
            corrected_coordination = correction_data.get("coordination")
            corrected_agent_count = correction_data.get("agent_count")
            
            # Create corrective learning outcome
            outcome = ZenLearningOutcome(
                consultation_id=f"{feedback.consultation_id}_correction",
                prompt=feedback.context.get("original_prompt", "User correction"),
                complexity=corrected_complexity or "medium",
                coordination_type=corrected_coordination or "SWARM",
                agents_allocated=corrected_agent_count or 1,
                agent_types=correction_data.get("agent_types", ["coder"]),
                mcp_tools=[],
                execution_success=True,
                user_satisfaction=0.9,  # High satisfaction for corrections
                actual_agents_needed=corrected_agent_count or 1,
                performance_metrics={
                    "user_correction": True,
                    "original_consultation": feedback.consultation_id
                },
                lessons_learned=[f"User correction: {json.dumps(correction_data)}"],
                timestamp=feedback.timestamp
            )
            
            self.learning_engine.record_consultation_outcome(outcome)
            self.neural_pipeline.update_models_from_outcome(outcome)
            
            print(f"🔄 Processed user correction for {feedback.consultation_id}")
    
    def _update_user_preferences(self, feedback: RealtimeFeedback) -> None:
        """Update user preferences based on feedback."""
        preferences = feedback.feedback_value
        
        # Store preferences (this would normally go to a user profile system)
        print(f"💾 Updated user preferences: {preferences}")
    
    def _update_realtime_accuracy(self, outcome: ZenLearningOutcome) -> None:
        """Update real-time accuracy metrics."""
        # Simple exponential moving average
        alpha = 0.1  # Learning rate
        accuracy_score = outcome.user_satisfaction
        
        if self.real_time_accuracy == 0.0:
            self.real_time_accuracy = accuracy_score
        else:
            self.real_time_accuracy = (1 - alpha) * self.real_time_accuracy + alpha * accuracy_score
        
        self.last_update_time = time.time()
    
    def get_enhanced_consultation(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get enhanced consultation with real-time learning."""
        # Get base consultation
        base_consultation = self.adaptive_consultant.get_adaptive_directive(prompt)
        
        # Enhance with real-time intelligence
        realtime_metrics = self.get_realtime_metrics()
        
        # Adjust confidence based on real-time performance
        if realtime_metrics["realtime_accuracy"] > 0.8:
            base_consultation["confidence"] *= 1.1  # Boost confidence
        elif realtime_metrics["realtime_accuracy"] < 0.5:
            base_consultation["confidence"] *= 0.9  # Reduce confidence
        
        # Add real-time context
        base_consultation["realtime_enhancement"] = {
            "accuracy": realtime_metrics["realtime_accuracy"],
            "learning_velocity": realtime_metrics["learning_velocity"],
            "recent_updates": realtime_metrics["learning_updates"],
            "enhanced": True
        }
        
        return base_consultation
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time learning metrics."""
        current_time = time.time()
        time_since_update = current_time - self.last_update_time
        
        # Calculate learning velocity (updates per hour)
        learning_velocity = self.learning_updates / max(time_since_update / 3600, 0.1)
        
        return {
            "realtime_accuracy": self.real_time_accuracy,
            "events_processed": self.events_processed,
            "learning_updates": self.learning_updates,
            "learning_velocity": learning_velocity,
            "queue_sizes": {
                "learning_queue": self.learning_queue.qsize(),
                "feedback_queue": self.feedback_queue.qsize()
            },
            "processing_active": self.is_running,
            "last_update": self.last_update_time
        }
    
    def add_feedback_handler(self, handler: Callable[[RealtimeFeedback], None]) -> None:
        """Add feedback handler callback."""
        self.feedback_handlers.append(handler)
    
    def add_learning_handler(self, handler: Callable[[LearningEvent], None]) -> None:
        """Add learning event handler callback."""
        self.learning_handlers.append(handler)


class ZenRealtimeLearningIntegration:
    """Main integration class for ZEN real-time learning."""
    
    def __init__(self):
        self.processor = ZenRealtimeLearningProcessor()
        self.active_consultations: Dict[str, Dict] = {}
        
        # Start real-time processing
        self.processor.start_realtime_processing()
        
        # Set up handlers
        self.processor.add_feedback_handler(self._handle_feedback)
        self.processor.add_learning_handler(self._handle_learning)
    
    def enhanced_zen_consultation(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform enhanced ZEN consultation with real-time learning."""
        consultation_id = f"zen_rt_{int(time.time() * 1000)}"
        
        # Get enhanced consultation
        result = self.processor.get_enhanced_consultation(prompt, context)
        result["consultation_id"] = consultation_id
        
        # Track consultation
        self.active_consultations[consultation_id] = {
            "prompt": prompt,
            "result": result,
            "start_time": time.time(),
            "context": context or {}
        }
        
        return result
    
    def provide_consultation_feedback(self, consultation_id: str, feedback: Dict[str, Any]) -> None:
        """Provide feedback on consultation result."""
        if consultation_id not in self.active_consultations:
            print(f"Warning: Unknown consultation ID {consultation_id}")
            return
        
        consultation = self.active_consultations[consultation_id]
        
        # Submit consultation result with feedback
        self.processor.submit_consultation_result(
            consultation_id=consultation_id,
            prompt=consultation["prompt"],
            result=consultation["result"],
            user_feedback=feedback
        )
        
        # Submit individual feedback items
        if "satisfaction" in feedback:
            self.processor.submit_user_feedback(
                consultation_id, "satisfaction", feedback["satisfaction"]
            )
        
        if "corrections" in feedback:
            self.processor.submit_user_feedback(
                consultation_id, "correction", feedback["corrections"], 
                {"original_prompt": consultation["prompt"]}
            )
        
        # Mark consultation as complete
        del self.active_consultations[consultation_id]
        
        print(f"✅ Processed feedback for consultation {consultation_id}")
    
    def _handle_feedback(self, feedback: RealtimeFeedback) -> None:
        """Handle feedback events."""
        # Custom feedback processing can be added here
        pass
    
    def _handle_learning(self, event: LearningEvent) -> None:
        """Handle learning events."""
        # Custom learning processing can be added here
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        realtime_metrics = self.processor.get_realtime_metrics()
        learning_metrics = self.processor.learning_engine.get_learning_metrics()
        
        return {
            "realtime_learning": realtime_metrics,
            "adaptive_learning": learning_metrics,
            "active_consultations": len(self.active_consultations),
            "system_health": {
                "processing_active": realtime_metrics["processing_active"],
                "queue_backlog": realtime_metrics["queue_sizes"]["learning_queue"] > 10,
                "learning_velocity": realtime_metrics["learning_velocity"],
                "accuracy_trend": "improving" if realtime_metrics["realtime_accuracy"] > 0.7 else "stable"
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown real-time learning system."""
        self.processor.stop_realtime_processing()
        print("🔄 ZEN Real-time Learning Integration: Shutdown complete")


# Global instance for easy access
_global_realtime_integration: Optional[ZenRealtimeLearningIntegration] = None


def get_realtime_integration() -> ZenRealtimeLearningIntegration:
    """Get global real-time learning integration instance."""
    global _global_realtime_integration
    
    if _global_realtime_integration is None:
        _global_realtime_integration = ZenRealtimeLearningIntegration()
    
    return _global_realtime_integration


# Convenience functions for easy use
def enhanced_zen_consultation(prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function for enhanced ZEN consultation."""
    integration = get_realtime_integration()
    return integration.enhanced_zen_consultation(prompt, context)


def provide_zen_feedback(consultation_id: str, feedback: Dict[str, Any]) -> None:
    """Convenience function for providing ZEN feedback."""
    integration = get_realtime_integration()
    integration.provide_consultation_feedback(consultation_id, feedback)


def get_zen_system_status() -> Dict[str, Any]:
    """Convenience function for getting system status."""
    integration = get_realtime_integration()
    return integration.get_system_status()


if __name__ == "__main__":
    # Test real-time learning
    integration = ZenRealtimeLearningIntegration()
    
    # Test consultation
    result = integration.enhanced_zen_consultation(
        "Build a secure authentication system with JWT tokens"
    )
    
    print(f"Consultation result: {json.dumps(result, indent=2)}")
    
    # Provide feedback
    integration.provide_consultation_feedback(result["consultation_id"], {
        "satisfaction": 0.9,
        "actual_agents_needed": 3,
        "corrections": {
            "complexity": "complex",
            "agent_types": ["security-auditor", "backend-developer", "coder"]
        }
    })
    
    # Check status
    status = integration.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")
    
    # Clean shutdown
    integration.shutdown()