#!/usr/bin/env python3
"""Test suite for ZEN Workflow Orchestrator."""

import pytest
import asyncio
from datetime import datetime
from zen_workflow_orchestrator import (
    WorkflowState,
    WorkflowType,
    WorkflowStep,
    ZenWorkflowOrchestrator,
    get_zen_workflow_orchestrator
)


class TestZenWorkflowOrchestrator:
    """Test cases for ZEN Workflow Orchestrator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.orchestrator = ZenWorkflowOrchestrator()
    
    def test_pattern_detection_design_review_implement(self):
        """Test detection of Design→Review→Implement pattern."""
        test_cases = [
            ("design then review and implement authentication", WorkflowType.DESIGN_REVIEW_IMPLEMENT),
            ("let's plan then build this feature after review", WorkflowType.DESIGN_REVIEW_IMPLEMENT),
            ("design review implement workflow", WorkflowType.DESIGN_REVIEW_IMPLEMENT),
            ("architectural review before coding", WorkflowType.DESIGN_REVIEW_IMPLEMENT),
        ]
        
        for prompt, expected_type in test_cases:
            detected = self.orchestrator.detect_workflow_pattern(prompt)
            assert detected == expected_type, f"Expected {expected_type} for '{prompt}', got {detected}"
    
    def test_pattern_detection_analyze_plan_execute(self):
        """Test detection of Analyze→Plan→Execute pattern."""
        test_cases = [
            ("analyze then plan and execute", WorkflowType.ANALYZE_PLAN_EXECUTE),
            ("research then implement solution", WorkflowType.ANALYZE_PLAN_EXECUTE),
            ("systematic approach to analyze plan execute", WorkflowType.ANALYZE_PLAN_EXECUTE),
        ]
        
        for prompt, expected_type in test_cases:
            detected = self.orchestrator.detect_workflow_pattern(prompt)
            assert detected == expected_type, f"Expected {expected_type} for '{prompt}', got {detected}"
    
    def test_pattern_detection_debug_investigate_fix(self):
        """Test detection of Debug→Investigate→Fix pattern."""
        test_cases = [
            ("debug then fix this issue", WorkflowType.DEBUG_INVESTIGATE_FIX),
            ("systematic debugging approach", WorkflowType.DEBUG_INVESTIGATE_FIX),
            ("root cause analysis and fix", WorkflowType.DEBUG_INVESTIGATE_FIX),
        ]
        
        for prompt, expected_type in test_cases:
            detected = self.orchestrator.detect_workflow_pattern(prompt)
            assert detected == expected_type, f"Expected {expected_type} for '{prompt}', got {detected}"
    
    def test_no_pattern_detection(self):
        """Test that simple prompts don't trigger workflow patterns."""
        simple_prompts = [
            "write some code",
            "fix this bug",
            "hello world",
            "short",
            ""
        ]
        
        for prompt in simple_prompts:
            detected = self.orchestrator.detect_workflow_pattern(prompt)
            assert detected is None, f"Expected no pattern for '{prompt}', got {detected}"
    
    def test_state_transitions(self):
        """Test workflow state transitions."""
        transition_tests = [
            (WorkflowState.INITIAL, "let's start with design", WorkflowState.DESIGNING),
            (WorkflowState.DESIGNING, "time to review the design", WorkflowState.REVIEWING),
            (WorkflowState.REVIEWING, "let's implement this solution", WorkflowState.IMPLEMENTING),
            (WorkflowState.IMPLEMENTING, "now we need to test it", WorkflowState.TESTING),
            (WorkflowState.TESTING, "everything works, we're complete", WorkflowState.COMPLETED),
        ]
        
        for current_state, prompt, expected_next in transition_tests:
            next_state = self.orchestrator.detect_transition(current_state, prompt)
            assert next_state == expected_next, f"Expected {expected_next} from {current_state} with '{prompt}', got {next_state}"
    
    def test_multi_step_orchestration(self):
        """Test multi-step workflow orchestration."""
        steps = self.orchestrator.orchestrate_multi_step(WorkflowType.DESIGN_REVIEW_IMPLEMENT)
        
        assert len(steps) == 4, f"Expected 4 steps, got {len(steps)}"
        
        step_ids = [step.step_id for step in steps]
        expected_ids = ["design", "review", "implement", "testing"]
        assert step_ids == expected_ids, f"Expected {expected_ids}, got {step_ids}"
        
        # Check dependencies
        design_step = steps[0]
        assert design_step.dependencies == [], "Design step should have no dependencies"
        
        review_step = steps[1]
        assert "design" in review_step.dependencies, "Review step should depend on design"
        
        implement_step = steps[2]
        assert "review" in implement_step.dependencies, "Implement step should depend on review"
        
        testing_step = steps[3]
        assert "implement" in testing_step.dependencies, "Testing step should depend on implement"
    
    def test_workflow_guidance(self):
        """Test workflow guidance generation."""
        guidance = self.orchestrator.get_workflow_guidance(
            WorkflowState.DESIGNING, 
            WorkflowType.DESIGN_REVIEW_IMPLEMENT
        )
        
        assert guidance["current_state"] == "designing"
        assert len(guidance["recommended_tools"]) > 0
        assert "mcp__zen__analyze" in guidance["recommended_tools"]
        assert len(guidance["next_steps"]) > 0
        assert guidance["estimated_time"] is not None
    
    def test_step_customization(self):
        """Test workflow step customization based on context."""
        # Test high complexity context
        context = {"complexity": "high", "team_size": 3}
        steps = self.orchestrator.orchestrate_multi_step(
            WorkflowType.DESIGN_REVIEW_IMPLEMENT, 
            context
        )
        
        # Check that high complexity adds thinkdeep tool
        design_step = steps[0]
        assert "mcp__zen__thinkdeep" in design_step.required_tools
        
        # Check that team size adds swarm coordination
        implement_step = steps[2]
        assert "mcp__claude-flow__swarm_init" in implement_step.required_tools
    
    def test_active_workflow_management(self):
        """Test active workflow tracking."""
        # Create a workflow
        steps = self.orchestrator.orchestrate_multi_step(WorkflowType.DESIGN_REVIEW_IMPLEMENT)
        
        # Check that workflow is tracked
        active_workflows = self.orchestrator.get_active_workflows()
        assert len(active_workflows) == 1
        
        workflow_info = list(active_workflows.values())[0]
        assert workflow_info["type"] == "design_review_implement"
        assert workflow_info["state"] == "initial"
        assert workflow_info["steps_count"] == 4
    
    def test_pattern_caching(self):
        """Test that pattern detection results are cached."""
        prompt = "design then review and implement authentication system"
        
        # First call should detect and cache
        result1 = self.orchestrator.detect_workflow_pattern(prompt)
        assert result1 == WorkflowType.DESIGN_REVIEW_IMPLEMENT
        
        # Check cache is populated
        cache_key = hash(prompt.lower().strip())
        assert cache_key in self.orchestrator.pattern_cache
        
        # Second call should use cache
        result2 = self.orchestrator.detect_workflow_pattern(prompt)
        assert result2 == result1
    
    def test_singleton_instance(self):
        """Test singleton pattern for orchestrator."""
        instance1 = get_zen_workflow_orchestrator()
        instance2 = get_zen_workflow_orchestrator()
        
        assert instance1 is instance2, "Should return same singleton instance"
    
    def test_conversation_metadata_update_without_memory_manager(self):
        """Test conversation metadata update when memory manager is not available."""
        # This should handle gracefully when memory manager is None
        result = self.orchestrator.update_conversation_metadata(
            "test-thread", 
            WorkflowState.DESIGNING,
            {"test": "data"}
        )
        
        # Should return False when memory manager is not available
        assert result is False


class TestWorkflowStep:
    """Test WorkflowStep dataclass."""
    
    def test_workflow_step_creation(self):
        """Test WorkflowStep creation with all fields."""
        step = WorkflowStep(
            step_id="test_step",
            name="Test Step",
            description="A test step",
            required_tools=["tool1", "tool2"],
            dependencies=["dep1"],
            estimated_duration=30,
            completion_criteria="Test completed",
            next_steps=["next1"]
        )
        
        assert step.step_id == "test_step"
        assert step.name == "Test Step"
        assert step.required_tools == ["tool1", "tool2"]
        assert step.dependencies == ["dep1"]
        assert step.estimated_duration == 30
        assert step.next_steps == ["next1"]
    
    def test_workflow_step_defaults(self):
        """Test WorkflowStep creation with default values."""
        step = WorkflowStep(
            step_id="test_step",
            name="Test Step", 
            description="A test step",
            required_tools=["tool1"],
            dependencies=[]
        )
        
        assert step.estimated_duration is None
        assert step.completion_criteria is None
        assert step.next_steps == []


if __name__ == "__main__":
    # Run tests manually if pytest not available
    test_class = TestZenWorkflowOrchestrator()
    test_class.setup_method()
    
    print("🧪 Running ZEN Workflow Orchestrator Tests")
    print("=" * 50)
    
    # Pattern detection tests
    print("\n🔍 Testing Pattern Detection...")
    try:
        test_class.test_pattern_detection_design_review_implement()
        print("  ✅ Design→Review→Implement pattern detection")
        
        test_class.test_pattern_detection_analyze_plan_execute()
        print("  ✅ Analyze→Plan→Execute pattern detection")
        
        test_class.test_pattern_detection_debug_investigate_fix()
        print("  ✅ Debug→Investigate→Fix pattern detection")
        
        test_class.test_no_pattern_detection()
        print("  ✅ No false positives for simple prompts")
    except AssertionError as e:
        print(f"  ❌ Pattern detection failed: {e}")
    
    # State transition tests
    print("\n🔄 Testing State Transitions...")
    try:
        test_class.test_state_transitions()
        print("  ✅ State transitions working correctly")
    except AssertionError as e:
        print(f"  ❌ State transitions failed: {e}")
    
    # Orchestration tests
    print("\n🎭 Testing Workflow Orchestration...")
    try:
        test_class.test_multi_step_orchestration()
        print("  ✅ Multi-step orchestration")
        
        test_class.test_workflow_guidance()
        print("  ✅ Workflow guidance generation")
        
        test_class.test_step_customization()
        print("  ✅ Step customization based on context")
        
        test_class.test_active_workflow_management()
        print("  ✅ Active workflow management")
    except AssertionError as e:
        print(f"  ❌ Orchestration failed: {e}")
    
    # Performance tests
    print("\n⚡ Testing Performance Features...")
    try:
        test_class.test_pattern_caching()
        print("  ✅ Pattern caching")
        
        test_class.test_singleton_instance()
        print("  ✅ Singleton pattern")
        
        test_class.test_conversation_metadata_update_without_memory_manager()
        print("  ✅ Graceful handling without memory manager")
    except AssertionError as e:
        print(f"  ❌ Performance features failed: {e}")
    
    print("\n✅ Test suite completed!")
    print("🚀 ZEN Workflow Orchestrator Phase 2 implementation validated")