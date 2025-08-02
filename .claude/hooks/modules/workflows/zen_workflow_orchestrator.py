#!/usr/bin/env python3
"""ZEN Workflow Orchestrator - Phase 2 Integration.

This module implements workflow state management and orchestration for ZEN's 
collaborative investigation patterns. It detects multi-step development patterns 
(Design→Review→Implement→Test) and manages state transitions through ConversationThread
metadata integration.

Key Features:
- State machine pattern with validated transitions
- Pattern detection for collaborative workflows
- Integration with ConversationThread metadata system
- Extensible workflow pattern system
- Multi-step orchestration with dependency tracking
"""

import re
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

# Import ConversationThread from existing memory system
try:
    from ..memory.zen_memory_integration import ConversationThread, get_zen_memory_manager
except ImportError:
    # Fallback for testing or standalone usage
    ConversationThread = None
    get_zen_memory_manager = None


class WorkflowState(Enum):
    """Core workflow states for ZEN orchestration."""
    INITIAL = "initial"
    DESIGNING = "designing"
    REVIEWING = "reviewing"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowType(Enum):
    """Types of workflows that can be orchestrated."""
    DESIGN_REVIEW_IMPLEMENT = "design_review_implement"
    ANALYZE_PLAN_EXECUTE = "analyze_plan_execute"
    DEBUG_INVESTIGATE_FIX = "debug_investigate_fix"
    RESEARCH_DESIGN_BUILD = "research_design_build"
    REVIEW_REFACTOR_TEST = "review_refactor_test"
    CUSTOM = "custom"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    name: str
    description: str
    required_tools: List[str]
    dependencies: List[str]
    estimated_duration: Optional[int] = None  # minutes
    completion_criteria: Optional[str] = None
    next_steps: List[str] = None
    
    def __post_init__(self):
        if self.next_steps is None:
            self.next_steps = []


@dataclass
class WorkflowTransition:
    """Represents a state transition with validation."""
    from_state: WorkflowState
    to_state: WorkflowState
    trigger_patterns: List[str]
    required_conditions: List[str]
    validation_rules: List[str]
    confidence_threshold: float = 0.7


@dataclass
class WorkflowOrchestration:
    """Complete workflow orchestration plan."""
    workflow_id: str
    workflow_type: WorkflowType
    current_state: WorkflowState
    steps: List[WorkflowStep]
    transitions: List[WorkflowTransition]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    completion_percentage: float = 0.0
    
    def __post_init__(self):
        if not self.workflow_id:
            self.workflow_id = str(uuid.uuid4())


class ZenWorkflowOrchestrator:
    """Advanced workflow orchestrator for ZEN collaborative patterns."""
    
    # Workflow pattern definitions
    WORKFLOW_PATTERNS = {
        WorkflowType.DESIGN_REVIEW_IMPLEMENT: {
            "keywords": [
                "design then review", "design and review", "plan then implement",
                "design before implementing", "review before coding", "plan and execute",
                "design review implement", "architectural review", "design phase",
                "review design", "implement after review"
            ],
            "steps": [
                WorkflowStep(
                    step_id="design",
                    name="Design Phase",
                    description="Analyze requirements and create architectural design",
                    required_tools=["mcp__zen__analyze", "mcp__zen__planner"],
                    dependencies=[],
                    estimated_duration=30,
                    completion_criteria="Design documents created and validated",
                    next_steps=["review"]
                ),
                WorkflowStep(
                    step_id="review",
                    name="Review Phase", 
                    description="Review design decisions and validate approach",
                    required_tools=["mcp__zen__consensus", "mcp__zen__chat"],
                    dependencies=["design"],
                    estimated_duration=20,
                    completion_criteria="Design approved by review process",
                    next_steps=["implement"]
                ),
                WorkflowStep(
                    step_id="implement",
                    name="Implementation Phase",
                    description="Implement the designed solution",
                    required_tools=["mcp__claude-flow__swarm_init", "mcp__claude-flow__agent_spawn"],
                    dependencies=["review"],
                    estimated_duration=60,
                    completion_criteria="Implementation completed and tested",
                    next_steps=["testing"]
                ),
                WorkflowStep(
                    step_id="testing",
                    name="Testing Phase",
                    description="Validate implementation through testing",
                    required_tools=["mcp__zen__testgen", "mcp__zen__debug"],
                    dependencies=["implement"],
                    estimated_duration=25,
                    completion_criteria="All tests pass and quality gates met",
                    next_steps=[]
                )
            ]
        },
        
        WorkflowType.ANALYZE_PLAN_EXECUTE: {
            "keywords": [
                "analyze then plan", "investigate and execute", "study then build",
                "analyze plan execute", "research then implement", "understand then code",
                "investigation workflow", "systematic approach"
            ],
            "steps": [
                WorkflowStep(
                    step_id="analyze",
                    name="Analysis Phase",
                    description="Deep analysis of the problem domain",
                    required_tools=["mcp__zen__thinkdeep", "mcp__zen__analyze"],
                    dependencies=[],
                    estimated_duration=40,
                    completion_criteria="Comprehensive analysis completed",
                    next_steps=["plan"]
                ),
                WorkflowStep(
                    step_id="plan",
                    name="Planning Phase",
                    description="Create detailed execution plan",
                    required_tools=["mcp__zen__planner", "mcp__zen__consensus"],
                    dependencies=["analyze"],
                    estimated_duration=25,
                    completion_criteria="Detailed plan approved",
                    next_steps=["execute"]
                ),
                WorkflowStep(
                    step_id="execute",
                    name="Execution Phase",
                    description="Execute the planned solution",
                    required_tools=["mcp__claude-flow__task_orchestrate", "mcp__claude-flow__swarm_init"],
                    dependencies=["plan"],
                    estimated_duration=90,
                    completion_criteria="Solution successfully executed",
                    next_steps=[]
                )
            ]
        },
        
        WorkflowType.DEBUG_INVESTIGATE_FIX: {
            "keywords": [
                "debug then fix", "investigate and resolve", "troubleshoot systematically",
                "debug investigate fix", "root cause analysis", "systematic debugging",
                "investigate then implement fix"
            ],
            "steps": [
                WorkflowStep(
                    step_id="debug",
                    name="Debug Phase",
                    description="Identify and isolate the issue",
                    required_tools=["mcp__zen__debug", "mcp__zen__thinkdeep"],
                    dependencies=[],
                    estimated_duration=35,
                    completion_criteria="Root cause identified",
                    next_steps=["investigate"]
                ),
                WorkflowStep(
                    step_id="investigate",
                    name="Investigation Phase",
                    description="Deep investigation of root cause",
                    required_tools=["mcp__zen__analyze", "mcp__zen__tracer"],
                    dependencies=["debug"],
                    estimated_duration=30,
                    completion_criteria="Investigation completed with solution path",
                    next_steps=["fix"]
                ),
                WorkflowStep(
                    step_id="fix",
                    name="Fix Phase",
                    description="Implement and validate the fix",
                    required_tools=["mcp__claude-flow__agent_spawn", "mcp__zen__testgen"],
                    dependencies=["investigate"],
                    estimated_duration=45,
                    completion_criteria="Fix implemented and validated",
                    next_steps=[]
                )
            ]
        }
    }
    
    # State transition rules
    STATE_TRANSITIONS = [
        WorkflowTransition(
            from_state=WorkflowState.INITIAL,
            to_state=WorkflowState.DESIGNING,
            trigger_patterns=["design", "plan", "architect", "structure"],
            required_conditions=["workflow_type_identified"],
            validation_rules=["has_design_requirements"],
            confidence_threshold=0.4
        ),
        WorkflowTransition(
            from_state=WorkflowState.DESIGNING,
            to_state=WorkflowState.REVIEWING,
            trigger_patterns=["review", "validate", "check", "approve"],
            required_conditions=["design_artifacts_created"],
            validation_rules=["design_completeness_check"],
            confidence_threshold=0.4
        ),
        WorkflowTransition(
            from_state=WorkflowState.REVIEWING,
            to_state=WorkflowState.IMPLEMENTING,
            trigger_patterns=["implement", "build", "code", "execute"],
            required_conditions=["review_completed", "design_approved"],
            validation_rules=["implementation_readiness"],
            confidence_threshold=0.4
        ),
        WorkflowTransition(
            from_state=WorkflowState.IMPLEMENTING,
            to_state=WorkflowState.TESTING,
            trigger_patterns=["test", "validate", "verify", "qa"],
            required_conditions=["implementation_completed"],
            validation_rules=["testability_check"],
            confidence_threshold=0.4
        ),
        WorkflowTransition(
            from_state=WorkflowState.TESTING,
            to_state=WorkflowState.COMPLETED,
            trigger_patterns=["complete", "finish", "done", "deploy"],
            required_conditions=["all_tests_passed"],
            validation_rules=["quality_gates_met"],
            confidence_threshold=0.4
        )
    ]
    
    def __init__(self):
        """Initialize the ZEN workflow orchestrator."""
        self.active_workflows: Dict[str, WorkflowOrchestration] = {}
        self.pattern_cache: Dict[str, Tuple[WorkflowType, float]] = {}
        self.memory_manager = get_zen_memory_manager() if get_zen_memory_manager else None
        
    def detect_workflow_pattern(self, prompt: str) -> Optional[WorkflowType]:
        """Detect workflow pattern from user prompt.
        
        Args:
            prompt: User input to analyze for workflow patterns
            
        Returns:
            Detected workflow type or None if no pattern matches
        """
        if not prompt or len(prompt.strip()) < 10:
            return None
        
        # Normalize prompt for analysis
        normalized_prompt = prompt.lower().strip()
        
        # Check cache first
        cache_key = hash(normalized_prompt)
        if cache_key in self.pattern_cache:
            workflow_type, confidence = self.pattern_cache[cache_key]
            if confidence > 0.7:
                return workflow_type
        
        # Analyze for workflow patterns
        best_match = None
        best_score = 0.0
        
        for workflow_type, pattern_info in self.WORKFLOW_PATTERNS.items():
            score = self._calculate_pattern_score(normalized_prompt, pattern_info["keywords"])
            
            if score > best_score and score > 0.3:
                best_match = workflow_type
                best_score = score
        
        # Cache the result
        if best_match and best_score > 0.5:
            self.pattern_cache[cache_key] = (best_match, best_score)
        
        return best_match if best_score > 0.4 else None
    
    def _calculate_pattern_score(self, prompt: str, keywords: List[str]) -> float:
        """Calculate how well a prompt matches workflow keywords."""
        if not keywords:
            return 0.0
        
        matches = 0
        keyword_weights = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in prompt:
                # Higher weight for exact matches
                weight = 1.0
                matches += 1
            elif any(word in prompt for word in keyword_lower.split()):
                # Lower weight for partial matches
                weight = 0.6
                matches += 0.6
            else:
                # Check for semantic matches
                semantic_score = self._calculate_semantic_match(prompt, keyword_lower)
                if semantic_score > 0.3:
                    weight = semantic_score
                    matches += semantic_score
                else:
                    weight = 0.0
            
            keyword_weights.append(weight)
        
        # Calculate weighted score
        if not keyword_weights:
            return 0.0
        
        # Bonus for multiple keyword matches
        match_bonus = min(matches / len(keywords), 1.0) * 0.2
        base_score = sum(keyword_weights) / len(keyword_weights)
        
        return min(base_score + match_bonus, 1.0)
    
    def _calculate_semantic_match(self, prompt: str, keyword: str) -> float:
        """Calculate semantic similarity for better pattern matching."""
        # Simple semantic matching based on related words
        semantic_mappings = {
            "design": ["plan", "architect", "structure", "blueprint", "layout"],
            "review": ["check", "validate", "assess", "evaluate", "examine"],
            "implement": ["build", "code", "create", "develop", "execute"],
            "analyze": ["study", "examine", "investigate", "research", "understand"],
            "debug": ["troubleshoot", "fix", "diagnose", "solve", "repair"],
            "test": ["verify", "validate", "check", "qa", "quality"]
        }
        
        # Check if any word in the keyword has semantic matches in prompt
        for word in keyword.split():
            if word in semantic_mappings:
                related_words = semantic_mappings[word]
                matches = sum(1 for related in related_words if related in prompt)
                if matches > 0:
                    return min(matches / len(related_words), 0.8)
        
        return 0.0
    
    def detect_transition(self, current_state: WorkflowState, user_prompt: str) -> Optional[WorkflowState]:
        """Detect state transition based on current state and user prompt.
        
        Args:
            current_state: Current workflow state
            user_prompt: User input indicating potential transition
            
        Returns:
            Next workflow state or None if no transition detected
        """
        if not user_prompt:
            return None
        
        normalized_prompt = user_prompt.lower().strip()
        
        # Find applicable transitions from current state
        applicable_transitions = [
            t for t in self.STATE_TRANSITIONS 
            if t.from_state == current_state
        ]
        
        if not applicable_transitions:
            return None
        
        # Score each possible transition
        best_transition = None
        best_score = 0.0
        
        for transition in applicable_transitions:
            score = self._score_transition(normalized_prompt, transition)
            if score > best_score and score > transition.confidence_threshold:
                best_transition = transition
                best_score = score
        
        return best_transition.to_state if best_transition else None
    
    def _score_transition(self, prompt: str, transition: WorkflowTransition) -> float:
        """Score how well a prompt matches a transition."""
        # Check for direct pattern matches
        pattern_matches = 0
        for pattern in transition.trigger_patterns:
            if pattern in prompt:
                pattern_matches += 1
            else:
                # Check for semantic matches
                semantic_score = self._calculate_semantic_match(prompt, pattern)
                if semantic_score > 0.3:
                    pattern_matches += semantic_score
        
        if pattern_matches == 0:
            return 0.0
        
        # Base score from pattern matches
        base_score = min(pattern_matches / len(transition.trigger_patterns), 1.0)
        
        # TODO: Add condition validation when integrated with memory system
        condition_score = 0.8  # Assume conditions are met for now
        
        return (base_score * 0.7) + (condition_score * 0.3)
    
    def orchestrate_multi_step(self, workflow_type: WorkflowType, 
                             context: Optional[Dict[str, Any]] = None) -> List[WorkflowStep]:
        """Create orchestration plan for a multi-step workflow.
        
        Args:
            workflow_type: Type of workflow to orchestrate
            context: Additional context for workflow customization
            
        Returns:
            List of workflow steps in execution order
        """
        if workflow_type not in self.WORKFLOW_PATTERNS:
            return []
        
        pattern_info = self.WORKFLOW_PATTERNS[workflow_type]
        base_steps = pattern_info["steps"].copy()
        
        # Customize steps based on context
        if context:
            base_steps = self._customize_steps(base_steps, context)
        
        # Create workflow orchestration
        orchestration = WorkflowOrchestration(
            workflow_id=str(uuid.uuid4()),
            workflow_type=workflow_type,
            current_state=WorkflowState.INITIAL,
            steps=base_steps,
            transitions=self.STATE_TRANSITIONS.copy(),
            metadata=context or {},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store active workflow
        self.active_workflows[orchestration.workflow_id] = orchestration
        
        return base_steps
    
    def _customize_steps(self, steps: List[WorkflowStep], context: Dict[str, Any]) -> List[WorkflowStep]:
        """Customize workflow steps based on context."""
        customized_steps = []
        
        for step in steps:
            customized_step = WorkflowStep(
                step_id=step.step_id,
                name=step.name,
                description=step.description,
                required_tools=step.required_tools.copy(),
                dependencies=step.dependencies.copy(),
                estimated_duration=step.estimated_duration,
                completion_criteria=step.completion_criteria,
                next_steps=step.next_steps.copy()
            )
            
            # Adjust based on context
            if context.get("complexity") == "high":
                customized_step.estimated_duration = int(customized_step.estimated_duration * 1.5) if customized_step.estimated_duration else None
                if "mcp__zen__thinkdeep" not in customized_step.required_tools:
                    customized_step.required_tools.append("mcp__zen__thinkdeep")
            
            if context.get("team_size", 1) > 1:
                if "mcp__claude-flow__swarm_init" not in customized_step.required_tools:
                    customized_step.required_tools.append("mcp__claude-flow__swarm_init")
            
            customized_steps.append(customized_step)
        
        return customized_steps
    
    def update_conversation_metadata(self, thread_id: str, workflow_state: WorkflowState, 
                                   workflow_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update conversation thread metadata with workflow state.
        
        Args:
            thread_id: Conversation thread identifier
            workflow_state: Current workflow state
            workflow_data: Additional workflow data to store
            
        Returns:
            True if update successful, False otherwise
        """
        if not self.memory_manager:
            # Fallback for when memory manager is not available
            return False
        
        try:
            # Get the conversation thread
            if thread_id not in self.memory_manager.conversation_threads:
                # Create new thread if it doesn't exist
                self.memory_manager.create_conversation_thread(thread_id)
            
            thread = self.memory_manager.conversation_threads[thread_id]
            
            # Update workflow metadata
            workflow_metadata = {
                "workflow_state": workflow_state.value,
                "updated_at": datetime.now().isoformat(),
                "state_history": thread.metadata.get("state_history", [])
            }
            
            # Add state to history
            workflow_metadata["state_history"].append({
                "state": workflow_state.value,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 state changes
            workflow_metadata["state_history"] = workflow_metadata["state_history"][-10:]
            
            # Add additional workflow data
            if workflow_data:
                workflow_metadata.update(workflow_data)
            
            # Update thread metadata
            thread.update_metadata("workflow", workflow_metadata)
            
            return True
            
        except Exception as e:
            # Log error if logging is available
            print(f"Error updating conversation metadata: {e}")
            return False
    
    def get_workflow_guidance(self, current_state: WorkflowState, 
                            workflow_type: Optional[WorkflowType] = None) -> Dict[str, Any]:
        """Get guidance for current workflow state.
        
        Args:
            current_state: Current workflow state
            workflow_type: Type of workflow (optional)
            
        Returns:
            Dictionary with guidance information
        """
        guidance = {
            "current_state": current_state.value,
            "recommended_tools": [],
            "next_steps": [],
            "estimated_time": None,
            "completion_criteria": "",
            "tips": []
        }
        
        # Get workflow-specific guidance
        if workflow_type and workflow_type in self.WORKFLOW_PATTERNS:
            pattern_info = self.WORKFLOW_PATTERNS[workflow_type]
            
            # Find current step
            current_step = None
            for step in pattern_info["steps"]:
                if self._state_matches_step(current_state, step.step_id):
                    current_step = step
                    break
            
            if current_step:
                guidance.update({
                    "recommended_tools": current_step.required_tools,
                    "next_steps": [f"Complete {current_step.name}: {current_step.description}"],
                    "estimated_time": current_step.estimated_duration,
                    "completion_criteria": current_step.completion_criteria or "Step completion verified",
                    "tips": self._get_state_specific_tips(current_state)
                })
        
        return guidance
    
    def _state_matches_step(self, state: WorkflowState, step_id: str) -> bool:
        """Check if a workflow state matches a step ID."""
        state_step_mapping = {
            WorkflowState.DESIGNING: ["design", "analyze"],
            WorkflowState.REVIEWING: ["review", "plan"],
            WorkflowState.IMPLEMENTING: ["implement", "execute", "fix"],
            WorkflowState.TESTING: ["testing", "test"]
        }
        
        return step_id in state_step_mapping.get(state, [])
    
    def _get_state_specific_tips(self, state: WorkflowState) -> List[str]:
        """Get tips specific to workflow state."""
        tips_by_state = {
            WorkflowState.INITIAL: [
                "Start by clearly defining the problem or goal",
                "Consider using mcp__zen__analyze for initial assessment"
            ],
            WorkflowState.DESIGNING: [
                "Use mcp__zen__planner for systematic approach",
                "Consider multiple solution approaches",
                "Document architectural decisions"
            ],
            WorkflowState.REVIEWING: [
                "Use mcp__zen__consensus for multi-perspective review",
                "Validate assumptions and edge cases",
                "Ensure design meets all requirements"
            ],
            WorkflowState.IMPLEMENTING: [
                "Initialize swarm coordination with mcp__claude-flow__swarm_init",
                "Break down implementation into manageable tasks",
                "Use appropriate agents for different aspects"
            ],
            WorkflowState.TESTING: [
                "Use mcp__zen__testgen for comprehensive test generation",
                "Test both happy path and edge cases",
                "Validate against original requirements"
            ]
        }
        
        return tips_by_state.get(state, ["Continue with current workflow step"])
    
    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all active workflows."""
        return {
            workflow_id: {
                "type": workflow.workflow_type.value,
                "state": workflow.current_state.value,
                "progress": workflow.completion_percentage,
                "created": workflow.created_at.isoformat(),
                "updated": workflow.updated_at.isoformat(),
                "steps_count": len(workflow.steps)
            }
            for workflow_id, workflow in self.active_workflows.items()
        }
    
    def cleanup_expired_workflows(self, max_age_hours: int = 8) -> int:
        """Clean up workflows older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_workflows = []
        
        for workflow_id, workflow in self.active_workflows.items():
            if workflow.updated_at < cutoff_time:
                expired_workflows.append(workflow_id)
        
        for workflow_id in expired_workflows:
            del self.active_workflows[workflow_id]
        
        return len(expired_workflows)


# Singleton instance for global access
_zen_workflow_orchestrator = None

def get_zen_workflow_orchestrator() -> ZenWorkflowOrchestrator:
    """Get singleton ZEN workflow orchestrator instance."""
    global _zen_workflow_orchestrator
    if _zen_workflow_orchestrator is None:
        _zen_workflow_orchestrator = ZenWorkflowOrchestrator()
    return _zen_workflow_orchestrator


# Test and demonstration functions
async def simulate_design_review_implement_workflow():
    """Simulate a complete Design→Review→Implement workflow."""
    orchestrator = get_zen_workflow_orchestrator()
    
    print("🔄 Simulating Design→Review→Implement Workflow")
    print("=" * 50)
    
    # Step 1: Detect workflow pattern
    user_prompt = "I need to design then review and implement a user authentication system"
    detected_type = orchestrator.detect_workflow_pattern(user_prompt)
    print(f"🎯 Detected workflow: {detected_type.value if detected_type else 'None'}")
    
    if not detected_type:
        print("❌ No workflow pattern detected")
        return
    
    # Step 2: Create orchestration plan
    steps = orchestrator.orchestrate_multi_step(detected_type, {
        "complexity": "high",
        "team_size": 2
    })
    print(f"📋 Created {len(steps)} workflow steps")
    
    # Step 3: Simulate state transitions
    current_state = WorkflowState.INITIAL
    thread_id = "test-thread-" + str(time.time())
    
    state_transitions = [
        ("Let's start with the design phase", WorkflowState.DESIGNING),
        ("Now let's review the design", WorkflowState.REVIEWING),
        ("Design looks good, let's implement it", WorkflowState.IMPLEMENTING),
        ("Implementation done, time to test", WorkflowState.TESTING),
        ("All tests pass, we're complete", WorkflowState.COMPLETED)
    ]
    
    for prompt, expected_state in state_transitions:
        next_state = orchestrator.detect_transition(current_state, prompt)
        print(f"🔄 '{prompt}' → {current_state.value} → {next_state.value if next_state else 'No transition'}")
        
        if next_state:
            # Update conversation metadata
            orchestrator.update_conversation_metadata(thread_id, next_state, {
                "workflow_type": detected_type.value,
                "prompt": prompt
            })
            
            # Get guidance for new state
            guidance = orchestrator.get_workflow_guidance(next_state, detected_type)
            print(f"💡 Guidance: {guidance['next_steps'][0] if guidance['next_steps'] else 'Continue workflow'}")
            
            current_state = next_state
        
        print()
    
    # Step 4: Show final status
    active_workflows = orchestrator.get_active_workflows()
    print(f"✅ Workflow completed. Active workflows: {len(active_workflows)}")
    
    return {
        "detected_type": detected_type.value,
        "steps_count": len(steps),
        "final_state": current_state.value,
        "thread_id": thread_id
    }


if __name__ == "__main__":
    # Run demonstration
    import asyncio
    
    print("🧠 ZEN Workflow Orchestrator - Phase 2")
    print("=" * 40)
    
    # Test pattern detection
    orchestrator = get_zen_workflow_orchestrator()
    
    test_prompts = [
        "Design then review and implement user authentication",
        "Let's analyze this problem, plan a solution, then execute it",
        "I need to debug this issue, investigate the root cause, and fix it",
        "Just write some code quickly"  # Should not match any pattern
    ]
    
    print("\n🔍 Pattern Detection Tests:")
    for prompt in test_prompts:
        detected = orchestrator.detect_workflow_pattern(prompt)
        print(f"  '{prompt}' → {detected.value if detected else 'No pattern'}")
    
    # Run full workflow simulation
    print("\n🚀 Full Workflow Simulation:")
    result = asyncio.run(simulate_design_review_implement_workflow())
    
    print(f"\n📊 Simulation Results:")
    print(f"  • Workflow Type: {result['detected_type']}")
    print(f"  • Steps Created: {result['steps_count']}")
    print(f"  • Final State: {result['final_state']}")
    print(f"  • Thread ID: {result['thread_id'][:8]}...")