#!/usr/bin/env python3
"""Debug script to check pattern detection."""

from zen_workflow_orchestrator import ZenWorkflowOrchestrator, WorkflowType, WorkflowState

orchestrator = ZenWorkflowOrchestrator()

# Test pattern detection with debug info
test_prompts = [
    "design then review and implement authentication",
    "let's plan then build this feature", 
    "let's start with design"
]

print("🔍 Pattern Detection Debug")
print("=" * 40)

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    
    # Test each workflow type manually
    for workflow_type, pattern_info in orchestrator.WORKFLOW_PATTERNS.items():
        score = orchestrator._calculate_pattern_score(prompt.lower(), pattern_info["keywords"])
        print(f"  {workflow_type.value}: {score:.3f}")
        
        if score > 0.5:
            print(f"    → Would detect: {workflow_type}")
    
    detected = orchestrator.detect_workflow_pattern(prompt)
    print(f"  Final detection: {detected.value if detected else 'None'}")

print("\n🔄 State Transition Debug")
print("=" * 40)

transition_prompt = "let's start with design"
current_state = WorkflowState.INITIAL

print(f"Prompt: '{transition_prompt}'")
print(f"Current state: {current_state.value}")

# Check each transition
for transition in orchestrator.STATE_TRANSITIONS:
    if transition.from_state == current_state:
        score = orchestrator._score_transition(transition_prompt.lower(), transition)
        print(f"  Transition to {transition.to_state.value}: {score:.3f}")
        print(f"    Trigger patterns: {transition.trigger_patterns}")
        print(f"    Confidence threshold: {transition.confidence_threshold}")

detected_next = orchestrator.detect_transition(current_state, transition_prompt)
print(f"Final transition: {detected_next.value if detected_next else 'None'}")