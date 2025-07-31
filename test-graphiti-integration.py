#!/usr/bin/env python3
"""
Test script to verify Graphiti integration with Claude Code hooks
"""

import json
import subprocess
import os

def test_hook(hook_name, input_data):
    """Test a specific hook with given input"""
    print(f"\n🧪 Testing {hook_name}...")
    
    hook_path = f"/home/blake/flowed/.claude/hooks/{hook_name}"
    
    # Run the hook
    process = subprocess.Popen(
        ["python3", hook_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send input data
    stdout, stderr = process.communicate(input=json.dumps(input_data))
    
    if stderr:
        print(f"❌ Error: {stderr}")
    else:
        print(f"✅ Success!")
        output = json.loads(stdout)
        print(f"   Context: {output.get('additionalContext', '')[:100]}...")
        print(f"   Metadata: {output.get('metadata', {})}")
    
    return stdout

# Test 1: Session Start Hook
print("=" * 60)
print("🚀 Testing Session Start Hook")
test_hook("session-start-zen.py", {})

# Test 2: Prompt Submit Zen Hook
print("\n" + "=" * 60)
print("📝 Testing Prompt Submit Zen Hook")
test_hook("prompt-submit-zen.py", {
    "prompt": "I need to debug why my authentication is failing"
})

# Test 3: Graphiti Integration Hook
print("\n" + "=" * 60)
print("🧠 Testing Graphiti Integration Hook")
test_hook("graphiti-integration.py", {
    "prompt": "How do I implement JWT authentication in Python?"
})

# Test 4: Zen-Graphiti Bridge Hook
print("\n" + "=" * 60)
print("🌉 Testing Zen-Graphiti Bridge Hook")
test_hook("zen-graphiti-bridge.py", {
    "tool_name": "mcp__zen__debug",
    "tool_input": {"prompt": "Debug authentication issue"},
    "tool_output": {"solution": "Check token expiry", "findings": ["Token expired", "Invalid signature"]}
})

print("\n" + "=" * 60)
print("✨ Integration test complete!")
print("\nIf all tests passed, your hooks are working correctly!")
print("Remember to update the OpenAI API key in graphiti/mcp_server/.env")