#!/usr/bin/env python3
"""
ZEN-Graphiti Bridge Hook
Captures outputs from ZEN tools and stores them in Graphiti memory.
"""

import json
import sys
import os
import re
from datetime import datetime

def extract_zen_tool_info(tool_input):
    """Extract ZEN tool usage information"""
    tool_patterns = {
        'chat': r'mcp__zen__chat',
        'thinkdeep': r'mcp__zen__thinkdeep',
        'planner': r'mcp__zen__planner',
        'consensus': r'mcp__zen__consensus',
        'codereview': r'mcp__zen__codereview',
        'debug': r'mcp__zen__debug',
        'analyze': r'mcp__zen__analyze',
        'refactor': r'mcp__zen__refactor',
        'precommit': r'mcp__zen__precommit',
        'testgen': r'mcp__zen__testgen',
        'secaudit': r'mcp__zen__secaudit',
        'docgen': r'mcp__zen__docgen'
    }
    
    tool_used = None
    for tool, pattern in tool_patterns.items():
        if pattern in str(tool_input):
            tool_used = tool
            break
    
    return tool_used

def create_zen_memory_episode(tool_name, tool_input, tool_output):
    """Create a Graphiti episode from ZEN tool usage"""
    episode_name = f"ZEN {tool_name.title()} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Structure the data for better entity extraction
    episode_data = {
        "type": "zen_tool_usage",
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "input": tool_input,
        "output": tool_output,
        "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
        "project_path": os.environ.get("PWD", "unknown")
    }
    
    # Add tool-specific metadata
    if tool_name == "codereview":
        episode_data["review_type"] = "code_quality"
        episode_data["findings"] = extract_findings(tool_output)
    elif tool_name == "debug":
        episode_data["debug_type"] = "root_cause_analysis"
        episode_data["solution"] = extract_solution(tool_output)
    elif tool_name == "planner":
        episode_data["plan_type"] = "development_roadmap"
        episode_data["steps"] = extract_plan_steps(tool_output)
    elif tool_name == "consensus":
        episode_data["consensus_type"] = "multi_model_decision"
        episode_data["perspectives"] = extract_perspectives(tool_output)
    
    return {
        "name": episode_name,
        "episode_body": json.dumps(episode_data),
        "source": "json",
        "source_description": f"ZEN {tool_name} tool output",
        "group_id": os.environ.get("GRAPHITI_GROUP_ID", "claude-code-default")
    }

def extract_findings(output):
    """Extract findings from code review output"""
    # Simple extraction - can be made more sophisticated
    findings = []
    if isinstance(output, dict) and "findings" in output:
        findings = output["findings"]
    elif isinstance(output, str):
        # Look for common patterns in review output
        lines = output.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['issue', 'problem', 'suggestion', 'improvement']):
                findings.append(line.strip())
    return findings[:5]  # Limit to top 5 findings

def extract_solution(output):
    """Extract solution from debug output"""
    if isinstance(output, dict) and "solution" in output:
        return output["solution"]
    elif isinstance(output, str):
        # Look for solution indicators
        if "solution:" in output.lower():
            return output.split("solution:", 1)[1].strip()[:500]
    return "Solution captured in full output"

def extract_plan_steps(output):
    """Extract steps from planner output"""
    steps = []
    if isinstance(output, dict) and "steps" in output:
        steps = output["steps"]
    elif isinstance(output, str):
        # Look for numbered steps
        lines = output.split('\n')
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                steps.append(line.strip())
    return steps[:10]  # Limit to top 10 steps

def extract_perspectives(output):
    """Extract different model perspectives from consensus"""
    perspectives = {}
    if isinstance(output, dict) and "models" in output:
        perspectives = output["models"]
    elif isinstance(output, str):
        # Look for model names
        for model in ['gemini', 'o3', 'flash', 'pro', 'claude']:
            if model in output.lower():
                # Simple extraction of text around model name
                perspectives[model] = f"Perspective from {model} captured"
    return perspectives

def main():
    # Read hook input
    input_data = sys.stdin.read() if not sys.stdin.isatty() else ""
    
    try:
        data = json.loads(input_data) if input_data else {}
    except:
        data = {}
    
    # Check if this is a ZEN tool usage
    tool_name = data.get("tool_name")
    if tool_name and "zen" in tool_name:
        tool_type = extract_zen_tool_info(tool_name)
        if tool_type:
            # Create memory episode for ZEN tool usage
            tool_input = data.get("tool_input", {})
            tool_output = data.get("tool_output", {})
            
            episode = create_zen_memory_episode(tool_type, tool_input, tool_output)
            
            # Add context about memory storage
            additional_context = f"""
🧠 ZEN Tool Output Captured in Memory

The {tool_type} analysis has been stored in your knowledge graph.

📊 Memory Details:
- Tool: ZEN {tool_type}
- Episode: "{episode['name']}"
- Group: {episode['group_id']}

🔍 Future Queries:
- "Show me previous {tool_type} results"
- "Find similar {tool_type} analyses"
- "What did {tool_type} recommend for this type of issue?"

🔗 Cross-Tool Intelligence:
Your ZEN tool outputs are interconnected in the knowledge graph, allowing you to:
- Track how code evolves after reviews
- Find patterns in debug solutions
- See how plans were implemented
- Review consensus decisions over time
"""
            
            response = {
                "additionalContext": additional_context,
                "metadata": {
                    "source": "zen-graphiti-bridge",
                    "tool": tool_type,
                    "memory_stored": True
                },
                "graphiti_episode": episode
            }
            
            print(json.dumps(response, indent=2))
            return
    
    # If not a ZEN tool, pass through
    print(json.dumps({"metadata": {"source": "zen-graphiti-bridge", "passthrough": True}}))

if __name__ == "__main__":
    main()