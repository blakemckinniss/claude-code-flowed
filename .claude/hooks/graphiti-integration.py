#!/usr/bin/env python3
"""
Graphiti Integration Hook for Claude Code
Automatically captures and stores conversation context in a knowledge graph.
"""

import json
import sys
import os
import subprocess
from datetime import datetime

# Configuration
GRAPHITI_GROUP_ID = os.getenv("GRAPHITI_GROUP_ID", "claude-code-default")

def call_graphiti_mcp(tool_name, params):
    """Call Graphiti MCP server via Claude's MCP integration"""
    # This would be called through Claude's MCP integration
    # For now, we'll prepare the structure
    return {
        "tool": f"mcp__graphiti__{tool_name}",
        "params": params
    }

def extract_code_context(prompt):
    """Extract code-related context from the prompt"""
    context = {
        "has_code": any(marker in prompt for marker in ["```", "def ", "class ", "function", "import"]),
        "mentions_file": ".py" in prompt or ".js" in prompt or ".ts" in prompt,
        "is_question": "?" in prompt,
        "is_debug": any(word in prompt.lower() for word in ["debug", "error", "bug", "fix"]),
        "is_feature": any(word in prompt.lower() for word in ["add", "create", "implement", "build"])
    }
    return context

def create_episode_from_prompt(prompt, context_data):
    """Create a Graphiti episode from user prompt"""
    episode_name = f"User Query - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Determine the type of interaction
    if context_data["is_debug"]:
        episode_name = f"Debug Request - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    elif context_data["is_feature"]:
        episode_name = f"Feature Request - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    elif context_data["is_question"]:
        episode_name = f"Question - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Create structured data for better entity extraction
    episode_data = {
        "type": "user_interaction",
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "context": context_data,
        "session_id": os.getenv("CLAUDE_SESSION_ID", "unknown"),
        "project_path": os.getcwd()
    }
    
    return {
        "name": episode_name,
        "episode_body": json.dumps(episode_data),
        "source": "json",
        "source_description": "Claude Code user interaction",
        "group_id": GRAPHITI_GROUP_ID
    }

def suggest_memory_queries(context_data):
    """Suggest relevant memory queries based on context"""
    suggestions = []
    
    if context_data["is_debug"]:
        suggestions.append("Search for similar debug issues: 'previous errors in this module'")
        suggestions.append("Find related fixes: 'solutions for similar problems'")
    
    if context_data["is_feature"]:
        suggestions.append("Check existing patterns: 'similar features implemented'")
        suggestions.append("Find requirements: 'project requirements and constraints'")
    
    if context_data["mentions_file"]:
        suggestions.append("Get file history: 'recent changes to this file'")
        suggestions.append("Find dependencies: 'files that depend on this module'")
    
    return suggestions

def main():
    # Read input from stdin
    input_data = sys.stdin.read() if not sys.stdin.isatty() else ""
    
    try:
        data = json.loads(input_data) if input_data else {}
        prompt = data.get("prompt", "")
    except:
        prompt = input_data
    
    # Extract context from the prompt
    context_data = extract_code_context(prompt)
    
    # Create episode data
    episode = create_episode_from_prompt(prompt, context_data)
    
    # Get memory query suggestions
    suggestions = suggest_memory_queries(context_data)
    
    # Build the additional context
    additional_context = f"""
🧠 GRAPHITI MEMORY INTEGRATION

This conversation is being recorded in your knowledge graph memory (group: {GRAPHITI_GROUP_ID}).

📝 Memory Action:
Will store this interaction as: "{episode['name']}"

🔍 Suggested Memory Queries:
{chr(10).join(f'• {s}' for s in suggestions) if suggestions else '• No specific suggestions for this context'}

💡 Memory Commands Available:
• To search your memory: "Search my memory for [query]"
• To find related facts: "What do I know about [topic]"
• To see recent interactions: "Show my recent conversations"
• To connect concepts: "How is [X] related to [Y]"

🎯 Integration with ZEN:
When using ZEN tools, Graphiti will automatically capture:
- Code analysis results from 'analyze' and 'codereview'
- Debug findings from 'debug' tool
- Planning decisions from 'planner'
- Consensus outcomes from multi-model discussions

This creates a rich, interconnected memory of your development process.
"""
    
    # Prepare response
    response = {
        "additionalContext": additional_context,
        "metadata": {
            "source": "graphiti-integration-hook",
            "episode_id": episode.get("uuid", "pending"),
            "group_id": GRAPHITI_GROUP_ID,
            "context_extracted": context_data
        },
        # Include the MCP call structure for Claude to use
        "suggested_mcp_call": call_graphiti_mcp("add_memory", episode)
    }
    
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()