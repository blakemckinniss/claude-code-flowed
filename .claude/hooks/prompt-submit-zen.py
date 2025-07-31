#!/usr/bin/env python3
"""
Zen MCP UserPromptSubmit Hook
Injects contextual Zen MCP usage reminders on every user prompt.
"""

import json
import sys
import re

def analyze_prompt(prompt):
    """Analyze the user's prompt to provide contextual Zen suggestions"""
    prompt_lower = prompt.lower()
    
    suggestions = []
    
    # Check for specific patterns and suggest appropriate Zen tools
    patterns = {
        # Debugging patterns
        r'\b(debug|bug|error|issue|problem|broken|fix|crash|fail)\b': 
            "💡 Consider using Zen's 'debug' tool for systematic investigation: 'Use zen debug to find why [issue]'",
        
        # Code review patterns
        r'\b(review|check|audit|quality|security|vulnerable|secure)\b':
            "🔍 Consider using Zen's 'codereview' or 'secaudit' tools: 'Perform a codereview with gemini pro'",
        
        # Planning patterns
        r'\b(plan|design|architect|structure|organize|break\s*down)\b':
            "📋 Consider using Zen's 'planner' tool: 'Use zen planner to break down this project'",
        
        # Analysis patterns
        r'\b(understand|analyze|explain|how\s*does|what\s*is)\b':
            "🧐 Consider using Zen's 'analyze' tool: 'Use zen to analyze this code'",
        
        # Testing patterns
        r'\b(test|tests|testing|coverage|unit\s*test|integration)\b':
            "🧪 Consider using Zen's 'testgen' tool: 'Use zen to generate tests for [component]'",
        
        # Refactoring patterns
        r'\b(refactor|improve|clean|optimize|decompose|split)\b':
            "♻️ Consider using Zen's 'refactor' tool: 'Use zen to refactor this class'",
        
        # Documentation patterns
        r'\b(document|documentation|docs|comment|explain\s*code)\b':
            "📝 Consider using Zen's 'docgen' tool: 'Use docgen to document this module'",
        
        # Decision patterns
        r'\b(should|decide|choice|option|alternative|vs|versus)\b':
            "🤔 Consider using Zen's 'consensus' tool: 'Get consensus on whether to [option A vs B]'",
        
        # Thinking patterns
        r'\b(think|consider|explore|brainstorm|idea)\b':
            "💭 Consider using Zen's 'chat' or 'thinkdeep' tools: 'Think deeper about this with zen'"
    }
    
    for pattern, suggestion in patterns.items():
        if re.search(pattern, prompt_lower):
            suggestions.append(suggestion)
    
    return suggestions

def get_contextual_reminder(prompt):
    """Generate a contextual reminder based on the user's prompt"""
    suggestions = analyze_prompt(prompt)
    
    if suggestions:
        reminder = "🧘 ZEN MCP SUGGESTIONS:\n\n"
        reminder += "\n".join(suggestions[:3])  # Limit to top 3 suggestions
        reminder += "\n\nZen MCP provides AI-powered collaboration with multiple models working together to enhance Claude's capabilities."
    else:
        # Generic reminder if no specific patterns match
        reminder = """🧘 ZEN MCP READY: Remember you can use Zen tools for enhanced AI collaboration:
• 'chat' - Brainstorm with AI partners
• 'thinkdeep' - Extended reasoning
• 'debug' - Systematic debugging
• 'codereview' - Professional code analysis
• 'planner' - Break down complex tasks
• 'consensus' - Get multiple AI perspectives"""
    
    return reminder

def main():
    # Read the prompt from stdin
    input_data = sys.stdin.read() if not sys.stdin.isatty() else ""
    
    try:
        data = json.loads(input_data) if input_data else {}
        prompt = data.get("prompt", "")
    except:
        prompt = input_data
    
    # Generate contextual reminder
    reminder = get_contextual_reminder(prompt)
    
    # Prepare the response
    response = {
        "additionalContext": reminder,
        "metadata": {
            "source": "zen-mcp-prompt-hook",
            "analyzed": bool(prompt),
            "prompt_length": len(prompt)
        }
    }
    
    # Output the response
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()