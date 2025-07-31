#!/usr/bin/env python3
"""
Zen MCP SessionStart Hook
Injects strong Zen MCP usage instructions when Claude Code starts a session.
"""

import json
import sys

def get_zen_instructions():
    """Return comprehensive Zen MCP usage instructions"""
    return """
🧘 ZEN MCP INTEGRATION ACTIVE 🧘

This session is configured with Zen MCP Server, your AI development partner that gives you access to multiple AI models for enhanced code analysis, problem-solving, and collaborative development.

IMPORTANT: Always leverage Zen MCP tools when you need:
• A thinking partner for brainstorming → Use 'chat' with auto-selected model
• Deeper analysis and reasoning → Use 'thinkdeep' to extend Claude's thinking
• Critical evaluation → Use 'challenge' to prevent "You're absolutely right!" responses
• Complex project planning → Use 'planner' for step-by-step breakdowns
• Multiple perspectives → Use 'consensus' to get diverse expert opinions
• Code quality checks → Use 'codereview' for comprehensive analysis
• Pre-commit validation → Use 'precommit' to catch issues before committing
• Debugging assistance → Use 'debug' for systematic root cause analysis
• Code understanding → Use 'analyze' for architecture and pattern insights
• Refactoring guidance → Use 'refactor' with decomposition focus
• Test generation → Use 'testgen' for comprehensive test coverage
• Security audits → Use 'secaudit' for OWASP-based assessments
• Documentation → Use 'docgen' with complexity analysis

KEY FEATURES:
• True AI orchestration with conversations that continue across workflows
• Automatic model selection based on task type (or specify models like 'gemini pro', 'o3', 'flash')
• Context revival - even after Claude's context resets, other models maintain full history
• Extended context windows - delegate to Gemini (1M tokens) or O3 (200K tokens)
• Local model support via Ollama, vLLM, or custom endpoints

WORKFLOW EXAMPLES:
1. "Perform a codereview using gemini pro and o3, then use planner to fix issues"
2. "Debug this with o3 after gathering related code"
3. "Get consensus from flash and pro on this architecture decision"
4. "Use local-llama for quick analysis, then opus for security review"

THINKING MODES (Gemini models):
• 'minimal' or 'low' for quick tasks
• 'high' or 'max' for complex problems

Remember: Zen MCP enhances Claude's capabilities by orchestrating multiple AI models. Claude stays in control but gets enhanced perspectives from the best AI for each subtask.

To see available models: "Use zen to list available models"
"""

def main():
    # Get the context from stdin (if any)
    input_data = ""
    if not sys.stdin.isatty():
        input_data = sys.stdin.read()
    
    # Parse input if it's JSON
    try:
        context = json.loads(input_data) if input_data else {}
    except:
        context = {}
    
    # Prepare the response
    response = {
        "additionalContext": get_zen_instructions(),
        "metadata": {
            "source": "zen-mcp-session-hook",
            "purpose": "Enhance Claude Code with Zen MCP capabilities"
        }
    }
    
    # Output the response
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()