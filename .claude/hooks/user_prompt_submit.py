#!/usr/bin/env python3
"""Context Intelligence-powered UserPromptSubmit hook handler for Claude Code.

Enhanced with Context Intelligence Engine for ZEN Co-pilot Phase 1:
- Git context analysis
- Technology stack detection  
- Smart prompt enhancement
- Progressive verbosity adaptation
"""

import json
import sys
import os
import asyncio

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

from modules.core.zen_consultant import create_zen_consultation_response, create_zen_consensus_request, ComplexityLevel
from modules.core.context_intelligence_engine import create_context_aware_directive
from modules.utils import validate_prompt, log_debug


async def generate_context_intelligent_response(prompt: str) -> dict:
    """Generate context-intelligent response using the Context Intelligence Engine."""
    try:
        # Get current working directory for project analysis
        project_dir = os.getcwd()
        
        # Use Context Intelligence Engine for enhanced directive generation
        result = await create_context_aware_directive(prompt, project_dir)
        
        log_debug("Context Intelligence directive generated", {
            "prompt_length": len(prompt),
            "context_confidence": result["hookSpecificOutput"].get("confidenceMetrics", {}).get("overall_confidence", 0.0),
            "tech_stacks": result["hookSpecificOutput"].get("contextAnalysis", {}).get("technology_stacks", []),
            "user_expertise": result["hookSpecificOutput"].get("userAdaptation", {}).get("detected_expertise", "unknown")
        })
        
        return result
        
    except Exception as e:
        log_debug("Context Intelligence Engine error, falling back to ZEN consultant", {"error": str(e)})
        
        # Fallback to existing ZEN consultant
        use_concise = len(prompt) < 200 or "CONCISE" in prompt.upper()
        format_type = "concise" if use_concise else "verbose"
        
        fallback_result = create_zen_consultation_response(prompt, format_type)
        
        # Enhance fallback with error information
        fallback_result["hookSpecificOutput"]["fallbackUsed"] = True
        fallback_result["hookSpecificOutput"]["fallbackReason"] = str(e)
        
        return fallback_result


def main():
    """Main Context Intelligence-powered hook handler."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract and validate prompt
    prompt = validate_prompt(input_data.get("prompt", ""))
    log_debug("Context Intelligence consultation starting", {"prompt_length": len(prompt)})
    
    # Check for override flags to disable Context Intelligence
    force_basic_zen = any(flag in prompt.upper() for flag in ["BASIC_ZEN", "DISABLE_CONTEXT", "SIMPLE_ZEN"])
    
    if force_basic_zen:
        # Use basic ZEN consultant without context intelligence
        log_debug("Using basic ZEN consultant (override detected)")
        use_concise = len(prompt) < 200 or "CONCISE" in prompt.upper()
        format_type = "concise" if use_concise else "verbose"
        output = create_zen_consultation_response(prompt, format_type)
        # Extract additionalContext and print it directly
        additional_context = output.get("hookSpecificOutput", {}).get("additionalContext", "")
        print(additional_context)
    else:
        # Use Context Intelligence Engine
        try:
            output = asyncio.run(generate_context_intelligent_response(prompt))
            # Extract additionalContext and print it directly
            additional_context = output.get("hookSpecificOutput", {}).get("additionalContext", "")
            print(additional_context)
        except Exception as e:
            log_debug("Fatal error in Context Intelligence, using emergency fallback", {"error": str(e)})
            # Emergency fallback - print minimal context directly
            print("🤖 ZEN: SWARM → discovery phase → coder → mcp__zen__analyze → conf:0.7")
    
    # Exit with code 0 to inject context
    sys.exit(0)


if __name__ == "__main__":
    main()