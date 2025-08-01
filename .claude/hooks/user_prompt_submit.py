#!/usr/bin/env python3
"""UserPromptSubmit hook handler for Claude Code.

This modular hook system analyzes user prompts and injects
relevant context based on pattern matching.
"""

import json
import sys

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

from modules.core import Config, ContextBuilder
from modules.analyzers import AnalyzerManager
from modules.utils import validate_prompt, log_debug, CustomPatternLoader


def main():
    """Main hook handler."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract and validate prompt
    prompt = validate_prompt(input_data.get("prompt", ""))
    log_debug("Processing prompt", {"length": len(prompt)})
    
    # Load configuration
    config = Config()
    
    # Initialize analyzer manager with enabled analyzers
    enabled_analyzers = config.get_enabled_analyzers()
    manager = AnalyzerManager(enabled_analyzers)
    
    # Load custom patterns if available
    custom_analyzer = CustomPatternLoader.load_from_config(config.config)
    if custom_analyzer:
        manager.add_analyzer(custom_analyzer)
        log_debug("Loaded custom patterns", {"count": len(custom_analyzer.patterns)})
    
    # Analyze prompt
    matches = manager.analyze(prompt)
    log_debug("Pattern matches found", {"count": len(matches)})
    
    # Build context
    context_builder = ContextBuilder()
    
    # Add quick tips if enabled
    quick_tips_config = config.get("quick_tips", {})
    if quick_tips_config.get("enabled", True):
        context_builder.add_footer_section(quick_tips_config.get("message", ""))
    
    # Build final context
    context = context_builder.build_context(
        matches, 
        deduplication=config.get("deduplication", True)
    )
    
    # Create hook response
    output = context_builder.build_hook_response(context)
    
    # Output the JSON response
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()