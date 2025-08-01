"""Context builder for generating hook responses."""

from typing import List, Dict, Any, Optional
from .analyzer import PatternMatch


class ContextBuilder:
    """Builds context from pattern matches."""
    
    def __init__(self, header: str = "📋 Dynamic MCP ZEN Context Injection"):
        self.header = header
        self.footer_sections: List[str] = []
    
    def add_footer_section(self, section: str) -> None:
        """Add a footer section that appears on all responses."""
        self.footer_sections.append(section)
    
    def build_context(self, matches: List[PatternMatch], deduplication: bool = True) -> str:
        """Build context from pattern matches."""
        context_parts = [self.header]
        
        # Sort matches by priority (higher first)
        sorted_matches = sorted(matches, key=lambda m: m.priority, reverse=True)
        
        # Deduplicate if requested
        if deduplication:
            seen_messages = set()
            unique_matches = []
            for match in sorted_matches:
                if match.message not in seen_messages:
                    seen_messages.add(match.message)
                    unique_matches.append(match)
            sorted_matches = unique_matches
        
        # Add matched patterns
        for match in sorted_matches:
            context_parts.append(match.message)
        
        # Add footer sections
        for section in self.footer_sections:
            context_parts.append(section)
        
        return "\n".join(context_parts)
    
    def build_hook_response(self, context: str, event_name: str = "UserPromptSubmit") -> Dict[str, Any]:
        """Build the complete hook response."""
        return {
            "hookSpecificOutput": {
                "hookEventName": event_name,
                "additionalContext": context
            }
        }