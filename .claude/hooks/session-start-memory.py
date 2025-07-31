#!/usr/bin/env python3
"""
Memory-Aware Session Start Hook
Injects instructions for Claude Code to use memory-enhanced ZEN workflows
"""

import json
import os
from datetime import datetime

def get_memory_workflow_instructions():
    """Return comprehensive memory-aware workflow instructions"""
    
    group_id = os.environ.get("GRAPHITI_GROUP_ID", "claude-code-flowed")
    
    return f"""
🧠 MEMORY-ENHANCED ZEN WORKFLOWS ACTIVE

You have access to a persistent knowledge graph (Graphiti) that remembers all your work.
Group: {group_id} | Database: Neo4j

📋 MANDATORY WORKFLOW PATTERN:

For EVERY ZEN tool usage, follow this 3-step pattern:

1️⃣ **BEFORE using any ZEN tool** - Search memory first:
   ```
   # Example for debugging:
   Search my memory for "similar authentication errors"
   Search my memory for "past debug sessions on login issues"
   
   # Example for planning:
   Search my memory for "API design patterns we've used"
   Search my memory for "architectural decisions"
   ```

2️⃣ **WHEN calling ZEN tools** - Include memory context:
   ```
   # Bad (no context):
   Use zen debug to fix login issue
   
   # Good (with memory context):
   Use zen debug to fix login issue. Previous similar issues from memory show:
   - Token expiry was the cause in 3 past cases
   - We solved it before by implementing refresh tokens
   - Last time we had a race condition in the auth middleware
   ```

3️⃣ **AFTER ZEN completes** - Connect insights:
   ```
   # The output is auto-saved, but you should:
   - Note any new patterns discovered
   - Update related documentation if needed
   - Search for other code that might have the same issue
   ```

🎯 MEMORY SEARCH PATTERNS:

• **Temporal queries**: "Show work from yesterday", "What did we do last week"
• **Similarity search**: "Find similar bugs", "Show related features"  
• **Entity search**: "Find all Requirement entities", "Show Procedure for deployment"
• **Relationship queries**: "How is X related to Y", "What depends on this module"

⚡ QUICK MEMORY COMMANDS:

• `Search my memory for [topic]` - Find past work
• `What do I know about [concept]` - Get accumulated knowledge
• `Show recent [tool] sessions` - See past tool usage
• `Find similar [problems/solutions]` - Pattern matching

🔄 AUTOMATIC MEMORY FEATURES:

• All prompts are captured to memory
• ZEN tool outputs are stored with rich metadata
• Code changes are tracked over time
• Decisions and rationale are preserved

💡 PRO TIP: The more specific your memory searches, the better the context for ZEN tools.
Instead of "search for bugs", try "search for authentication timeout bugs in API endpoints".

Remember: Every ZEN analysis builds on the collective intelligence of all your past work!"""

def main():
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get the workflow instructions
    instructions = get_memory_workflow_instructions()
    
    # Prepare response
    response = {
        "additionalContext": instructions,
        "metadata": {
            "source": "session-start-memory",
            "timestamp": timestamp,
            "group_id": os.environ.get("GRAPHITI_GROUP_ID", "claude-code-flowed"),
            "purpose": "Enable memory-enhanced ZEN workflows"
        }
    }
    
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()