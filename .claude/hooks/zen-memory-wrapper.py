#!/usr/bin/env python3
"""
ZEN Memory Wrapper Hook
Injects memory-aware instructions to make ZEN tools smarter by guiding Claude Code
to search memory before and after using ZEN tools.
"""

import json
import sys
import re
import os

def detect_zen_intent(prompt):
    """Detect if user intends to use a ZEN tool"""
    zen_patterns = {
        'debug': r'\b(debug|bug|error|issue|problem|broken|fix|crash|fail)\b',
        'codereview': r'\b(review|check|audit|quality|security|vulnerable|secure)\b',
        'planner': r'\b(plan|design|architect|structure|organize|break\s*down)\b',
        'analyze': r'\b(understand|analyze|explain|how\s*does|what\s*is)\b',
        'testgen': r'\b(test|tests|testing|coverage|unit\s*test|integration)\b',
        'refactor': r'\b(refactor|improve|clean|optimize|decompose|split)\b',
        'docgen': r'\b(document|documentation|docs|comment|explain\s*code)\b',
        'consensus': r'\b(should|decide|choice|option|alternative|vs|versus)\b',
        'thinkdeep': r'\b(think|consider|explore|brainstorm|idea|complex|deep)\b'
    }
    
    prompt_lower = prompt.lower()
    detected_tools = []
    
    for tool, pattern in zen_patterns.items():
        if re.search(pattern, prompt_lower):
            detected_tools.append(tool)
    
    return detected_tools

def extract_context_keywords(prompt):
    """Extract key concepts from the prompt for memory search"""
    # Remove common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    # Extract potential keywords
    words = re.findall(r'\b\w+\b', prompt.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Look for technical terms
    tech_terms = []
    tech_patterns = [
        r'\b\w+\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb)\b',  # file names
        r'\b[A-Z][a-z]+[A-Z]\w*\b',  # CamelCase
        r'\b\w+_\w+\b',  # snake_case
        r'\b\w+-\w+\b',  # kebab-case
    ]
    
    for pattern in tech_patterns:
        tech_terms.extend(re.findall(pattern, prompt))
    
    return list(set(keywords + tech_terms))[:5]  # Top 5 unique terms

def generate_memory_instructions(prompt, detected_tools):
    """Generate instructions for Claude Code to search and use memory"""
    keywords = extract_context_keywords(prompt)
    
    instructions = []
    
    # Pre-ZEN memory search instructions
    if detected_tools:
        tool_name = detected_tools[0]
        
        pre_search = f"""
🧠 MEMORY-ENHANCED ZEN WORKFLOW DETECTED

Before using ZEN {tool_name}, enrich the context with memory:

1️⃣ **Search Historical Context** (execute these searches):
   • Search memory for "{' '.join(keywords[:2])}"
   • Find past {tool_name} sessions related to this topic
   • Look for similar issues and their solutions

2️⃣ **Include Memory in ZEN Prompt**:
   When calling the ZEN tool, include relevant findings:
   "Use zen {tool_name} to {prompt}. 
    Previous context: [include relevant memory findings]"

3️⃣ **Connect to Past Patterns**:
   • Reference successful approaches from memory
   • Mention what didn't work before
   • Build on previous decisions
"""
        instructions.append(pre_search)
        
        # Post-ZEN instructions
        post_action = f"""
4️⃣ **After ZEN Completes**:
   • The output will be automatically stored via zen-graphiti-bridge
   • Consider searching for related work to connect insights
   • Update any changed decisions or approaches
"""
        instructions.append(post_action)
    
    # General memory-aware instructions
    else:
        general = f"""
🧠 MEMORY-AWARE DEVELOPMENT

Consider enriching your request with historical context:

📍 **Quick Memory Searches**:
   • Search memory for "{' '.join(keywords[:3])}"
   • Find similar work or decisions
   • Check for established patterns

💡 **When Using ZEN Tools**:
   Always include relevant context from memory to get better results.
   The AI will build on past learnings rather than starting fresh.
"""
        instructions.append(general)
    
    return '\n'.join(instructions)

def generate_workflow_examples(detected_tools):
    """Provide specific workflow examples based on detected intent"""
    examples = {
        'debug': """
📋 **Memory-Enhanced Debug Example**:
```
# First, search for similar issues
Search memory for "authentication timeout errors"

# Then use ZEN with context
Use zen debug to investigate timeout issue. Previous similar issues showed: [memory findings]

# After solving, the solution is auto-saved for future reference
```""",
        
        'codereview': """
📋 **Memory-Enhanced Review Example**:
```
# Check past reviews of this module
Search memory for "user authentication code reviews"

# Include history in review request
Use zen codereview on auth.py with context of past security concerns: [memory findings]
```""",
        
        'planner': """
📋 **Memory-Enhanced Planning Example**:
```
# Find related architectural decisions
Search memory for "API design decisions"

# Plan with historical context
Use zen planner for new feature, considering past patterns: [memory findings]
```"""
    }
    
    if detected_tools and detected_tools[0] in examples:
        return examples[detected_tools[0]]
    return ""

def main():
    # Read input
    input_data = sys.stdin.read() if not sys.stdin.isatty() else ""
    
    try:
        data = json.loads(input_data) if input_data else {}
        prompt = data.get("prompt", "")
    except:
        prompt = input_data
    
    # Detect ZEN tool intent
    detected_tools = detect_zen_intent(prompt)
    
    # Generate memory-aware instructions
    instructions = generate_memory_instructions(prompt, detected_tools)
    workflow_example = generate_workflow_examples(detected_tools)
    
    # Combine all guidance
    additional_context = instructions
    if workflow_example:
        additional_context += "\n" + workflow_example
    
    # Add memory status
    group_id = os.environ.get("GRAPHITI_GROUP_ID", "claude-code-flowed")
    additional_context += f"\n\n🔗 Memory Group: {group_id}"
    
    response = {
        "additionalContext": additional_context,
        "metadata": {
            "source": "zen-memory-wrapper",
            "detected_tools": detected_tools,
            "memory_enhanced": True,
            "keywords": extract_context_keywords(prompt)
        }
    }
    
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()