# ZEN Memory Wrapper - How It Works

## Overview

The "ZEN Memory Wrapper" isn't a new tool - it's intelligent context injection that guides Claude Code to automatically integrate memory searches with ZEN tool usage. This creates a seamless workflow where every ZEN analysis builds on past knowledge.

## How the Wrapper Works

### 1. **Session Start Injection**
When Claude Code starts, two hooks inject instructions:
- `session-start-zen.py` - Lists available ZEN tools
- `session-start-memory.py` - Injects mandatory memory workflow patterns

### 2. **Prompt Analysis & Guidance**
When you type a prompt, three hooks work together:
- `prompt-submit-zen.py` - Suggests appropriate ZEN tools
- `zen-memory-wrapper.py` - Injects memory search instructions
- `graphiti-integration.py` - Captures the interaction

### 3. **Automatic Memory Integration**
The wrapper detects ZEN intent and injects specific instructions to Claude Code.

## Example Workflows

### Before (Without Memory Wrapper):
```
User: Debug why login is failing

Claude Code: I'll use zen debug to investigate...
[Runs zen debug with no historical context]
```

### After (With Memory Wrapper):
```
User: Debug why login is failing

[zen-memory-wrapper.py injects:]
🧠 MEMORY-ENHANCED ZEN WORKFLOW DETECTED

Before using ZEN debug, enrich the context with memory:

1️⃣ Search Historical Context:
   • Search memory for "login failing"
   • Find past debug sessions related to this topic
   • Look for similar issues and their solutions

2️⃣ Include Memory in ZEN Prompt:
   When calling the ZEN tool, include relevant findings...

Claude Code: I'll search your memory for similar login issues first...
[Searches memory]
Found 3 similar cases where token expiry was the issue.

Now I'll use zen debug with this context...
[Runs zen debug with historical context included]
```

## Real Example: Debugging with Memory

### Step 1: User Request
```
I'm getting authentication timeouts in production
```

### Step 2: Wrapper Injects Instructions
The `zen-memory-wrapper.py` detects "authentication" and "timeout" keywords and injects:
```
🧠 MEMORY-ENHANCED ZEN WORKFLOW DETECTED

Before using ZEN debug, search for:
• "authentication timeout"
• Past debug sessions on auth issues
• Similar timeout problems

Include findings in your ZEN prompt!
```

### Step 3: Claude Code Follows Instructions
```
Claude Code: I'll search your memory for similar authentication timeout issues...

[Executes: Search memory for "authentication timeout"]
Found 2 relevant episodes:
1. "Debug Request - 2024-01-15" - JWT token expiry issue
2. "Debug Request - 2024-01-22" - Database connection pool exhaustion

[Executes: Use zen debug to investigate authentication timeouts. 
Previous cases showed JWT expiry and connection pool issues...]
```

### Step 4: Enhanced Analysis
ZEN debug now has context about:
- Previous JWT expiry issues and how they were solved
- Connection pool problems that caused timeouts
- Successful solutions from the past

## Benefits of the Wrapper Approach

### 1. **No New Tools Needed**
- Works with existing ZEN tools
- Uses Claude Code's native orchestration
- No additional complexity

### 2. **Automatic Context Enrichment**
- Every ZEN analysis includes relevant history
- Patterns emerge from accumulated knowledge
- Mistakes are avoided by learning from the past

### 3. **Progressive Learning**
- Each use makes future uses smarter
- Solutions build on previous solutions
- Knowledge compounds over time

## Configuration

### Hook Execution Order:
1. `session-start-zen.py` - Base ZEN instructions
2. `session-start-memory.py` - Memory workflow patterns
3. `prompt-submit-zen.py` - Tool suggestions
4. `zen-memory-wrapper.py` - Memory integration instructions
5. `graphiti-integration.py` - Capture to memory
6. `zen-graphiti-bridge.py` - Capture ZEN outputs

### Key Features:

#### Keyword Extraction
The wrapper extracts technical terms from your prompt:
- File names (`.py`, `.js`, etc.)
- CamelCase identifiers
- snake_case variables
- Technical concepts

#### Smart Suggestions
Based on detected intent:
- Debug → Search for similar bugs
- Review → Find past reviews
- Plan → Look up architectural decisions
- Refactor → Find successful patterns

#### Workflow Examples
Provides specific examples for common patterns:
- Memory-enhanced debugging
- Historical code reviews
- Planning with context

## Usage Tips

### 1. **Let It Guide You**
Don't override the memory search suggestions - they're designed to find relevant context.

### 2. **Be Specific**
The more specific your initial prompt, the better the memory search keywords.

### 3. **Build Patterns**
After a few uses, you'll have rich patterns that make each subsequent use more effective.

### 4. **Trust the Process**
The wrapper ensures Claude Code always checks memory before using ZEN tools.

## Technical Details

### What Gets Injected:
```python
# For a debug request, the wrapper injects:
{
    "additionalContext": """
    🧠 MEMORY-ENHANCED ZEN WORKFLOW DETECTED
    
    Before using ZEN debug, enrich the context with memory:
    1. Search memory for "keyword1 keyword2"
    2. Find past debug sessions
    3. Include findings in ZEN prompt
    """,
    "metadata": {
        "detected_tools": ["debug"],
        "keywords": ["authentication", "timeout", "login"],
        "memory_enhanced": true
    }
}
```

### How Claude Code Responds:
1. Sees the injected instructions
2. Executes memory searches first
3. Includes findings in ZEN tool prompt
4. Gets better results from ZEN
5. Output is automatically saved to memory

## Summary

The ZEN Memory Wrapper creates an intelligent feedback loop:
- Past work informs current analysis
- Current analysis enriches future work
- Knowledge compounds automatically
- No manual memory management needed

This is how we "wrap" ZEN tools within Claude Code's orchestration model - through intelligent context injection that guides behavior rather than creating new tools.