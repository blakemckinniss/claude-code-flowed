# Graphiti Integration with Claude Code + ZEN + Claude Flow

## Overview

This document describes how to integrate Graphiti (temporal knowledge graph) into your Claude Code REPL environment alongside ZEN MCP and Claude Flow for a complete AI-enhanced development experience.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Claude Code                           │
│  ┌─────────────────────┬──────────────┬─────────────────┐  │
│  │   SessionStart Hook  │ UserPrompt   │  Other Hooks    │  │
│  │   (Context Loading)  │ Submit Hook  │  (Memory Save)  │  │
│  └──────────┬──────────┴──────┬───────┴────────┬────────┘  │
│             │                  │                 │           │
│  ┌──────────▼──────────────────▼─────────────────▼────────┐ │
│  │              MCP Servers (Model Context Protocol)        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │ │
│  │  │  ZEN MCP    │  │ Claude Flow │  │ Graphiti MCP  │  │ │
│  │  │ (AI Brain)  │  │   (Swarms)  │  │   (Memory)    │  │ │
│  │  └─────────────┘  └─────────────┘  └───────┬───────┘  │ │
│  └─────────────────────────────────────────────┼──────────┘ │
└────────────────────────────────────────────────┼────────────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │     Neo4j Database      │
                                    │  (Knowledge Graph)      │
                                    └─────────────────────────┘
```

## Key Benefits

1. **Persistent Memory**: All interactions, code changes, and AI consultations are stored in a knowledge graph
2. **Temporal Awareness**: Track how your code and understanding evolve over time
3. **Cross-Session Context**: Resume work with full historical context
4. **Intelligent Retrieval**: Find relevant past solutions, decisions, and patterns
5. **Multi-Modal Integration**: ZEN consultations are automatically captured in memory

## Setup Instructions

### 1. Install Graphiti MCP Server

```bash
# Clone Graphiti repository
git clone https://github.com/getzep/graphiti.git
cd graphiti/mcp_server

# Install dependencies using uv
uv sync
```

### 2. Set Up Neo4j Database

Option A: Neo4j Desktop (Recommended for development)
- Download and install [Neo4j Desktop](https://neo4j.com/download/)
- Create a new project and database
- Start the database

Option B: Docker
```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 3. Configure Environment Variables

Create `.env` file in the Graphiti MCP server directory:
```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# OpenAI Configuration (for entity extraction)
OPENAI_API_KEY=your-openai-key
MODEL_NAME=gpt-4-turbo-preview
SMALL_MODEL_NAME=gpt-3.5-turbo

# Performance
SEMAPHORE_LIMIT=10
```

### 4. Update Claude Code Settings

Edit `/home/blake/flowed/.claude/settings.json` to add Graphiti MCP server:

```json
{
  "mcpServers": {
    "zen": {
      // ... existing ZEN configuration ...
    },
    "claude-flow": {
      // ... existing Claude Flow configuration ...
    },
    "graphiti-memory": {
      "transport": "stdio",
      "command": "/path/to/uv",
      "args": [
        "run",
        "--directory",
        "/path/to/graphiti/mcp_server",
        "graphiti_mcp_server.py",
        "--transport",
        "stdio",
        "--group-id",
        "claude-code-project",
        "--use-custom-entities"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

### 5. Add Graphiti Integration Hook

Update your `.claude/settings.json` hooks section:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/home/blake/flowed/.claude/hooks/prompt-submit-zen.py"
          },
          {
            "type": "command",
            "command": "/home/blake/flowed/.claude/hooks/graphiti-integration.py"
          }
        ]
      }
    ],
    // ... other hooks ...
  }
}
```

## Usage Patterns

### 1. Automatic Memory Capture

Every interaction is automatically stored:
- User prompts and Claude's responses
- Code changes and file operations
- ZEN tool usage and AI consultations
- Debug sessions and solutions
- Planning and architectural decisions

### 2. Memory Queries

Use Graphiti MCP tools directly in Claude:

```
# Search for past solutions
Use graphiti-memory to search for "authentication implementation"

# Find related concepts
Use graphiti-memory to find facts about "user management"

# Get recent work
Use graphiti-memory to show episodes from today

# Find connections
Use graphiti-memory to search nodes related to "API design"
```

### 3. Integration with ZEN Tools

When using ZEN tools, Graphiti automatically captures:

```
# Code review outcomes
Use zen codereview on auth.py
# → Automatically stores review findings and recommendations

# Debug solutions
Use zen debug to find why tests are failing
# → Captures the problem, investigation, and solution

# Planning decisions
Use zen planner to design the notification system
# → Records architectural decisions and reasoning

# Consensus outcomes
Get consensus from o3 and gemini on database design
# → Stores multiple perspectives and final decision
```

### 4. Structured Data Storage

Graphiti can process JSON data for rich entity extraction:

```python
# Example: Store project requirements
add_memory(
    name="Project Requirements",
    episode_body=json.dumps({
        "project": "E-commerce Platform",
        "requirements": [
            {"type": "functional", "priority": "high", "description": "User authentication"},
            {"type": "performance", "target": "< 200ms response time"}
        ]
    }),
    source="json",
    source_description="Initial requirements gathering"
)
```

## Advanced Features

### 1. Custom Entity Types

The integration includes custom entities for development:
- **Requirement**: Project requirements and specifications
- **Preference**: User preferences and coding styles
- **Procedure**: Development workflows and processes

### 2. Temporal Queries

Track how things change over time:
```
# Find when a bug was introduced
Search memory for "login bug" and show temporal progression

# Track feature evolution
Show how "payment system" evolved over the past week
```

### 3. Cross-Project Memory

Use different group_ids for different projects:
```bash
# Start Graphiti with project-specific memory
--group-id "project-alpha"
```

### 4. Memory Export/Import

Backup and share knowledge:
```
# Export knowledge graph
Use graphiti-memory to export group "project-alpha"

# Import into new environment
Use graphiti-memory to import knowledge from backup
```

## Best Practices

1. **Meaningful Episode Names**: When manually adding memory, use descriptive names
2. **Structured Data**: Use JSON format for complex information
3. **Regular Queries**: Periodically search memory to leverage past learnings
4. **Group Organization**: Use consistent group_ids for related work
5. **Context Enrichment**: Include relevant metadata in episodes

## Troubleshooting

### Neo4j Connection Issues
- Ensure Neo4j is running: `neo4j status`
- Check credentials in `.env` file
- Verify port 7687 is available

### MCP Server Not Found
- Ensure full path to `uv` is specified
- Check Python environment has all dependencies
- Verify `graphiti_mcp_server.py` path is correct

### Memory Not Being Stored
- Check Graphiti MCP server logs
- Verify OpenAI API key for entity extraction
- Ensure group_id is consistent

## Example Workflow

1. **Start Development Session**
   ```
   claude --resume  # Loads previous context
   ```

2. **Query Past Work**
   ```
   Use graphiti-memory to search for "similar authentication implementations"
   ```

3. **Implement Feature with ZEN**
   ```
   Use zen planner to break down OAuth integration
   Use zen codereview after implementation
   ```

4. **Debug with History**
   ```
   Use graphiti-memory to find "previous OAuth errors"
   Use zen debug with context from past solutions
   ```

5. **Document Decisions**
   ```
   Add to graphiti-memory: "Chose JWT over sessions because..."
   ```

## Future Enhancements

1. **Visual Knowledge Graph**: Neo4j browser for exploring connections
2. **Automated Summaries**: Daily/weekly development summaries
3. **Pattern Detection**: Identify recurring issues or successful patterns
4. **Team Sharing**: Shared knowledge graphs for team learning
5. **AI Training**: Use accumulated knowledge to fine-tune models

## Conclusion

The integration of Graphiti with Claude Code, ZEN, and Claude Flow creates a powerful development environment where:
- Every interaction builds persistent knowledge
- Past solutions inform current work
- AI consultations are captured and reusable
- Development patterns emerge from accumulated data

This creates a true "learning" development environment that gets smarter with use.