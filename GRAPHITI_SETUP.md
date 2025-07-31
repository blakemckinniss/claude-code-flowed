# Graphiti Memory Integration Setup Guide

## Quick Start

Run the setup script to configure Graphiti integration:

```bash
./setup-graphiti.sh
```

## Manual Setup

### 1. Install Neo4j

Choose one of the following options:

#### Option A: Docker (Recommended)
```bash
docker run -d \
  --name neo4j-graphiti \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:latest
```

#### Option B: Neo4j Desktop
- Download from https://neo4j.com/download/
- Create a new database
- Start the database

### 2. Clone Graphiti
```bash
git clone https://github.com/getzep/graphiti.git
cd graphiti/mcp_server
uv sync  # or pip install -r requirements.txt
```

### 3. Update Configuration

Edit `.claude/settings.json` and update the Graphiti path and credentials:

```json
"mcpServers": {
  "graphiti-memory": {
    "args": [
      "run",
      "--directory",
      "/actual/path/to/graphiti/mcp_server",  // Update this
      ...
    ],
    "env": {
      "NEO4J_URI": "bolt://localhost:7687",
      "NEO4J_USER": "neo4j",
      "NEO4J_PASSWORD": "your-password",      // Update this
      "OPENAI_API_KEY": "your-openai-key"     // Update this
    }
  }
}
```

### 4. Make Hooks Executable
```bash
chmod +x .claude/hooks/*.py
```

### 5. Restart Claude Code
The new configuration will be loaded on the next session.

## Verifying Integration

### Check Active Hooks

1. **SessionStart Hook**: You should see Zen MCP instructions when starting Claude
2. **UserPromptSubmit Hook**: Each prompt should show memory capture notifications
3. **PostToolUse Hook**: Zen tool outputs should indicate memory storage

### Test Memory Storage

Try these commands:

```bash
# Use a Zen tool
Use zen to analyze this code

# Check if it was stored
Search my memory for "code analysis"
```

### View Neo4j Data

Open Neo4j Browser at http://localhost:7474 and run:

```cypher
// See all episodes
MATCH (e:Episode) RETURN e LIMIT 10

// See all entities
MATCH (n:Entity) RETURN n LIMIT 20

// See relationships
MATCH (e1:Entity)-[r]->(e2:Entity) 
RETURN e1, r, e2 LIMIT 50
```

## Troubleshooting

### Neo4j Connection Issues
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Test connection
curl -u neo4j:password http://localhost:7474/db/neo4j/
```

### MCP Server Not Starting
- Check the path in settings.json is correct
- Ensure Python dependencies are installed
- Check logs: `claude --debug`

### Memory Not Being Stored
- Verify OPENAI_API_KEY is set correctly
- Check Neo4j is accessible
- Look for errors in Claude logs

## Integration Features

### Automatic Capture
- All user prompts and Claude responses
- Zen tool inputs and outputs
- Code changes and file operations
- Debug sessions and solutions

### Memory Queries
```
# Search past work
Search my memory for "authentication"

# Find related concepts  
What do I know about "user management"

# Get temporal view
Show my work from yesterday

# Find connections
How is "API design" related to "security"
```

### Cross-Tool Intelligence
- Code reviews → Implementation changes
- Debug findings → Solution patterns
- Planning decisions → Actual implementations
- Consensus outcomes → Final decisions

## Best Practices

1. **Use Descriptive Prompts**: More context = better memory extraction
2. **Regular Queries**: Leverage past knowledge frequently
3. **Structured Data**: Use JSON format for complex information
4. **Group Projects**: Use consistent group_ids for related work
5. **Session Continuity**: Use `claude --resume` to maintain context

## Architecture Overview

```
Claude Code → Hooks → Memory Capture
     ↓           ↓            ↓
  Zen Tools → Bridge Hook → Graphiti
     ↓                         ↓
  Analysis ← Knowledge Graph ← Neo4j
```

Your development environment now has persistent, searchable memory that grows smarter with every interaction!