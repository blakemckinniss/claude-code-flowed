# Graphiti Memory Integration - Example Usage Patterns

## 🚀 Getting Started

### Prerequisites
1. Neo4j running at localhost:7687 (via Docker)
2. OpenAI API key configured in `graphiti/mcp_server/.env`
3. Claude Code restarted after configuration

## 📋 Example Workflows

### 1. Debug Session with Memory

```bash
# Start a debug session - automatically captured
Use zen debug to find why user authentication is failing

# Later, search for similar issues
Search my memory for "authentication failures"

# Find past solutions
What solutions have I used for login bugs?
```

### 2. Code Review with Historical Context

```bash
# Perform code review
Use zen codereview on src/auth/jwt_handler.py

# Find previous reviews of similar code
Search memory for "JWT code reviews"

# Track improvement over time
Show how JWT implementation evolved this week
```

### 3. Planning with Memory

```bash
# Create a development plan
Use zen planner to design notification system

# Reference past architectural decisions
What architectural patterns have I used before?

# Find related plans
Search memory for "message queue designs"
```

### 4. Building on Past Work

```bash
# Find how you solved similar problems
Search my memory for "database connection pooling"

# Get consensus from past decisions
What did the AI consensus say about using Redis vs RabbitMQ?

# Track feature evolution
Show how the payment system was implemented step by step
```

### 5. Cross-Session Context

```bash
# Resume work with full context
claude --resume

# Ask about previous session
What was I working on yesterday?

# Find specific decisions
Why did we choose PostgreSQL over MongoDB for this project?
```

## 🧠 Memory Query Patterns

### Basic Searches
```bash
# Search by keyword
Search memory for "OAuth implementation"

# Search by concept
What do I know about "microservices"

# Search by time
Show work from the last 3 days
```

### Advanced Queries
```bash
# Find relationships
How is "user authentication" related to "session management"

# Track decisions
Show all architectural decisions for the API

# Find patterns
What debugging approaches have been most successful?
```

### Entity-Specific Searches
```bash
# Find requirements
Search for all Requirement entities about "performance"

# Find procedures
What Procedure entities exist for "deployment"

# Find preferences
Show all Preference entities for "code style"
```

## 🔗 Integration Patterns

### Zen Tool + Memory
Every Zen tool usage is automatically captured:

```bash
# Debug session → Memory
Use zen debug on connection timeout issue
# Automatically stores: problem, investigation, solution

# Code review → Memory
Use zen codereview with gemini on auth module
# Stores: findings, recommendations, code quality metrics

# Planning → Memory
Use zen planner for microservices migration
# Stores: architectural decisions, migration steps
```

### Structured Data Storage
```python
# Store project requirements as structured data
add_memory(
    name="E-commerce Requirements v2",
    episode_body=json.dumps({
        "project": "E-commerce Platform",
        "version": "2.0",
        "requirements": {
            "functional": [
                "Multi-tenant support",
                "Real-time inventory",
                "Payment gateway integration"
            ],
            "performance": {
                "response_time": "< 200ms",
                "concurrent_users": 10000
            }
        }
    }),
    source="json"
)
```

## 🎯 Best Practices

### 1. Descriptive Prompts
```bash
# Good - provides context
Use zen debug to investigate why JWT tokens are expiring prematurely in production

# Less effective
Debug token issue
```

### 2. Regular Memory Queries
```bash
# Before starting new work
Search memory for similar implementations

# During problem-solving
What approaches have I tried for this type of issue?

# After completing tasks
# Memory is automatically updated!
```

### 3. Leverage Temporal Awareness
```bash
# Track evolution
Show how the authentication system changed over time

# Find when issues started
When did we first encounter rate limiting problems?

# Review decision history
Show all consensus decisions this month
```

### 4. Cross-Project Learning
```bash
# Use different group IDs for projects
# In settings.json: "group_id": "project-alpha"

# Later, in another project
Search memory from project-alpha for "API patterns"
```

## 📊 Neo4j Exploration

Access Neo4j Browser at http://localhost:7474

### Useful Cypher Queries

```cypher
// View recent episodes
MATCH (e:Episode)
RETURN e
ORDER BY e.created_at DESC
LIMIT 20

// Find all entities from Zen tools
MATCH (e:Entity)-[:MENTIONED_IN]->(ep:Episode)
WHERE ep.source_description CONTAINS 'ZEN'
RETURN e, ep

// Track code evolution
MATCH (e:Entity {name: 'authentication'})-[r]->(related)
RETURN e, r, related

// Find most connected concepts
MATCH (e:Entity)
RETURN e.name, COUNT(*) as connections
ORDER BY connections DESC
LIMIT 10
```

## 🔄 Workflow Integration

### Morning Routine
1. `claude --resume` - Load previous context
2. "What was I working on?" - Quick memory refresh
3. "Show today's priorities" - From stored plans

### Before Implementing
1. "Search memory for similar features"
2. "What patterns work well for this?"
3. Use Zen planner with memory context

### During Development
1. Zen tools automatically capture progress
2. Decisions are stored as you make them
3. Problems and solutions build knowledge

### End of Session
1. Session automatically saved
2. Summary generated if configured
3. Ready for next session with full context

## 🚨 Troubleshooting

### Memory Not Storing
1. Check OpenAI API key in `.env`
2. Verify Neo4j is running: `docker ps`
3. Check logs: `docker logs neo4j-graphiti`

### MCP Server Issues
1. Test with: `uv run graphiti_mcp_server.py --help`
2. Check Claude logs for MCP errors
3. Verify paths in settings.json

### Query Not Finding Results
1. Check group_id matches
2. Allow time for entity extraction
3. Use broader search terms

## 💡 Pro Tips

1. **Memory Builds Over Time**: The more you use it, the smarter it gets
2. **Cross-Tool Intelligence**: Combine Zen analysis with memory queries
3. **Pattern Recognition**: Memory helps identify recurring issues
4. **Decision Tracking**: All consensus outcomes are searchable
5. **Knowledge Sharing**: Export memory for team learning

Your AI development environment now has perfect memory!