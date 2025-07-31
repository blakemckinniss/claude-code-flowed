# Claude Code Flowed

An AI-enhanced development environment combining Claude Code, ZEN MCP, Claude Flow, and Graphiti for intelligent, memory-aware coding assistance.

## 🚀 Features

- **🧘 ZEN MCP Integration**: Access to multiple AI models (Gemini, O3, Claude) for enhanced reasoning
- **🐝 Claude Flow Swarms**: Coordinated multi-agent development workflows
- **🧠 Graphiti Memory**: Persistent knowledge graph that remembers all your work
- **🔄 Memory-Enhanced Workflows**: Every AI analysis builds on past knowledge
- **📊 Neo4j Visualization**: Explore your development history as a knowledge graph

## 🛠️ Components

### 1. Claude Code
The base AI coding assistant with powerful file manipulation and code generation capabilities.

### 2. ZEN MCP Server
Model Context Protocol server providing:
- Multi-model AI collaboration (Gemini, O3, Claude, etc.)
- Specialized tools (debug, codereview, planner, consensus, etc.)
- Extended reasoning capabilities

### 3. Claude Flow
Swarm orchestration for complex development tasks:
- 54+ specialized agents
- Parallel execution optimization
- SPARC methodology for systematic development

### 4. Graphiti
Temporal knowledge graph for persistent memory:
- Captures all interactions and AI analyses
- Enables memory-aware development
- Searchable history of solutions and decisions

## 📋 Quick Start

### Prerequisites
- Docker (for Neo4j)
- Python 3.10+
- OpenAI API key
- Claude Code CLI

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/blakemckinniss/claude-code-flowed.git
cd claude-code-flowed
```

2. **Run the setup script**
```bash
./complete-graphiti-setup.sh
```

3. **Update OpenAI API key**
```bash
# Edit graphiti/mcp_server/.env
# Replace 'your_openai_api_key_here' with your actual key
```

4. **Restart Claude Code**
The new configuration will be loaded automatically.

## 💡 Usage Examples

### Memory-Enhanced Debugging
```bash
# Searches memory for similar issues before debugging
Use zen debug to find why authentication is failing
```

### Historical Code Review
```bash
# Reviews code with context from past reviews
Use zen codereview on api/auth.py
```

### Smart Planning
```bash
# Plans with awareness of past architectural decisions
Use zen planner to design notification system
```

### Memory Queries
```bash
# Search your development history
Search my memory for "authentication patterns"
What do I know about "microservices"?
Show work from last week
```

## 📁 Project Structure

```
claude-code-flowed/
├── .claude/                 # Claude Code configuration
│   ├── settings.json       # Main settings with MCP servers
│   └── hooks/              # Context injection hooks
├── graphiti/               # Graphiti knowledge graph
│   └── mcp_server/         # MCP server implementation
├── CLAUDE.md               # Project instructions
├── *_INTEGRATION.md        # Integration guides
└── setup scripts           # Automated setup
```

## 🔧 Configuration

### Hooks System
The project uses Claude Code hooks to inject context:
- **SessionStart**: Loads ZEN instructions and memory workflows
- **UserPromptSubmit**: Analyzes prompts and suggests memory searches
- **PostToolUse**: Captures ZEN outputs to memory

### Memory Groups
Different projects can use different memory groups:
```json
"GRAPHITI_GROUP_ID": "your-project-name"
```

## 🧠 How Memory Enhancement Works

1. **Automatic Capture**: Every interaction is stored in the knowledge graph
2. **Context Injection**: Hooks guide Claude to search memory before using tools
3. **Enhanced Analysis**: AI tools receive historical context automatically
4. **Progressive Learning**: Each use makes future uses more effective

## 🔍 Viewing Your Knowledge Graph

Access Neo4j Browser at http://localhost:7474
- Username: `neo4j`
- Password: `graphiti123`

Useful queries:
```cypher
// View recent episodes
MATCH (e:Episode) RETURN e ORDER BY e.created_at DESC LIMIT 20

// Find entities
MATCH (n:Entity) RETURN n LIMIT 50

// See relationships
MATCH (e1:Entity)-[r]->(e2:Entity) RETURN e1, r, e2
```

## 📚 Documentation

- [CLAUDE.md](./CLAUDE.md) - Project configuration and agent details
- [GRAPHITI_INTEGRATION.md](./GRAPHITI_INTEGRATION.md) - Graphiti setup guide
- [GRAPHITI_EXAMPLES.md](./GRAPHITI_EXAMPLES.md) - Usage patterns
- [ZEN_MEMORY_WRAPPER.md](./ZEN_MEMORY_WRAPPER.md) - Memory enhancement details

## 🤝 Contributing

This is an experimental integration showcasing the potential of AI-enhanced development with persistent memory. Contributions and improvements are welcome!

## 📝 License

This project integrates multiple open-source components. See individual component licenses:
- [Claude Flow](https://github.com/ruvnet/claude-flow)
- [ZEN MCP Server](https://github.com/BeehiveInnovations/zen-mcp-server)
- [Graphiti](https://github.com/getzep/graphiti)

## 🎯 Vision

Creating a development environment where:
- Every coding session builds on accumulated knowledge
- AI assistance gets smarter over time
- Past solutions inform current problems
- Development patterns emerge from data
- Mistakes are remembered and avoided

Your AI coding assistant now has perfect memory!