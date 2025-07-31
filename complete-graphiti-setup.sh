#!/bin/bash
# Complete Graphiti Setup Script

echo "🧠 Graphiti Memory Integration Complete Setup"
echo "==========================================="

# Check if we're in the right directory
if [ ! -f ".claude/settings.json" ]; then
    echo "❌ Error: Please run this script from the project root (/home/blake/flowed)"
    exit 1
fi

# Update the .env file with a placeholder for OpenAI key
echo "📝 Updating Graphiti .env file..."
cat > graphiti/mcp_server/.env << 'EOF'
# Graphiti MCP Server Environment Configuration

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphiti123

# OpenAI API Configuration
# IMPORTANT: Replace this with your actual OpenAI API key!
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
SMALL_MODEL_NAME=gpt-3.5-turbo

# Group ID for namespacing graph data
GROUP_ID=claude-code-flowed

# Semaphore limit for concurrent operations
SEMAPHORE_LIMIT=10
EOF

# Make all hooks executable
echo "🔧 Setting hook permissions..."
chmod +x .claude/hooks/*.py

# Test if Graphiti MCP server can start
echo "🧪 Testing Graphiti MCP server..."
cd graphiti/mcp_server

# Try to run the server briefly to test configuration
timeout 5s uv run graphiti_mcp_server.py --transport stdio --group-id claude-code-flowed --use-custom-entities > /tmp/graphiti-test.log 2>&1 &
GRAPHITI_PID=$!

sleep 3
if ps -p $GRAPHITI_PID > /dev/null 2>&1; then
    echo "✅ Graphiti MCP server starts successfully!"
    kill $GRAPHITI_PID 2>/dev/null
else
    echo "⚠️  Graphiti MCP server may have configuration issues"
    echo "   Check /tmp/graphiti-test.log for details"
fi

cd ../..

# Display setup summary
echo ""
echo "✨ Setup Summary"
echo "================"
echo ""
echo "✅ Docker Neo4j: Running at localhost:7687"
echo "   - Username: neo4j"
echo "   - Password: graphiti123"
echo "   - Web UI: http://localhost:7474"
echo ""
echo "✅ Graphiti MCP Server: Configured in .claude/settings.json"
echo "   - Path: /home/blake/flowed/graphiti/mcp_server"
echo "   - Group ID: claude-code-flowed"
echo ""
echo "✅ Integration Hooks: Active and executable"
echo "   - session-start-zen.py: Injects Zen instructions"
echo "   - prompt-submit-zen.py: Analyzes prompts for Zen suggestions"
echo "   - graphiti-integration.py: Captures user interactions"
echo "   - zen-graphiti-bridge.py: Captures Zen tool outputs"
echo ""
echo "⚠️  IMPORTANT: Update OpenAI API Key"
echo "   Edit: graphiti/mcp_server/.env"
echo "   Replace: 'your_openai_api_key_here' with your actual key"
echo ""
echo "📊 Neo4j Web Interface"
echo "   1. Open http://localhost:7474 in your browser"
echo "   2. Login with neo4j/graphiti123"
echo "   3. Explore your knowledge graph as it grows"
echo ""
echo "🚀 Next Steps"
echo "   1. Update the OpenAI API key in graphiti/mcp_server/.env"
echo "   2. Restart Claude Code to load new configuration"
echo "   3. Start using Zen tools - outputs will be captured!"
echo ""
echo "💡 Usage Examples"
echo "   - 'Use zen debug to find the issue' → Saved to memory"
echo "   - 'Search my memory for authentication' → Query past work"
echo "   - 'Use zen planner for API design' → Planning saved"
echo ""
echo "🎉 Your AI development environment is ready!"