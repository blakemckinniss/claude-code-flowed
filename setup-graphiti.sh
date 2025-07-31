#!/bin/bash
# Graphiti Setup Script for Claude Code + ZEN + Claude Flow Integration

echo "🧠 Graphiti Memory Integration Setup"
echo "===================================="

# Check if script is run from the correct directory
if [ ! -f ".claude/settings.json" ]; then
    echo "❌ Error: Please run this script from the project root (/home/blake/flowed)"
    exit 1
fi

echo ""
echo "📋 Prerequisites:"
echo "1. Neo4j database running (localhost:7687)"
echo "2. OpenAI API key for entity extraction"
echo "3. Python with uv package manager installed"
echo ""

# Prompt for configuration
read -p "Enter the path to Graphiti repository (or press Enter to clone it): " GRAPHITI_PATH
if [ -z "$GRAPHITI_PATH" ]; then
    echo "📥 Cloning Graphiti repository..."
    git clone https://github.com/getzep/graphiti.git /tmp/graphiti
    GRAPHITI_PATH="/tmp/graphiti"
fi

# Check if Neo4j is accessible
read -p "Neo4j URI (default: bolt://localhost:7687): " NEO4J_URI
NEO4J_URI=${NEO4J_URI:-"bolt://localhost:7687"}

read -p "Neo4j username (default: neo4j): " NEO4J_USER
NEO4J_USER=${NEO4J_USER:-"neo4j"}

read -sp "Neo4j password: " NEO4J_PASSWORD
echo ""

read -sp "OpenAI API Key: " OPENAI_API_KEY
echo ""

# Update settings.json with actual paths
echo ""
echo "📝 Updating Claude settings.json..."

# Create a temporary file with the updated configuration
cat > /tmp/graphiti-config.json << EOF
{
  "path": "$GRAPHITI_PATH/mcp_server",
  "neo4j_uri": "$NEO4J_URI",
  "neo4j_user": "$NEO4J_USER",
  "neo4j_password": "$NEO4J_PASSWORD",
  "openai_api_key": "$OPENAI_API_KEY"
}
EOF

# Update the settings.json file using jq
if command -v jq &> /dev/null; then
    # Read the config values
    GRAPHITI_MCP_PATH=$(jq -r '.path' /tmp/graphiti-config.json)
    
    # Update the settings.json
    jq --arg path "$GRAPHITI_MCP_PATH" \
       --arg uri "$NEO4J_URI" \
       --arg user "$NEO4J_USER" \
       --arg pass "$NEO4J_PASSWORD" \
       --arg key "$OPENAI_API_KEY" \
       '.mcpServers["graphiti-memory"].args[2] = $path |
        .mcpServers["graphiti-memory"].env.NEO4J_URI = $uri |
        .mcpServers["graphiti-memory"].env.NEO4J_USER = $user |
        .mcpServers["graphiti-memory"].env.NEO4J_PASSWORD = $pass |
        .mcpServers["graphiti-memory"].env.OPENAI_API_KEY = $key' \
       .claude/settings.json > .claude/settings.json.tmp && \
    mv .claude/settings.json.tmp .claude/settings.json
else
    echo "⚠️  jq not found. Please manually update the paths in .claude/settings.json"
    echo "   Update the graphiti-memory section with:"
    echo "   - Path: $GRAPHITI_PATH/mcp_server"
    echo "   - Neo4j credentials"
    echo "   - OpenAI API key"
fi

# Make hooks executable
echo ""
echo "🔧 Making hooks executable..."
chmod +x .claude/hooks/*.py

# Install Graphiti dependencies
echo ""
echo "📦 Installing Graphiti dependencies..."
cd "$GRAPHITI_PATH/mcp_server"
if command -v uv &> /dev/null; then
    uv sync
else
    echo "⚠️  uv not found. Please install uv or manually install dependencies in $GRAPHITI_PATH/mcp_server"
fi

# Test Neo4j connection
echo ""
echo "🔍 Testing Neo4j connection..."
python3 -c "
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USER', '$NEO4J_PASSWORD'))
    driver.verify_connectivity()
    print('✅ Neo4j connection successful!')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j connection failed: {e}')
" 2>/dev/null || echo "⚠️  Could not test Neo4j connection. Make sure Neo4j is running."

# Cleanup
rm -f /tmp/graphiti-config.json

echo ""
echo "✨ Setup complete!"
echo ""
echo "📌 Next steps:"
echo "1. Ensure Neo4j is running at $NEO4J_URI"
echo "2. Restart Claude Code to load the new configuration"
echo "3. The following hooks are now active:"
echo "   - SessionStart: Zen MCP instructions injection"
echo "   - UserPromptSubmit: Zen suggestions + Graphiti memory capture"
echo "   - PostToolUse: Zen tool output capture to Graphiti"
echo ""
echo "🎯 Usage examples:"
echo "   - 'Use zen debug to find the issue' → Automatically saved to memory"
echo "   - 'Search my memory for authentication bugs' → Query past solutions"
echo "   - 'Get consensus on API design' → Multi-model perspectives saved"
echo ""
echo "🧘 Your AI development environment is now memory-enhanced!"