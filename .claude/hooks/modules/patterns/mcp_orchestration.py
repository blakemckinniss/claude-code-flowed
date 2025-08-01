"""MCP orchestration patterns analyzer with ZEN emphasis."""

from ..core import Analyzer


class MCPOrchestrationAnalyzer(Analyzer):
    """Analyzes MCP orchestration patterns with ZEN as master orchestrator."""
    
    def get_name(self) -> str:
        return "mcp_orchestration"
    
    def _initialize_patterns(self) -> None:
        """Initialize MCP orchestration patterns."""
        
        # Master orchestration pattern - always show first
        self.add_pattern(
            r".*",  # Match everything - MCP ZEN is always relevant
            """
👑 MCP ZEN HIVE: QUEEN ORCHESTRATOR OF THE DIGITAL ECOSYSTEM
═══════════════════════════════════════════════════════════

🐝 QUEEN ZEN commands the entire MCP hive:
  ├── 👑 Queen ZEN: Supreme hive intelligence & orchestrator
  ├── 🤖 Flow Workers: Swarm coordination & hive memory
  ├── 📁 Storage Workers: File operations & data management  
  └── 🔧 Execution Drones: Task implementation & results

⚡ HIVE COMMAND CHAIN: QUEEN → WORKERS → DRONES → RESULTS
1️⃣ QUEEN ZEN issues royal commands & hive strategy
2️⃣ FLOW WORKERS coordinate agent castes & hive memory
3️⃣ STORAGE WORKERS manage the hive's data ecosystem  
4️⃣ EXECUTION DRONES implement Queen's royal decrees

🚨 HIVE LAW: Always begin with Queen ZEN's royal commands!""",
            {
                "category": "mcp-master",
                "priority": 1000,
                "always_show": True
            }
        )
        
        self.add_pattern(
            r"(mcp|orchestrat|coordinat|zen|flow|filesystem)",
            """
🔄 MCP TOOL HIERARCHY & RELATIONSHIPS
────────────────────────────────────

📊 Tool Usage Pattern:
┌─────────────────────────────────────┐
│         🧠 MCP ZEN (Master)         │
│  • mcp__zen__chat - Strategic planning │
│  • mcp__zen__thinkdeep - Deep analysis │
│  • mcp__zen__consensus - Decisions    │
└─────────────┬───────────────────────┘
              │ Orchestrates
┌─────────────▼───────────────────────┐
│      🤖 Claude Flow MCP (Coord)     │
│  • mcp__claude-flow__swarm_init     │
│  • mcp__claude-flow__agent_spawn    │
│  • mcp__claude-flow__memory_usage   │
└─────────────┬───────────────────────┘
              │ Manages
┌─────────────▼───────────────────────┐
│      📁 Filesystem MCP (Storage)    │
│  • mcp__filesystem__read_file       │
│  • mcp__filesystem__write_file      │
│  • mcp__filesystem__list_directory  │
└─────────────┬───────────────────────┘
              │ Provides to
┌─────────────▼───────────────────────┐
│      🔧 Claude Code (Execution)     │
│  • Read, Write, Edit, Bash          │
│  • TodoWrite, Task, Grep, Glob      │
└─────────────────────────────────────┘""",
            {
                "category": "mcp-hierarchy",
                "emphasis": "tool-relationships"
            }
        )
        
        self.add_pattern(
            r"(plan|strategy|approach|think|analyze|design)",
            """
🧠 ZEN-FIRST ORCHESTRATION APPROACH
───────────────────────────────────

⚡ MANDATORY: Start with MCP ZEN for strategic planning:

1️⃣ **Initial Analysis** (ALWAYS FIRST):
   ```
   mcp__zen__thinkdeep {
     step: "Analyze requirements and approach",
     thinking_mode: "high",
     model: "anthropic/claude-opus-4"
   }
   ```

2️⃣ **Swarm Coordination** (After ZEN planning):
   ```
   mcp__claude-flow__swarm_init {
     topology: "hierarchical",  // ZEN recommended
     maxAgents: 8              // ZEN calculated
   }
   ```

3️⃣ **File Operations** (Through filesystem MCP):
   ```
   mcp__filesystem__read_multiple_files {
     paths: [...]  // Paths identified by ZEN
   }
   ```

🚨 NEVER skip ZEN orchestration - it prevents:
• Uncoordinated agent chaos
• Inefficient file operations
• Missing strategic insights
• Poor resource utilization""",
            {
                "category": "zen-first",
                "workflow": "strategic"
            }
        )
        
        self.add_pattern(
            r"(complex|difficult|large|multi|big|complicated)",
            """
🎯 COMPLEX TASK MCP ORCHESTRATION
─────────────────────────────────

For complex tasks, use ZEN's multi-layer orchestration:

📋 **Step 1: ZEN Strategic Analysis**
```
mcp__zen__thinkdeep {
  step: "Break down complex requirements",
  total_steps: 5,
  thinking_mode: "max",
  use_websearch: true
}
```

📊 **Step 2: ZEN Consensus Building**
```
mcp__zen__consensus {
  models: ["opus-4", "o3", "gemini-pro"],
  step: "Evaluate architectural options"
}
```

🤖 **Step 3: Flow Swarm Setup** (Based on ZEN)
```
mcp__claude-flow__swarm_init { 
  topology: <ZEN's recommendation>,
  strategy: "parallel"
}
```

📁 **Step 4: Filesystem Preparation**
```
mcp__filesystem__create_directory
mcp__filesystem__list_directory
mcp__filesystem__read_multiple_files
```

⚡ ZEN coordinates ALL layers for optimal results!""",
            {
                "category": "complex-orchestration",
                "layers": 4
            }
        )
        
        self.add_pattern(
            r"(debug|error|issue|problem|fix|troubleshoot)",
            """
🔍 ZEN DEBUG ORCHESTRATION WORKFLOW
──────────────────────────────────

🚨 For debugging, ZEN provides systematic orchestration:

1️⃣ **ZEN Debug Analysis**:
   ```
   mcp__zen__debug {
     step: "Identify issue patterns",
     confidence: "exploring",
     model: "anthropic/claude-opus-4"
   }
   ```

2️⃣ **Filesystem Investigation** (ZEN-directed):
   ```
   mcp__filesystem__search_files {
     pattern: <ZEN identified>,
     excludePatterns: [".git", "node_modules"]
   }
   ```

3️⃣ **Flow Memory Check**:
   ```
   mcp__claude-flow__memory_usage {
     action: "retrieve",
     key: "debug/<issue-type>"
   }
   ```

4️⃣ **ZEN Solution Synthesis**:
   ```
   mcp__zen__debug {
     step: "Synthesize findings",
     confidence: "high"
   }
   ```

💡 ZEN ensures systematic debugging across all tools!""",
            {
                "category": "debug-orchestration",
                "systematic": True
            }
        )
        
        self.add_pattern(
            r"(performance|optimize|speed|fast|efficient)",
            """
⚡ ZEN PERFORMANCE ORCHESTRATION
────────────────────────────────

🚀 ZEN optimizes performance across all MCP layers:

**Layer 1: ZEN Analysis**
```
mcp__zen__analyze {
  analysis_type: "performance",
  model: "google/gemini-2.5-flash"  // Fast analysis
}
```

**Layer 2: Parallel Filesystem Ops**
```
// ZEN coordinates batch operations
mcp__filesystem__read_multiple_files {
  paths: [...],  // All at once per ZEN
}
```

**Layer 3: Flow Agent Optimization**
```
mcp__claude-flow__topology_optimize {
  swarmId: "perf-swarm"
}
```

**Layer 4: ZEN Bottleneck Analysis**
```
mcp__zen__tracer {
  trace_mode: "precision",
  target_description: "Performance bottlenecks"
}
```

📊 Result: 300-400% performance improvement!""",
            {
                "category": "performance-orchestration",
                "optimization": "multi-layer"
            }
        )
        
        self.add_pattern(
            r"(file|directory|folder|path|read|write)",
            """
📁 ZEN-FILESYSTEM MCP COORDINATION
─────────────────────────────────

🎯 ZEN orchestrates ALL filesystem operations:

**Pattern 1: Batch Reading** (ZEN-optimized)
```
// First: ZEN identifies files
mcp__zen__chat { 
  prompt: "What files need analysis?"
}

// Then: Filesystem batch read
mcp__filesystem__read_multiple_files {
  paths: <ZEN's recommendations>
}
```

**Pattern 2: Smart Writing** (ZEN-structured)
```
// ZEN plans file structure
mcp__zen__planner {
  step: "Design project structure"
}

// Filesystem executes plan
mcp__filesystem__create_directory
mcp__filesystem__write_file
```

**Pattern 3: Search & Analysis**
```
// ZEN defines search strategy
mcp__zen__thinkdeep {
  step: "Identify search patterns"
}

// Filesystem searches
mcp__filesystem__search_files {
  pattern: <ZEN's pattern>
}
```

⚡ Never use filesystem MCP without ZEN coordination!""",
            {
                "category": "filesystem-coordination",
                "zen_required": True
            }
        )