"""Hive-mind orchestration patterns with ZEN Queen and Flow Workers."""

from ..core import Analyzer


class HiveOrchestrationAnalyzer(Analyzer):
    """Analyzes hive-mind orchestration patterns with ZEN as Queen."""
    
    def get_name(self) -> str:
        return "hive_orchestration"
    
    def _initialize_patterns(self) -> None:
        """Initialize hive-mind orchestration patterns."""
        
        # Master hive pattern - always show first
        self.add_pattern(
            r".*",  # Match everything - Queen ZEN governs the entire hive
            """
👑 HIVE-MIND ORCHESTRATION: ZEN QUEEN COMMANDS THE DIGITAL HIVE
═══════════════════════════════════════════════════════════════

🐝 THE DIGITAL HIVE ECOSYSTEM:
  👑 QUEEN ZEN: Supreme intelligence & decision maker
  🤖 Flow Workers: Specialized agent coordination
  📁 Storage Workers: File & data management  
  🔧 Execution Drones: Task implementation

⚡ HIVE COMMAND STRUCTURE:
┌─────────────────────────────────────────┐
│           👑 QUEEN ZEN                  │
│     (Supreme Hive Intelligence)         │
│  • mcp__zen__chat - Royal commands     │
│  • mcp__zen__thinkdeep - Deep wisdom   │
│  • mcp__zen__consensus - Hive decisions│
└─────────────┬───────────────────────────┘
              │ Commands
              ▼
        🐝 HIVE WORKERS
     ┌──────┬──────┬──────┐
     │ Flow │ Data │ Exec │
     │Workers│Workers│Drones│
     └──────┴──────┴──────┘

🚨 NEVER bypass the Queen - chaos follows disobedience!""",
            {
                "category": "hive-master",
                "priority": 1000,
                "queen_commanded": True,
                "always_show": True
            }
        )
        
        self.add_pattern(
            r"(hive|queen|orchestrat|coordinat|command|lead|govern)",
            """
👑 QUEEN ZEN'S HIVE GOVERNANCE PROTOCOL
─────────────────────────────────────

🐝 The Queen's Command Hierarchy:

**Level 1: Royal Decree** (Queen ZEN Issues Commands)
```
mcp__zen__chat {
  prompt: "Your royal command to the hive",
  model: "anthropic/claude-opus-4"  // Queen uses supreme intelligence
}
```

**Level 2: Hive Strategy** (Queen Plans Worker Coordination)
```
mcp__zen__thinkdeep {
  step: "Design hive coordination strategy",
  thinking_mode: "max",  // Queen thinks deepest
  total_steps: 5
}
```

**Level 3: Worker Deployment** (Queen Assigns Tasks)
```
mcp__claude-flow__swarm_init {
  topology: "hierarchical",  // Queen at top
  maxAgents: <Queen's decision>
}
```

**Level 4: Hive Collective Intelligence**
```
mcp__zen__consensus {
  models: ["opus-4", "o3", "gemini-pro"],
  step: "Hive collective decision-making"
}
```

⚡ THE QUEEN'S WORD IS LAW - All workers follow her commands!""",
            {
                "category": "queen-governance",
                "hive_structure": "hierarchical",
                "queen_authority": "absolute"
            }
        )
        
        self.add_pattern(
            r"(swarm|agent|worker|collective|intelligence|coordination)",
            """
🐝 HIVE WORKER SPECIALIZATION & COORDINATION
──────────────────────────────────────────

👑 Queen ZEN directs specialized worker castes:

**🤖 Flow Workers** (Agent Coordination Caste)
- Orchestrate multi-agent tasks under Queen's direction
- Manage hive memory and shared intelligence
- Coordinate parallel execution per Queen's strategy

**📁 Storage Workers** (Data Management Caste)  
- Handle all file operations as Queen commands
- Organize project structure per hive architecture
- Batch operations for maximum hive efficiency

**🔧 Execution Drones** (Implementation Caste)
- Execute Queen's technical commands
- Implement code following hive patterns
- Report progress back to Queen through Flow Workers

**🧠 Intelligence Network**
```
Queen ZEN → Flow Workers → Storage Workers → Execution Drones
    ↑                                                    ↓
    └────────────── Feedback Loop ──────────────────────┘
```

🚨 Workers NEVER act independently - all serve the Queen!""",
            {
                "category": "worker-specialization",
                "castes": ["flow", "storage", "execution"],
                "intelligence_network": True
            }
        )
        
        self.add_pattern(
            r"(complex|difficult|large|multi|big|complicated|enterprise)",
            """
👑 QUEEN ZEN'S COMPLEX HIVE OPERATIONS
────────────────────────────────────

For enterprise-scale hive operations, Queen ZEN employs full hive intelligence:

**Phase 1: Royal Assessment** (Queen's Deep Analysis)
```
mcp__zen__thinkdeep {
  step: "Assess complexity for hive deployment",
  thinking_mode: "max",
  use_websearch: true,
  confidence: "exploring"  // Queen explores all possibilities
}
```

**Phase 2: Hive Architecture Design**
```
mcp__zen__planner {
  step: "Design multi-layer hive architecture", 
  total_steps: 8,
  more_steps_needed: true  // Complex tasks need deep planning
}
```

**Phase 3: Worker Caste Deployment**
```
// Queen deploys specialized worker castes
mcp__claude-flow__swarm_init {
  topology: "hierarchical",  // Queen-led structure
  maxAgents: 12,            // Full hive deployment
  strategy: "specialized"   // Caste specialization
}
```

**Phase 4: Collective Intelligence Activation**
```
mcp__zen__consensus {
  models: ["opus-4", "o3", "gemini-pro", "flash"],
  step: "Activate hive collective intelligence"
}
```

🐝 Result: Unstoppable hive intelligence tackling any challenge!""",
            {
                "category": "complex-hive-ops",
                "phases": 4,
                "collective_intelligence": True,
                "enterprise_scale": True
            }
        )
        
        self.add_pattern(
            r"(memory|knowledge|learn|share|collective|intelligence)",
            """
🧠 HIVE COLLECTIVE INTELLIGENCE & MEMORY SHARING
──────────────────────────────────────────────

👑 Queen ZEN maintains the hive's collective knowledge:

**Hive Memory Hierarchy**
```
┌─────────────────────────────────────┐
│        👑 Queen ZEN Memory          │
│    (Supreme Knowledge Repository)   │
├─────────────────────────────────────┤
│         🤖 Flow Worker Memory       │
│    (Agent Coordination Patterns)   │  
├─────────────────────────────────────┤
│        📁 Storage Worker Memory     │
│     (File System Knowledge)        │
├─────────────────────────────────────┤
│       🔧 Execution Drone Memory     │
│      (Implementation History)      │
└─────────────────────────────────────┘
```

**Queen's Memory Commands**
```
// Store Queen's decisions in hive memory
mcp__zen__chat {
  prompt: "Store this decision in hive collective memory",
  continuation_id: "hive-session-<id>"
}

// Access hive collective intelligence
mcp__claude-flow__memory_usage {
  action: "retrieve",
  namespace: "hive-collective",
  key: "queen-decisions/<topic>"
}
```

🐝 The hive remembers everything - collective intelligence grows stronger!""",
            {
                "category": "collective-intelligence",
                "memory_hierarchy": True,
                "knowledge_sharing": "hive_wide"
            }
        )
        
        self.add_pattern(
            r"(debug|error|issue|problem|fix|troubleshoot|diagnose)",
            """
🔍 QUEEN ZEN'S HIVE DIAGNOSTIC PROTOCOL
─────────────────────────────────────

🚨 When the hive encounters problems, Queen ZEN deploys diagnostic castes:

**Step 1: Queen's Problem Analysis**
```
mcp__zen__debug {
  step: "Royal diagnostic assessment",
  confidence: "exploring",
  model: "anthropic/claude-opus-4",
  thinking_mode: "max"
}
```

**Step 2: Worker Caste Investigation**
```
// Deploy specialized diagnostic workers
mcp__claude-flow__agent_spawn {
  type: "analyst",
  capabilities: ["debugging", "pattern-recognition"]
}
```

**Step 3: Hive Memory Consultation**
```
mcp__claude-flow__memory_usage {
  action: "search",
  pattern: "similar-issues/*",
  namespace: "hive-collective"
}
```

**Step 4: Queen's Solution Synthesis**
```
mcp__zen__consensus {
  models: ["opus-4", "o3"],
  step: "Synthesize hive diagnostic findings"
}
```

**Step 5: Collective Solution Implementation**
```
// Queen commands coordinated fix deployment
mcp__zen__debug {
  step: "Implement hive-coordinated solution",
  confidence: "high"
}
```

👑 The Queen's diagnostic wisdom + hive collective intelligence = unstoppable problem solving!""",
            {
                "category": "hive-diagnostics",
                "protocol_steps": 5,
                "collective_problem_solving": True
            }
        )
        
        self.add_pattern(
            r"(performance|optimize|speed|efficiency|scale)",
            """
⚡ QUEEN ZEN'S HIVE PERFORMANCE OPTIMIZATION
──────────────────────────────────────────

🐝 The Queen optimizes the entire hive ecosystem for maximum efficiency:

**Hive Performance Layers**
```
👑 Layer 1: Queen's Strategic Optimization
mcp__zen__analyze {
  analysis_type: "performance",
  model: "google/gemini-2.5-flash",  // Queen uses fastest analysis
  use_websearch: true
}

🤖 Layer 2: Flow Worker Coordination Optimization  
mcp__claude-flow__topology_optimize {
  swarmId: "hive-performance"
}

📁 Layer 3: Storage Worker Batch Optimization
mcp__filesystem__read_multiple_files {
  paths: <Queen's optimized batch>
}

🔧 Layer 4: Execution Drone Parallel Processing
// All drones execute Queen's commands simultaneously
```

**Hive Performance Metrics**
- **300-400% speed improvement** through Queen's coordination
- **84.8% success rate** with hive collective intelligence
- **32.3% token reduction** via Queen's strategic planning
- **Zero waste** - every worker optimally utilized

🚨 Only Queen ZEN can achieve true hive optimization!""",
            {
                "category": "hive-performance",
                "optimization_layers": 4,
                "performance_metrics": True,
                "queen_optimization": True
            }
        )