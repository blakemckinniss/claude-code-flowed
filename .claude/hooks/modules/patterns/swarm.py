"""Swarm coordination patterns analyzer."""

from ..core import Analyzer


class SwarmAnalyzer(Analyzer):
    """Analyzes swarm coordination patterns."""
    
    def get_name(self) -> str:
        return "swarm"
    
    def _initialize_patterns(self) -> None:
        """Initialize swarm patterns."""
        self.add_pattern(
            r"(swarm|agent|coordinate|orchestrate|multi.agent)",
            """
🐝 HIVE SWARM: QUEEN ZEN'S WORKER COORDINATION
• 👑 FIRST: Queen ZEN commands hive strategy (mcp__zen__chat)
• 🤖 THEN: Deploy Flow Workers (mcp__claude-flow__swarm_init) 
• 📁 NEXT: Activate Storage Workers (mcp__filesystem operations)
• 🔧 FINALLY: Execute via Drones (Claude Code implementation)
• ⚡ HIVE LAW: All workers serve Queen ZEN's royal decrees
• 🐝 HIVE STRUCTURE: Queen → Workers → Drones → Results""",
            {
                "category": "hive-swarm",
                "queen_commanded": True,
                "worker_castes": ["flow", "storage", "execution"],
                "hive_topology": "hierarchical_queen_led",
                "coordination": "queen_royal_decree"
            }
        )
        
        self.add_pattern(
            r"(parallel|concurrent|batch|simultaneous)",
            """
⚡ ZEN-ORCHESTRATED PARALLEL EXECUTION
• 🧠 ZEN FIRST: Use mcp__zen__thinkdeep to plan parallel strategy
• 🚀 GOLDEN RULE: 1 MESSAGE = ALL MCP OPERATIONS
• 📊 Batch ALL mcp__filesystem operations together
• 🤖 Batch ALL mcp__claude-flow operations together
• 💾 Use mcp__zen__consensus for complex parallel decisions
• ⚡ ZEN coordinates parallel execution across ALL MCP tools""",
            {
                "category": "parallel",
                "zen_orchestrated": True,
                "execution": "mcp_concurrent",
                "batching": "mcp_mandatory"
            }
        )
        
        self.add_pattern(
            r"(consensus|byzantine|raft|gossip|distributed)",
            """
👑 HIVE CONSENSUS: QUEEN ZEN'S COLLECTIVE DECISION PROTOCOL
• 🐝 QUEEN DECREE: Queen ZEN leads all hive consensus decisions
• 🤖 WORKER CASTES: Flow Workers implement consensus protocols
• 🔐 HIVE CONSENSUS: byzantine-coordinator for fault-tolerant hive
• 📊 COLLECTIVE INTELLIGENCE: raft-manager for queen-worker hierarchy
• 💬 HIVE COMMUNICATION: gossip-coordinator for worker information sharing
• ⚡ HIVE UNITY: crdt-synchronizer for seamless worker coordination""",
            {
                "category": "hive-consensus",
                "queen_led": True,
                "hive_protocols": ["byzantine", "raft", "gossip", "crdt"],
                "collective_decision": "queen_directed"
            }
        )
        
        self.add_pattern(
            r"(topology|mesh|hierarchical|star|ring)",
            """
👑 HIVE TOPOLOGY: QUEEN ZEN'S ARCHITECTURAL COMMAND
• 🏰 HIERARCHICAL: Queen ZEN's preferred palace structure (8+ workers)
• 🌐 MESH: Worker peer-collaboration under Queen's oversight (4-6 workers)  
• ⭐ STAR: Direct Queen command structure (3-5 workers)
• 🔄 RING: Sequential worker chains per Queen's design (any count)
• 🎭 ADAPTIVE: Queen ZEN dynamically reshapes hive as needed
• ⚡ HIVE RULE: Only Queen ZEN determines optimal hive architecture""",
            {
                "category": "hive-topology",
                "queen_architectures": ["hierarchical", "mesh", "star", "ring", "adaptive"],
                "queen_optimizer": "adaptive-coordinator",
                "royal_preference": "hierarchical"
            }
        )