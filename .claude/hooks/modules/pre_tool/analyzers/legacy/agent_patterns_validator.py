#!/usr/bin/env python3
"""Agent Patterns Validator for Claude Code.

Provides intelligent agent recommendations based on task context.
"""

import re
from typing import Dict, Any, Optional, List
from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class AgentPatternsValidator(HiveWorkflowValidator):
    """Suggests appropriate agents based on task context."""
    
    def get_validator_name(self) -> str:
        return "agent_patterns_validator"
    
    def __init__(self, priority: int = 85):
        super().__init__(priority)
        
        # Complete agent catalog
        self.agent_catalog = {
            "core_development": {
                "agents": ["coder", "reviewer", "tester", "planner", "researcher"],
                "description": "Core development team"
            },
            "swarm_coordination": {
                "agents": ["hierarchical-coordinator", "mesh-coordinator", "adaptive-coordinator", 
                          "collective-intelligence-coordinator", "swarm-memory-manager"],
                "description": "Swarm coordination specialists"
            },
            "consensus_distributed": {
                "agents": ["byzantine-coordinator", "raft-manager", "gossip-coordinator",
                          "consensus-builder", "crdt-synchronizer", "quorum-manager", "security-manager"],
                "description": "Distributed systems consensus"
            },
            "performance_optimization": {
                "agents": ["perf-analyzer", "performance-benchmarker", "task-orchestrator",
                          "memory-coordinator", "smart-agent"],
                "description": "Performance and optimization"
            },
            "github_repository": {
                "agents": ["github-modes", "pr-manager", "code-review-swarm", "issue-tracker",
                          "release-manager", "workflow-automation", "project-board-sync",
                          "repo-architect", "multi-repo-swarm"],
                "description": "GitHub repository management"
            },
            "sparc_methodology": {
                "agents": ["sparc-coord", "sparc-coder", "specification", "pseudocode",
                          "architecture", "refinement"],
                "description": "SPARC methodology specialists"
            },
            "specialized_development": {
                "agents": ["backend-dev", "mobile-dev", "ml-developer", "cicd-engineer",
                          "api-docs", "system-architect", "code-analyzer", "base-template-generator"],
                "description": "Specialized development roles"
            }
        }
        
        # Predefined swarm patterns
        self.swarm_patterns = {
            "full_stack": {
                "agents": ["system-architect", "backend-dev", "mobile-dev", "coder",
                          "api-docs", "cicd-engineer", "performance-benchmarker", "production-validator"],
                "count": 8,
                "description": "Full-stack development swarm"
            },
            "distributed_system": {
                "agents": ["byzantine-coordinator", "raft-manager", "gossip-coordinator",
                          "crdt-synchronizer", "security-manager", "perf-analyzer"],
                "count": 6,
                "description": "Distributed system swarm"
            },
            "github_workflow": {
                "agents": ["pr-manager", "code-review-swarm", "issue-tracker",
                          "release-manager", "workflow-automation"],
                "count": 5,
                "description": "GitHub workflow swarm"
            },
            "sparc_tdd": {
                "agents": ["specification", "pseudocode", "architecture", "sparc-coder",
                          "tdd-london-swarm", "refinement", "production-validator"],
                "count": 7,
                "description": "SPARC TDD swarm"
            }
        }
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate and suggest agent patterns."""
        
        # Only provide suggestions for Task or swarm-related operations
        if tool_name not in ["Task", "mcp__claude-flow__swarm_init", "mcp__claude-flow__task_orchestrate"]:
            return None
        
        # Check if we're spawning agents without proper swarm initialization
        if tool_name == "Task" and context.get_tools_since_flow() > 5:
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.MISSING_COORDINATION,
                message="🐝 Consider initializing a swarm for better agent coordination",
                suggested_alternative="Use mcp__claude-flow__swarm_init before spawning multiple agents",
                hive_guidance=self._get_swarm_suggestion_based_on_context(tool_input),
                priority_score=70
            )
        
        # For swarm initialization, suggest appropriate agent count
        if tool_name == "mcp__claude-flow__swarm_init":
            agent_count = tool_input.get("maxAgents", 5)
            if agent_count < 3:
                return ValidationResult(
                    severity=ValidationSeverity.SUGGEST,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message="🎯 Consider using more agents for complex tasks",
                    suggested_alternative="Increase maxAgents to 5-8 for better task distribution",
                    hive_guidance=self._get_agent_count_recommendation(tool_input),
                    priority_score=60
                )
        
        return None
    
    def _get_swarm_suggestion_based_on_context(self, tool_input: Dict[str, Any]) -> str:
        """Get swarm suggestion based on task context."""
        task_desc = str(tool_input.get("prompt", "")).lower()
        
        # Detect task type and suggest appropriate swarm
        if any(term in task_desc for term in ['full stack', 'fullstack', 'web app', 'complete app']):
            pattern = self.swarm_patterns['full_stack']
            return f"🐝 FULL-STACK SWARM ({pattern['count']} agents):\n" + \
                   "\n".join([f"  Task('{agent}', '...', '{agent}')" for agent in pattern['agents']])
        
        elif any(term in task_desc for term in ['distributed', 'consensus', 'byzantine', 'raft']):
            pattern = self.swarm_patterns['distributed_system']
            return f"🐝 DISTRIBUTED SYSTEM SWARM ({pattern['count']} agents):\n" + \
                   "\n".join([f"  Task('{agent}', '...', '{agent}')" for agent in pattern['agents']])
        
        elif any(term in task_desc for term in ['github', 'pull request', 'pr', 'issue']):
            pattern = self.swarm_patterns['github_workflow']
            return f"🐝 GITHUB WORKFLOW SWARM ({pattern['count']} agents):\n" + \
                   "\n".join([f"  Task('{agent}', '...', '{agent}')" for agent in pattern['agents']])
        
        else:
            return "🎯 AGENT COUNT RECOMMENDATION:\n" + \
                   "  Simple tasks (1-3 components): 3-4 agents\n" + \
                   "  Medium tasks (4-6 components): 5-7 agents\n" + \
                   "  Complex tasks (7+ components): 8-12 agents"
    
    def _get_agent_count_recommendation(self, tool_input: Dict[str, Any]) -> str:
        """Get agent count recommendation."""
        return """🎯 DYNAMIC AGENT COUNT RULES:
1. Check CLI Arguments First: If user runs `npx claude-flow@alpha --agents 5`, use 5 agents
2. Auto-Decide if No Args: Analyze task complexity:
   - Simple tasks: 3-4 agents
   - Medium tasks: 5-7 agents  
   - Complex tasks: 8-12 agents
3. Agent Type Distribution:
   - Always include 1 coordinator
   - Code-heavy tasks: more coders
   - Design tasks: more architects/analysts
   - Quality tasks: more testers/reviewers"""