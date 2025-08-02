#!/usr/bin/env python3
"""Agent Patterns Analyzer for Claude Code.

Provides intelligent agent recommendations based on task context.
"""

import re
from typing import List, Dict, Any, Optional
from modules.core import Analyzer, PatternMatch


class AgentPatternsAnalyzer(Analyzer):
    """Suggests appropriate agents based on task context."""
    
    def __init__(self):
        super().__init__()
        self.name = "agent_patterns"
        self.priority = 85
        
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
        
    def analyze(self, prompt: str) -> List[PatternMatch]:
        """Analyze prompt and suggest appropriate agents."""
        matches = []
        prompt_lower = prompt.lower()
        
        # Check for full-stack development
        if any(term in prompt_lower for term in ['full stack', 'fullstack', 'web app', 'complete app']):
            pattern = self.swarm_patterns['full_stack']
            matches.append(self._create_swarm_suggestion('full_stack', pattern))
        
        # Check for distributed systems
        elif any(term in prompt_lower for term in ['distributed', 'consensus', 'byzantine', 'raft']):
            pattern = self.swarm_patterns['distributed_system']
            matches.append(self._create_swarm_suggestion('distributed_system', pattern))
        
        # Check for GitHub operations
        elif any(term in prompt_lower for term in ['github', 'pull request', 'pr', 'issue', 'repository']):
            pattern = self.swarm_patterns['github_workflow']
            matches.append(self._create_swarm_suggestion('github_workflow', pattern))
        
        # Check for SPARC/TDD
        elif any(term in prompt_lower for term in ['sparc', 'tdd', 'test driven', 'test-driven']):
            pattern = self.swarm_patterns['sparc_tdd']
            matches.append(self._create_swarm_suggestion('sparc_tdd', pattern))
        
        # Dynamic agent count recommendation
        if 'agent' in prompt_lower or 'swarm' in prompt_lower:
            complexity = self._analyze_complexity(prompt)
            matches.append(PatternMatch(
                analyzer=self.name,
                pattern="agent_count",
                confidence=0.8,
                metadata={
                    "message": f"🎯 AGENT COUNT RECOMMENDATION\n\n"
                               f"Task complexity: {complexity['level']}\n"
                               f"Recommended agents: {complexity['agent_count']}\n\n"
                               f"Distribution:\n"
                               f"- Always include 1 coordinator\n"
                               f"- {complexity['distribution']}",
                    "category": "agent_count",
                    "suggested_count": complexity['agent_count']
                }
            ))
        
        return matches
    
    def _create_swarm_suggestion(self, swarm_type: str, pattern: Dict[str, Any]) -> PatternMatch:
        """Create a swarm suggestion match."""
        agent_list = '\n'.join([f"Task('{agent}', '...', '{agent}')" for agent in pattern['agents']])
        
        return PatternMatch(
            analyzer=self.name,
            pattern=f"{swarm_type}_swarm",
            confidence=0.9,
            metadata={
                "message": f"🐝 {pattern['description'].upper()} ({pattern['count']} agents)\n\n"
                           f"Recommended concurrent deployment:\n"
                           f"```javascript\n"
                           f"{agent_list}\n"
                           f"```",
                "category": "swarm_pattern",
                "agents": pattern['agents'],
                "count": pattern['count']
            }
        )
    
    def _analyze_complexity(self, prompt: str) -> Dict[str, Any]:
        """Analyze task complexity to recommend agent count."""
        components = 0
        
        # Count technical components mentioned
        tech_terms = ['api', 'database', 'auth', 'frontend', 'backend', 'test', 
                      'deploy', 'monitor', 'security', 'performance', 'documentation']
        for term in tech_terms:
            if term in prompt.lower():
                components += 1
        
        # Determine complexity level and agent count
        if components <= 3:
            return {
                "level": "Simple",
                "agent_count": "3-4",
                "distribution": "Focus on core development agents"
            }
        elif components <= 6:
            return {
                "level": "Medium",
                "agent_count": "5-7",
                "distribution": "Balance between coders and support agents"
            }
        else:
            return {
                "level": "Complex",
                "agent_count": "8-12",
                "distribution": "Full team with specialized agents"
            }
    
    def get_agent_info(self, agent_type: str) -> Optional[Dict[str, str]]:
        """Get information about a specific agent type."""
        for category, info in self.agent_catalog.items():
            if agent_type in info['agents']:
                return {
                    "agent": agent_type,
                    "category": category,
                    "description": info['description']
                }
        return None