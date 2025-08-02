#!/usr/bin/env python3
"""Hive Coordination System for Multi-Agent Orchestration.

Enhances existing ZEN infrastructure with sophisticated hive coordination
capabilities, including Queen selection, worker scaling, and collective
intelligence coordination.
"""

import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

# Import existing infrastructure
from .zen_consultant import ZenConsultant, ComplexityLevel, CoordinationType, AgentAllocation
from .shared_types import ProjectContext, TechStack


class HiveRole(Enum):
    """Hive-specific roles for multi-agent coordination."""
    QUEEN = "queen"
    WORKER = "worker"
    SCOUT = "scout"
    GUARDIAN = "guardian"


class ConsensusType(Enum):
    """Types of consensus mechanisms for collective decision-making."""
    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    WEIGHTED = "weighted"
    QUEEN_DECISION = "queen_decision"


@dataclass
class HiveAgent:
    """Enhanced agent specification for hive coordination."""
    id: str
    role: HiveRole
    specialist_type: str
    capabilities: List[str]
    priority: int = 1  # 1-5, higher is more important
    active: bool = True
    workload: int = 0  # Current task count
    performance_score: float = 1.0  # Performance multiplier


@dataclass
class HiveStructure:
    """Complete hive structure with Queen and workers."""
    queen: Optional[HiveAgent]
    workers: List[HiveAgent]
    topology: CoordinationType
    consensus_type: ConsensusType
    auto_scaling_enabled: bool
    max_workers: int = 12
    current_capacity: float = 0.0


@dataclass
class CollectiveDecision:
    """Result of collective decision-making process."""
    decision: str
    confidence: float
    participating_agents: List[str]
    consensus_reached: bool
    dissenting_opinions: List[str]
    timestamp: datetime


class HiveCoordinator:
    """Advanced hive coordination system for multi-agent orchestration."""
    
    def __init__(self, zen_consultant: ZenConsultant):
        """Initialize hive coordinator with ZEN integration."""
        self.zen_consultant = zen_consultant
        self.active_hives = {}  # session_id -> HiveStructure
        self.collective_memory = {}  # Shared knowledge across agents
        self.performance_metrics = {}  # Agent performance tracking
        
    def create_hive_structure(self, 
                            prompt: str, 
                            project_context: ProjectContext,
                            session_id: str = None) -> HiveStructure:
        """Create optimized hive structure based on task analysis."""
        # Use existing ZEN analysis
        complexity, metadata = self.zen_consultant.analyze_prompt_complexity(prompt)
        coordination = self.zen_consultant.determine_coordination_type(
            complexity, metadata["categories"], prompt
        )
        agents = self.zen_consultant.allocate_initial_agents(
            complexity, metadata["categories"], coordination
        )
        
        # Create Queen if needed
        queen = None
        if coordination in [CoordinationType.HIVE, CoordinationType.QUEEN_MODE]:
            queen = self._create_queen_agent(
                agents.queen_role or "chief-architect",
                metadata["categories"],
                complexity
            )
        
        # Create initial workers
        workers = []
        for i, agent_type in enumerate(agents.types):
            worker = HiveAgent(
                id=f"worker_{i+1}",
                role=HiveRole.WORKER,
                specialist_type=agent_type,
                capabilities=self._get_agent_capabilities(agent_type),
                priority=3,  # Standard worker priority
                active=True
            )
            workers.append(worker)
        
        # Determine consensus mechanism
        consensus_type = self._select_consensus_mechanism(complexity, coordination)
        
        hive = HiveStructure(
            queen=queen,
            workers=workers,
            topology=coordination,
            consensus_type=consensus_type,
            auto_scaling_enabled=agents.auto_scaling,
            max_workers=self._calculate_max_workers(complexity, project_context),
            current_capacity=self._calculate_current_capacity(workers)
        )
        
        if session_id:
            self.active_hives[session_id] = hive
            
        return hive
    
    def _create_queen_agent(self, 
                          queen_role: str, 
                          categories: List[str], 
                          complexity: ComplexityLevel) -> HiveAgent:
        """Create Queen agent with appropriate capabilities."""
        base_capabilities = [
            "strategic-planning",
            "task-coordination", 
            "quality-oversight",
            "decision-making",
            "conflict-resolution"
        ]
        
        # Add category-specific capabilities
        category_capabilities = {
            "architecture": ["system-design", "scalability-planning", "tech-stack-decisions"],
            "security": ["security-assessment", "vulnerability-analysis", "compliance-review"],
            "performance": ["performance-analysis", "optimization-planning", "resource-management"],
            "development": ["code-review", "architecture-guidance", "best-practices"],
            "testing": ["test-strategy", "quality-assurance", "coverage-analysis"],
            "deployment": ["deployment-planning", "infrastructure-management", "release-coordination"]
        }
        
        capabilities = base_capabilities.copy()
        for category in categories[:3]:  # Top 3 categories
            if category in category_capabilities:
                capabilities.extend(category_capabilities[category])
        
        return HiveAgent(
            id="queen_zen",
            role=HiveRole.QUEEN,
            specialist_type=queen_role,
            capabilities=capabilities,
            priority=5,  # Highest priority
            active=True,
            performance_score=1.2  # Queen performance bonus
        )
    
    def _get_agent_capabilities(self, agent_type: str) -> List[str]:
        """Get capabilities for specific agent type."""
        capability_map = {
            "coder": ["implementation", "debugging", "code-review"],
            "tester": ["test-creation", "validation", "quality-assurance"],
            "reviewer": ["code-review", "quality-check", "best-practices"],
            "system-architect": ["system-design", "architecture-planning", "scalability"],
            "security-auditor": ["security-analysis", "vulnerability-assessment", "compliance"],
            "performance-optimizer": ["performance-analysis", "optimization", "profiling"],
            "documentation-specialist": ["documentation", "technical-writing", "knowledge-management"],
            "deployment-engineer": ["deployment", "infrastructure", "automation"],
            "data-engineer": ["data-processing", "pipeline-design", "analytics"],
            "frontend-developer": ["ui-development", "user-experience", "responsive-design"],
            "backend-developer": ["api-development", "database-design", "server-architecture"]
        }
        
        return capability_map.get(agent_type, ["general-development"])
    
    def _select_consensus_mechanism(self, 
                                  complexity: ComplexityLevel, 
                                  coordination: CoordinationType) -> ConsensusType:
        """Select appropriate consensus mechanism."""
        if coordination == CoordinationType.QUEEN_MODE:
            return ConsensusType.QUEEN_DECISION
        elif complexity == ComplexityLevel.ENTERPRISE:
            return ConsensusType.WEIGHTED
        elif complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.HIVE_REQUIRED]:
            return ConsensusType.MAJORITY
        else:
            return ConsensusType.UNANIMOUS
    
    def _calculate_max_workers(self, 
                             complexity: ComplexityLevel, 
                             project_context: ProjectContext) -> int:
        """Calculate maximum workers based on complexity and project context."""
        base_max = {
            ComplexityLevel.SIMPLE: 3,
            ComplexityLevel.MEDIUM: 6,
            ComplexityLevel.COMPLEX: 9,
            ComplexityLevel.ENTERPRISE: 12,
            ComplexityLevel.HIVE_REQUIRED: 15
        }.get(complexity, 6)
        
        # Adjust based on project size
        if project_context.project_size == "enterprise":
            base_max = min(20, base_max + 5)
        elif project_context.project_size == "large":
            base_max = min(15, base_max + 3)
        
        return base_max
    
    def _calculate_current_capacity(self, workers: List[HiveAgent]) -> float:
        """Calculate current hive capacity utilization."""
        if not workers:
            return 0.0
        
        total_capacity = len(workers) * 5  # Assume max 5 tasks per worker
        current_load = sum(worker.workload for worker in workers if worker.active)
        
        return min(1.0, current_load / total_capacity) if total_capacity > 0 else 0.0
    
    def should_scale_workers(self, 
                           hive: HiveStructure, 
                           task_complexity: ComplexityLevel) -> Tuple[bool, int]:
        """Determine if worker scaling is needed."""
        if not hive.auto_scaling_enabled:
            return False, 0
        
        current_workers = len([w for w in hive.workers if w.active])
        
        # Scale up conditions
        if hive.current_capacity > 0.8 and current_workers < hive.max_workers:
            # High capacity utilization - scale up
            scale_amount = min(3, hive.max_workers - current_workers)
            return True, scale_amount
        
        # Scale down conditions
        if hive.current_capacity < 0.3 and current_workers > 2:
            # Low capacity utilization - scale down
            scale_amount = -min(2, current_workers - 2)
            return True, scale_amount
        
        return False, 0
    
    def spawn_worker(self, 
                    hive: HiveStructure, 
                    specialist_type: str, 
                    task_context: str) -> HiveAgent:
        """Spawn new worker agent with specific capabilities."""
        worker_id = f"worker_{len(hive.workers) + 1}"
        
        # Determine priority based on task urgency and specialist type
        priority_map = {
            "security-auditor": 4,
            "performance-optimizer": 4,
            "system-architect": 4,
            "quality-engineer": 3,
            "coder": 3,
            "tester": 2,
            "documentation-specialist": 1
        }
        
        worker = HiveAgent(
            id=worker_id,
            role=HiveRole.WORKER,
            specialist_type=specialist_type,
            capabilities=self._get_agent_capabilities(specialist_type),
            priority=priority_map.get(specialist_type, 2),
            active=True
        )
        
        hive.workers.append(worker)
        hive.current_capacity = self._calculate_current_capacity(hive.workers)
        
        return worker
    
    async def coordinate_collective_decision(self, 
                                           hive: HiveStructure, 
                                           decision_prompt: str,
                                           context: Dict[str, Any]) -> CollectiveDecision:
        """Coordinate collective decision-making across hive agents."""
        participating_agents = []
        agent_opinions = {}
        
        # Get Queen's opinion first (if exists)
        if hive.queen:
            queen_opinion = await self._get_agent_opinion(
                hive.queen, decision_prompt, context, is_queen=True
            )
            agent_opinions[hive.queen.id] = queen_opinion
            participating_agents.append(hive.queen.id)
        
        # Get worker opinions based on relevance
        relevant_workers = self._select_relevant_workers(hive.workers, decision_prompt)
        for worker in relevant_workers[:5]:  # Limit to 5 workers for efficiency
            worker_opinion = await self._get_agent_opinion(
                worker, decision_prompt, context, is_queen=False
            )
            agent_opinions[worker.id] = worker_opinion
            participating_agents.append(worker.id)
        
        # Apply consensus mechanism
        decision_result = self._apply_consensus(
            hive.consensus_type, agent_opinions, hive.queen
        )
        
        return CollectiveDecision(
            decision=decision_result["decision"],
            confidence=decision_result["confidence"],
            participating_agents=participating_agents,
            consensus_reached=decision_result["consensus"],
            dissenting_opinions=decision_result["dissenting"],
            timestamp=datetime.now()
        )
    
    def _select_relevant_workers(self, 
                               workers: List[HiveAgent], 
                               decision_prompt: str) -> List[HiveAgent]:
        """Select workers most relevant to the decision."""
        prompt_lower = decision_prompt.lower()
        relevant_keywords = {
            "security": ["security", "vulnerability", "auth", "encryption"],
            "performance": ["performance", "speed", "optimization", "scale"],
            "architecture": ["architecture", "design", "structure", "pattern"],
            "testing": ["test", "quality", "validation", "coverage"],
            "deployment": ["deploy", "release", "production", "infrastructure"],
            "data": ["data", "database", "analytics", "pipeline"]
        }
        
        scored_workers = []
        for worker in workers:
            if not worker.active:
                continue
                
            relevance_score = 0
            for capability in worker.capabilities:
                for category, keywords in relevant_keywords.items():
                    if any(keyword in capability for keyword in keywords):
                        if any(keyword in prompt_lower for keyword in keywords):
                            relevance_score += 1
            
            # Add priority bonus
            relevance_score += worker.priority * 0.5
            scored_workers.append((worker, relevance_score))
        
        # Sort by relevance score and return top workers
        scored_workers.sort(key=lambda x: x[1], reverse=True)
        return [worker for worker, _ in scored_workers]
    
    async def _get_agent_opinion(self, 
                               agent: HiveAgent, 
                               prompt: str, 
                               context: Dict[str, Any],
                               is_queen: bool = False) -> Dict[str, Any]:
        """Get opinion from specific agent (simulated)."""
        # In a real implementation, this would call the actual agent
        # For now, we simulate based on agent capabilities and role
        
        confidence_base = 0.8 if is_queen else 0.6
        confidence = min(0.95, confidence_base + agent.performance_score * 0.1)
        
        # Generate opinion based on agent specialization
        opinion_templates = {
            "security": "Recommend security-first approach with comprehensive threat analysis",
            "performance": "Prioritize performance optimization and scalability considerations", 
            "architecture": "Focus on maintainable architecture with clear separation of concerns",
            "testing": "Ensure comprehensive test coverage and quality validation",
            "deployment": "Plan for reliable deployment with rollback capabilities",
            "coder": "Implement with clean, maintainable code following best practices"
        }
        
        # Select template based on agent's primary capability
        primary_capability = agent.capabilities[0] if agent.capabilities else "general"
        opinion_base = opinion_templates.get(
            primary_capability.split("-")[0], 
            "Provide balanced technical solution"
        )
        
        return {
            "opinion": f"{opinion_base} (from {agent.specialist_type})",
            "confidence": confidence,
            "reasoning": f"Based on {agent.specialist_type} expertise and {len(agent.capabilities)} capabilities",
            "priority_weight": agent.priority
        }
    
    def _apply_consensus(self, 
                        consensus_type: ConsensusType, 
                        opinions: Dict[str, Dict[str, Any]], 
                        queen: Optional[HiveAgent]) -> Dict[str, Any]:
        """Apply consensus mechanism to agent opinions."""
        if consensus_type == ConsensusType.QUEEN_DECISION and queen:
            # Queen makes final decision
            queen_opinion = opinions.get(queen.id, {})
            return {
                "decision": queen_opinion.get("opinion", "Queen guidance required"),
                "confidence": queen_opinion.get("confidence", 0.8),
                "consensus": True,
                "dissenting": []
            }
        
        # Extract all opinions and confidences
        all_opinions = [op["opinion"] for op in opinions.values()]
        all_confidences = [op["confidence"] for op in opinions.values()]
        
        if not all_opinions:
            return {
                "decision": "No consensus reached - insufficient input",
                "confidence": 0.0,
                "consensus": False,
                "dissenting": []
            }
        
        # Simple consensus: use highest confidence opinion
        best_idx = all_confidences.index(max(all_confidences))
        best_opinion = all_opinions[best_idx]
        
        # Calculate overall confidence
        avg_confidence = sum(all_confidences) / len(all_confidences)
        
        # Determine consensus
        consensus_reached = avg_confidence > 0.7
        
        return {
            "decision": best_opinion,
            "confidence": avg_confidence,
            "consensus": consensus_reached,
            "dissenting": [op for op in all_opinions if op != best_opinion]
        }
    
    def get_hive_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of hive structure."""
        hive = self.active_hives.get(session_id)
        if not hive:
            return None
        
        return {
            "topology": hive.topology.value,
            "consensus_type": hive.consensus_type.value,
            "queen": {
                "id": hive.queen.id,
                "type": hive.queen.specialist_type,
                "capabilities": len(hive.queen.capabilities)
            } if hive.queen else None,
            "workers": [
                {
                    "id": worker.id,
                    "type": worker.specialist_type,
                    "active": worker.active,
                    "workload": worker.workload,
                    "priority": worker.priority
                }
                for worker in hive.workers
            ],
            "auto_scaling": hive.auto_scaling_enabled,
            "capacity_utilization": hive.current_capacity,
            "max_workers": hive.max_workers,
            "performance_metrics": self.performance_metrics.get(session_id, {})
        }
    
    def update_agent_performance(self, 
                               session_id: str, 
                               agent_id: str, 
                               performance_score: float):
        """Update agent performance metrics."""
        hive = self.active_hives.get(session_id)
        if not hive:
            return
        
        # Update agent performance score
        all_agents = [hive.queen] + hive.workers if hive.queen else hive.workers
        for agent in all_agents:
            if agent and agent.id == agent_id:
                agent.performance_score = max(0.1, min(2.0, performance_score))
                break
        
        # Update performance metrics
        if session_id not in self.performance_metrics:
            self.performance_metrics[session_id] = {}
        self.performance_metrics[session_id][agent_id] = {
            "score": performance_score,
            "last_updated": datetime.now().isoformat()
        }


# Integration functions for Context Intelligence Engine
def create_hive_enhanced_directive(prompt: str, 
                                 project_context: ProjectContext, 
                                 session_id: str = None) -> Dict[str, Any]:
    """Create hive-enhanced directive with advanced coordination."""
    zen_consultant = ZenConsultant()
    hive_coordinator = HiveCoordinator(zen_consultant)
    
    # Create hive structure
    hive = hive_coordinator.create_hive_structure(prompt, project_context, session_id)
    
    # Generate base directive
    base_directive = zen_consultant.generate_directive(prompt)
    
    # Enhance with hive information
    hive_info = []
    if hive.queen:
        hive_info.append(f"👑 Queen: {hive.queen.specialist_type}")
    
    worker_info = f"{len(hive.workers)} workers" if hive.workers else "0 workers"
    hive_info.append(f"🐝 Workers: {worker_info}")
    
    if hive.auto_scaling_enabled:
        hive_info.append(f"📈 Auto-scaling: up to {hive.max_workers} agents")
    
    hive_summary = " | ".join(hive_info)
    
    enhanced_directive = f"{base_directive}\n\n🏗️ HIVE STRUCTURE: {hive_summary}"
    
    return {
        "directive": enhanced_directive,
        "hive_structure": asdict(hive),
        "session_id": session_id
    }


if __name__ == "__main__":
    """Test hive coordination system."""
    print("🐝 Hive Coordination System Test")
    print("=" * 50)
    
    # Create test context
    from .shared_types import GitContext, ProjectContext, TechStack
    
    git_context = GitContext(
        is_repo=True, current_branch="main", uncommitted_changes=2,
        recent_commits=[], branch_health=0.8, last_activity=datetime.now(),
        repository_age_days=90, commit_frequency=0.5
    )
    
    project_context = ProjectContext(
        git_context=git_context,
        tech_stacks=[TechStack.PYTHON, TechStack.REACT],
        complexity_indicators={"code_files_count": 150},
        file_structure={".py": 80, ".js": 40, ".json": 10},
        project_size="medium",
        dependencies_count=45,
        test_coverage_estimate=0.7,
        documentation_quality=0.6
    )
    
    # Test hive creation
    zen_consultant = ZenConsultant()
    coordinator = HiveCoordinator(zen_consultant)
    
    test_prompts = [
        "Orchestrate a comprehensive security audit of our microservices architecture",
        "Plan and execute a complex database migration with zero downtime",
        "Design and implement a scalable real-time chat system"
    ]
    
    for prompt in test_prompts:
        print(f"\n🎯 Prompt: {prompt}")
        hive = coordinator.create_hive_structure(prompt, project_context, f"test_{hash(prompt)}")
        
        print(f"  👑 Queen: {hive.queen.specialist_type if hive.queen else 'None'}")
        print(f"  🐝 Workers: {len(hive.workers)}")
        print(f"  🏗️ Topology: {hive.topology.value}")
        print(f"  🤝 Consensus: {hive.consensus_type.value}")
        print(f"  📈 Auto-scaling: {hive.auto_scaling_enabled}")