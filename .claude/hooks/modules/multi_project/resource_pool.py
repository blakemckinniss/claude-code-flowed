"""
Global Resource Pool - Manages agents across all federation swarms

Provides unified resource management, agent allocation, and
load balancing across multiple project swarms.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import heapq


class AgentStatus(Enum):
    """Agent availability status"""
    IDLE = "idle"
    BUSY = "busy"
    MIGRATING = "migrating"
    RESERVED = "reserved"
    OFFLINE = "offline"


@dataclass
class AgentResource:
    """Represents an agent in the global pool"""
    agent_id: str
    agent_type: str
    home_swarm: str
    current_swarm: str
    status: AgentStatus
    capabilities: List[str]
    performance_score: float = 1.0
    last_task_completed: float = 0.0
    total_tasks: int = 0
    success_rate: float = 1.0
    current_task: Optional[str] = None
    reserved_until: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Tracks resource allocation to a task"""
    allocation_id: str
    agent_id: str
    requesting_swarm: str
    task_id: str
    allocated_at: float
    duration: int
    priority: str
    status: str = "active"


@dataclass 
class ResourceMetrics:
    """Metrics for resource pool performance"""
    total_agents: int = 0
    available_agents: int = 0
    busy_agents: int = 0
    migrating_agents: int = 0
    allocations_total: int = 0
    allocations_successful: int = 0
    allocations_failed: int = 0
    average_allocation_time: float = 0.0
    resource_utilization: float = 0.0


class GlobalResourcePool:
    """
    Manages the global pool of agents across all swarms.
    Handles allocation, migration, and load balancing.
    """
    
    def __init__(self):
        # Agent storage
        self._agents: Dict[str, AgentResource] = {}
        self._swarm_agents: Dict[str, Set[str]] = defaultdict(set)
        self._type_agents: Dict[str, Set[str]] = defaultdict(set)
        
        # Allocation tracking
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._agent_allocations: Dict[str, str] = {}  # agent_id -> allocation_id
        
        # Performance tracking
        self._allocation_history: List[Tuple[float, float]] = []  # (timestamp, duration)
        self._metrics = ResourceMetrics()
        
        # Load balancing
        self._load_scores: Dict[str, float] = defaultdict(float)
        self._rebalance_interval = 300  # 5 minutes
        self._last_rebalance = time.time()
        
        # Locks
        self._lock = asyncio.Lock()
        self._allocation_lock = asyncio.Lock()
    
    async def register_agent(self, agent_data: Dict[str, Any]) -> bool:
        """Register a new agent in the global pool"""
        async with self._lock:
            try:
                agent = AgentResource(
                    agent_id=agent_data["agent_id"],
                    agent_type=agent_data["agent_type"],
                    home_swarm=agent_data["home_swarm"],
                    current_swarm=agent_data["home_swarm"],
                    status=AgentStatus.IDLE,
                    capabilities=agent_data.get("capabilities", []),
                    performance_score=agent_data.get("performance_score", 1.0),
                    metadata=agent_data.get("metadata", {})
                )
                
                self._agents[agent.agent_id] = agent
                self._swarm_agents[agent.home_swarm].add(agent.agent_id)
                self._type_agents[agent.agent_type].add(agent.agent_id)
                
                self._update_metrics()
                return True
                
            except Exception as e:
                print(f"Failed to register agent: {e}")
                return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the global pool"""
        async with self._lock:
            if agent_id not in self._agents:
                return False
            
            agent = self._agents[agent_id]
            
            # Remove from indices
            self._swarm_agents[agent.home_swarm].discard(agent_id)
            self._swarm_agents[agent.current_swarm].discard(agent_id)
            self._type_agents[agent.agent_type].discard(agent_id)
            
            # Remove agent
            del self._agents[agent_id]
            
            # Clean up allocations
            if agent_id in self._agent_allocations:
                allocation_id = self._agent_allocations[agent_id]
                if allocation_id in self._allocations:
                    self._allocations[allocation_id].status = "terminated"
                del self._agent_allocations[agent_id]
            
            self._update_metrics()
            return True
    
    async def allocate_agent(self, request: Dict[str, Any]) -> Optional[str]:
        """Allocate an agent for a task"""
        async with self._allocation_lock:
            agent_type = request["agent_type"]
            duration = request["duration"]
            priority = request.get("priority", "normal")
            requesting_swarm = request["requesting_swarm"]
            task_id = request.get("task_id", str(uuid.uuid4()))
            
            # Find available agent
            agent = await self._find_best_agent(agent_type, requesting_swarm, priority)
            
            if not agent:
                self._metrics.allocations_failed += 1
                return None
            
            # Create allocation
            allocation = ResourceAllocation(
                allocation_id=f"alloc-{int(time.time())}-{uuid.uuid4().hex[:8]}",
                agent_id=agent.agent_id,
                requesting_swarm=requesting_swarm,
                task_id=task_id,
                allocated_at=time.time(),
                duration=duration,
                priority=priority
            )
            
            # Update agent status
            agent.status = AgentStatus.BUSY
            agent.current_task = task_id
            agent.reserved_until = time.time() + duration
            
            # Track allocation
            self._allocations[allocation.allocation_id] = allocation
            self._agent_allocations[agent.agent_id] = allocation.allocation_id
            
            # Update metrics
            self._metrics.allocations_total += 1
            self._metrics.allocations_successful += 1
            self._update_metrics()
            
            # Track allocation time
            self._allocation_history.append((time.time(), 0))
            
            return allocation.allocation_id
    
    async def release_agent(self, allocation_id: str) -> bool:
        """Release an allocated agent"""
        async with self._allocation_lock:
            if allocation_id not in self._allocations:
                return False
            
            allocation = self._allocations[allocation_id]
            agent_id = allocation.agent_id
            
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.status = AgentStatus.IDLE
                agent.current_task = None
                agent.reserved_until = None
                agent.last_task_completed = time.time()
                agent.total_tasks += 1
                
                # Update allocation
                allocation.status = "completed"
                duration = time.time() - allocation.allocated_at
                
                # Update allocation history
                if self._allocation_history:
                    self._allocation_history[-1] = (allocation.allocated_at, duration)
            
            # Clean up tracking
            if agent_id in self._agent_allocations:
                del self._agent_allocations[agent_id]
            
            self._update_metrics()
            return True
    
    async def migrate_agent(self, agent_id: str, target_swarm: str) -> bool:
        """Migrate an agent to a different swarm"""
        async with self._lock:
            if agent_id not in self._agents:
                return False
            
            agent = self._agents[agent_id]
            
            if agent.status != AgentStatus.IDLE:
                return False  # Can't migrate busy agents
            
            # Update status
            agent.status = AgentStatus.MIGRATING
            source_swarm = agent.current_swarm
            
            # Simulate migration (in real implementation, coordinate with swarms)
            await asyncio.sleep(0.1)  # Simulated migration delay
            
            # Update agent location
            self._swarm_agents[source_swarm].discard(agent_id)
            self._swarm_agents[target_swarm].add(agent_id)
            agent.current_swarm = target_swarm
            agent.status = AgentStatus.IDLE
            
            self._update_metrics()
            return True
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get current resource pool status"""
        async with self._lock:
            status_counts = defaultdict(int)
            for agent in self._agents.values():
                status_counts[agent.status.value] += 1
            
            swarm_distribution = {}
            for swarm_id, agent_ids in self._swarm_agents.items():
                swarm_distribution[swarm_id] = len(agent_ids)
            
            type_distribution = {}
            for agent_type, agent_ids in self._type_agents.items():
                type_distribution[agent_type] = len(agent_ids)
            
            return {
                "total_agents": len(self._agents),
                "status_distribution": dict(status_counts),
                "swarm_distribution": swarm_distribution,
                "type_distribution": type_distribution,
                "active_allocations": len([a for a in self._allocations.values() if a.status == "active"]),
                "metrics": {
                    "utilization": self._metrics.resource_utilization,
                    "success_rate": self._metrics.allocations_successful / max(self._metrics.allocations_total, 1),
                    "average_allocation_time": self._metrics.average_allocation_time
                }
            }
    
    async def rebalance_agents(self) -> Dict[str, int]:
        """Rebalance agents across swarms for optimal distribution"""
        async with self._lock:
            current_time = time.time()
            
            if current_time - self._last_rebalance < self._rebalance_interval:
                return {}
            
            migrations = defaultdict(int)
            
            # Calculate load scores for each swarm
            swarm_loads = {}
            for swarm_id, agent_ids in self._swarm_agents.items():
                total_agents = len(agent_ids)
                busy_agents = sum(1 for aid in agent_ids 
                                if self._agents[aid].status == AgentStatus.BUSY)
                swarm_loads[swarm_id] = busy_agents / max(total_agents, 1)
            
            # Find overloaded and underloaded swarms
            avg_load = sum(swarm_loads.values()) / max(len(swarm_loads), 1)
            overloaded = {s: l for s, l in swarm_loads.items() if l > avg_load + 0.2}
            underloaded = {s: l for s, l in swarm_loads.items() if l < avg_load - 0.2}
            
            # Plan migrations
            for source_swarm in overloaded:
                idle_agents = [
                    aid for aid in self._swarm_agents[source_swarm]
                    if self._agents[aid].status == AgentStatus.IDLE
                ]
                
                for target_swarm in underloaded:
                    if not idle_agents:
                        break
                    
                    # Migrate one agent
                    agent_id = idle_agents.pop()
                    if await self.migrate_agent(agent_id, target_swarm):
                        migrations[f"{source_swarm}->{target_swarm}"] += 1
            
            self._last_rebalance = current_time
            return dict(migrations)
    
    async def _find_best_agent(self, agent_type: str, requesting_swarm: str, 
                              priority: str) -> Optional[AgentResource]:
        """Find the best available agent for allocation"""
        candidates = []
        
        for agent_id in self._type_agents.get(agent_type, set()):
            agent = self._agents[agent_id]
            
            if agent.status == AgentStatus.IDLE:
                # Calculate score based on multiple factors
                score = self._calculate_agent_score(agent, requesting_swarm, priority)
                heapq.heappush(candidates, (-score, agent))
        
        if candidates:
            _, best_agent = heapq.heappop(candidates)
            return best_agent
        
        return None
    
    def _calculate_agent_score(self, agent: AgentResource, requesting_swarm: str, 
                              priority: str) -> float:
        """Calculate allocation score for an agent"""
        score = agent.performance_score * agent.success_rate
        
        # Prefer agents in the same swarm
        if agent.current_swarm == requesting_swarm:
            score *= 1.5
        
        # Boost score for high priority requests
        if priority == "high":
            score *= 1.2
        elif priority == "critical":
            score *= 1.5
        
        # Penalize recently used agents for load distribution
        time_since_last_task = time.time() - agent.last_task_completed
        if time_since_last_task < 60:  # Used in last minute
            score *= 0.8
        
        return score
    
    def _update_metrics(self):
        """Update resource pool metrics"""
        total = len(self._agents)
        self._metrics.total_agents = total
        
        status_counts = defaultdict(int)
        for agent in self._agents.values():
            status_counts[agent.status] += 1
        
        self._metrics.available_agents = status_counts[AgentStatus.IDLE]
        self._metrics.busy_agents = status_counts[AgentStatus.BUSY]
        self._metrics.migrating_agents = status_counts[AgentStatus.MIGRATING]
        
        if total > 0:
            self._metrics.resource_utilization = self._metrics.busy_agents / total
        
        # Calculate average allocation time
        if self._allocation_history:
            recent_allocations = self._allocation_history[-100:]  # Last 100
            durations = [d for _, d in recent_allocations if d > 0]
            if durations:
                self._metrics.average_allocation_time = sum(durations) / len(durations)