"""
Swarm Registry - Manages registration and discovery of project swarms

Provides centralized registry for all swarms in the federation,
enabling discovery, health monitoring, and capability tracking.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class SwarmCapability:
    """Capability descriptor for a swarm"""
    agent_types: Set[str]
    max_capacity: int
    specializations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SwarmRegistration:
    """Complete registration data for a swarm"""
    swarm_id: str
    project_id: str
    project_name: str
    queen_endpoint: str
    capabilities: SwarmCapability
    metadata: Dict[str, Any]
    registered_at: float
    last_updated: float
    health_score: float = 1.0
    availability: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "swarm_id": self.swarm_id,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "queen_endpoint": self.queen_endpoint,
            "capabilities": {
                "agent_types": list(self.capabilities.agent_types),
                "max_capacity": self.capabilities.max_capacity,
                "specializations": self.capabilities.specializations,
                "performance_metrics": self.capabilities.performance_metrics
            },
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "last_updated": self.last_updated,
            "health_score": self.health_score,
            "availability": self.availability
        }


class SwarmRegistry:
    """
    Central registry for all swarms in the federation.
    Manages discovery, health tracking, and capability queries.
    """
    
    def __init__(self):
        # Registry storage
        self._swarms: Dict[str, SwarmRegistration] = {}
        self._project_swarms: Dict[str, Set[str]] = defaultdict(set)
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Health tracking
        self._health_history: Dict[str, List[float]] = defaultdict(list)
        self._availability_window = 3600  # 1 hour window
        
        # Registry metadata
        self._registry_version = "1.0.0"
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
    
    async def register(self, registration_data: Dict[str, Any]) -> bool:
        """Register a new swarm or update existing registration"""
        async with self._lock:
            try:
                # Create capability object
                capabilities = SwarmCapability(
                    agent_types=set(registration_data["capabilities"]["agent_types"]),
                    max_capacity=registration_data["capabilities"]["max_capacity"],
                    specializations=registration_data["capabilities"].get("specializations", []),
                    performance_metrics=registration_data["capabilities"].get("performance_metrics", {})
                )
                
                # Create or update registration
                swarm_id = registration_data["swarm_id"]
                current_time = time.time()
                
                if swarm_id in self._swarms:
                    # Update existing
                    registration = self._swarms[swarm_id]
                    registration.capabilities = capabilities
                    registration.metadata = registration_data.get("metadata", {})
                    registration.last_updated = current_time
                else:
                    # Create new
                    registration = SwarmRegistration(
                        swarm_id=swarm_id,
                        project_id=registration_data["project_id"],
                        project_name=registration_data.get("project_name", ""),
                        queen_endpoint=registration_data["queen_endpoint"],
                        capabilities=capabilities,
                        metadata=registration_data.get("metadata", {}),
                        registered_at=current_time,
                        last_updated=current_time
                    )
                    self._swarms[swarm_id] = registration
                
                # Update indices
                self._project_swarms[registration.project_id].add(swarm_id)
                
                # Update capability index
                for agent_type in capabilities.agent_types:
                    self._capability_index[agent_type].add(swarm_id)
                
                # Perform cleanup if needed
                if current_time - self._last_cleanup > self._cleanup_interval:
                    await self._cleanup_stale_entries()
                
                return True
                
            except Exception as e:
                print(f"Registration error: {e}")
                return False
    
    async def unregister(self, swarm_id: str) -> bool:
        """Unregister a swarm from the registry"""
        async with self._lock:
            if swarm_id not in self._swarms:
                return False
            
            registration = self._swarms[swarm_id]
            
            # Remove from indices
            self._project_swarms[registration.project_id].discard(swarm_id)
            
            for agent_type in registration.capabilities.agent_types:
                self._capability_index[agent_type].discard(swarm_id)
            
            # Remove from main registry
            del self._swarms[swarm_id]
            
            # Clean up health history
            if swarm_id in self._health_history:
                del self._health_history[swarm_id]
            
            return True
    
    async def update_health(self, swarm_id: str, health_score: float, availability: float):
        """Update health metrics for a swarm"""
        async with self._lock:
            if swarm_id not in self._swarms:
                return
            
            registration = self._swarms[swarm_id]
            registration.health_score = health_score
            registration.availability = availability
            registration.last_updated = time.time()
            
            # Track health history
            self._health_history[swarm_id].append(health_score)
            
            # Keep only recent history
            max_history = 100
            if len(self._health_history[swarm_id]) > max_history:
                self._health_history[swarm_id] = self._health_history[swarm_id][-max_history:]
    
    async def find_swarms_by_capability(self, agent_type: str, 
                                       min_health: float = 0.5,
                                       min_availability: float = 0.5) -> List[SwarmRegistration]:
        """Find swarms that support a specific agent type"""
        async with self._lock:
            swarm_ids = self._capability_index.get(agent_type, set())
            
            results = []
            for swarm_id in swarm_ids:
                registration = self._swarms.get(swarm_id)
                if registration and \
                   registration.health_score >= min_health and \
                   registration.availability >= min_availability:
                    results.append(registration)
            
            # Sort by health score and availability
            results.sort(key=lambda r: (r.health_score * r.availability), reverse=True)
            
            return results
    
    async def find_swarms_by_project(self, project_id: str) -> List[SwarmRegistration]:
        """Find all swarms belonging to a project"""
        async with self._lock:
            swarm_ids = self._project_swarms.get(project_id, set())
            return [self._swarms[sid] for sid in swarm_ids if sid in self._swarms]
    
    async def get_swarm(self, swarm_id: str) -> Optional[SwarmRegistration]:
        """Get a specific swarm registration"""
        async with self._lock:
            return self._swarms.get(swarm_id)
    
    async def get_all_swarms(self) -> List[SwarmRegistration]:
        """Get all registered swarms"""
        async with self._lock:
            return list(self._swarms.values())
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        async with self._lock:
            total_capacity = sum(s.capabilities.max_capacity for s in self._swarms.values())
            avg_health = sum(s.health_score for s in self._swarms.values()) / max(len(self._swarms), 1)
            
            agent_type_counts = {}
            for agent_type, swarm_ids in self._capability_index.items():
                agent_type_counts[agent_type] = len(swarm_ids)
            
            return {
                "total_swarms": len(self._swarms),
                "total_projects": len(self._project_swarms),
                "total_capacity": total_capacity,
                "average_health": avg_health,
                "agent_type_distribution": agent_type_counts,
                "registry_version": self._registry_version,
                "last_cleanup": self._last_cleanup
            }
    
    async def calculate_swarm_similarity(self, swarm_id1: str, swarm_id2: str) -> float:
        """Calculate similarity score between two swarms based on capabilities"""
        async with self._lock:
            swarm1 = self._swarms.get(swarm_id1)
            swarm2 = self._swarms.get(swarm_id2)
            
            if not swarm1 or not swarm2:
                return 0.0
            
            # Calculate Jaccard similarity of agent types
            types1 = swarm1.capabilities.agent_types
            types2 = swarm2.capabilities.agent_types
            
            if not types1 and not types2:
                return 1.0
            
            intersection = len(types1 & types2)
            union = len(types1 | types2)
            
            return intersection / union if union > 0 else 0.0
    
    async def recommend_fallback_swarms(self, swarm_id: str, limit: int = 3) -> List[str]:
        """Recommend similar swarms as fallbacks"""
        async with self._lock:
            if swarm_id not in self._swarms:
                return []
            
            self._swarms[swarm_id]
            similarities = []
            
            for other_id, other_swarm in self._swarms.items():
                if other_id != swarm_id and other_swarm.health_score > 0.5:
                    similarity = await self.calculate_swarm_similarity(swarm_id, other_id)
                    similarities.append((other_id, similarity))
            
            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [sid for sid, _ in similarities[:limit]]
    
    async def _cleanup_stale_entries(self):
        """Remove stale entries from the registry"""
        current_time = time.time()
        stale_threshold = 3600 * 24  # 24 hours
        
        stale_swarms = []
        for swarm_id, registration in self._swarms.items():
            if current_time - registration.last_updated > stale_threshold:
                stale_swarms.append(swarm_id)
        
        for swarm_id in stale_swarms:
            await self.unregister(swarm_id)
        
        self._last_cleanup = current_time
    
    def generate_swarm_hash(self, swarm_data: Dict[str, Any]) -> str:
        """Generate a unique hash for swarm data"""
        # Create a stable hash from swarm properties
        hash_data = f"{swarm_data['project_id']}:{swarm_data['swarm_id']}"
        return hashlib.sha256(hash_data.encode()).hexdigest()[:16]