"""
Federation Controller - Central coordinator for multi-project swarms

Manages federation of multiple Hive Mind swarms across projects,
enabling resource sharing, cross-project coordination, and global optimization.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

from ..memory.project_memory_manager import ProjectMemoryManager
from ..optimization.async_orchestrator import AsyncOrchestrator
from ..predictive.workflow_prediction_engine import WorkflowPredictionEngine


@dataclass
class SwarmInfo:
    """Information about a registered swarm"""
    swarm_id: str
    project_id: str
    queen_endpoint: str
    worker_types: List[str]
    max_workers: int
    current_load: float
    memory_available: int
    last_heartbeat: float
    status: str = "active"
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceRequest:
    """Request for resources from another swarm"""
    request_id: str
    from_swarm: str
    to_swarm: str
    agent_type: str
    duration: int
    priority: str
    task_context: Dict[str, Any]
    status: str = "pending"
    created_at: float = field(default_factory=time.time)


@dataclass
class FederationMetrics:
    """Metrics for federation performance"""
    total_swarms: int = 0
    active_swarms: int = 0
    total_agents: int = 0
    available_agents: int = 0
    resource_requests: int = 0
    successful_migrations: int = 0
    failed_migrations: int = 0
    average_response_time: float = 0.0
    last_updated: float = field(default_factory=time.time)


class FederationController:
    """
    Central controller for multi-project swarm federation.
    Manages swarm registration, resource allocation, and global coordination.
    """
    
    def __init__(self, federation_id: Optional[str] = None):
        self.federation_id = federation_id or f"federation-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.logger = self._setup_logger()
        
        # Swarm management
        self.swarms: Dict[str, SwarmInfo] = {}
        self.resource_requests: Dict[str, ResourceRequest] = {}
        
        # Components
        self.memory_manager = ProjectMemoryManager()
        self.async_orchestrator = AsyncOrchestrator()
        self.workflow_predictor = WorkflowPredictionEngine()
        
        # Federation state
        self.is_running = False
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_timeout = 90  # seconds
        self._heartbeat_task = None
        
        # Metrics
        self.metrics = FederationMetrics()
        
        # Protocol handlers
        self.message_handlers = {
            "DISCOVERY": self._handle_discovery,
            "HEARTBEAT": self._handle_heartbeat,
            "RESOURCE_REQUEST": self._handle_resource_request,
            "RESOURCE_RESPONSE": self._handle_resource_response,
            "CONSENSUS_PROPOSAL": self._handle_consensus_proposal,
            "AGENT_MIGRATION": self._handle_agent_migration
        }
        
        self.logger.info(f"Federation Controller initialized: {self.federation_id}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup federation logger"""
        logger = logging.getLogger(f"federation.{self.federation_id}")
        logger.setLevel(logging.INFO)
        return logger
    
    async def start(self):
        """Start the federation controller"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting Federation Controller...")
        
        # Start heartbeat monitor
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        # Initialize memory namespace
        await self._initialize_federation_memory()
        
        # Start discovery service
        asyncio.create_task(self._discovery_service())
        
        self.logger.info("Federation Controller started successfully")
    
    async def stop(self):
        """Stop the federation controller"""
        self.is_running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Federation Controller stopped")
    
    async def register_swarm(self, swarm_data: Dict[str, Any]) -> bool:
        """Register a new swarm with the federation"""
        try:
            swarm_info = SwarmInfo(
                swarm_id=swarm_data["swarm_id"],
                project_id=swarm_data["project_id"],
                queen_endpoint=swarm_data["queen_endpoint"],
                worker_types=swarm_data["capabilities"]["worker_types"],
                max_workers=swarm_data["capabilities"]["max_workers"],
                current_load=swarm_data["capabilities"]["current_load"],
                memory_available=swarm_data["capabilities"]["memory_available"],
                last_heartbeat=time.time(),
                capabilities=swarm_data["capabilities"]
            )
            
            self.swarms[swarm_info.swarm_id] = swarm_info
            self.metrics.total_swarms = len(self.swarms)
            self.metrics.active_swarms = sum(1 for s in self.swarms.values() if s.status == "active")
            
            # Store in federation memory
            await self.memory_manager.store_memory(
                f"swarm:{swarm_info.swarm_id}",
                swarm_data,
                category="federation",
                ttl=3600
            )
            
            self.logger.info(f"Registered swarm: {swarm_info.swarm_id} from project: {swarm_info.project_id}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Failed to register swarm: {e}")
            return False
    
    async def request_resource(self, request_data: Dict[str, Any]) -> str:
        """Request resources from another swarm"""
        request = ResourceRequest(
            request_id=f"req-{int(time.time())}-{uuid.uuid4().hex[:8]}",
            from_swarm=request_data["from_swarm"],
            to_swarm=request_data.get("to_swarm", "any"),
            agent_type=request_data["agent_type"],
            duration=request_data["duration"],
            priority=request_data["priority"],
            task_context=request_data.get("task_context", {})
        )
        
        self.resource_requests[request.request_id] = request
        self.metrics.resource_requests += 1
        
        # Find best swarm for resource
        if request.to_swarm == "any":
            target_swarm = await self._find_best_swarm(request.agent_type)
            if target_swarm:
                request.to_swarm = target_swarm.swarm_id
        
        # Send resource request
        if request.to_swarm in self.swarms:
            await self._send_resource_request(request)
        
        return request.request_id
    
    async def get_federation_status(self) -> Dict[str, Any]:
        """Get current federation status"""
        return {
            "federation_id": self.federation_id,
            "status": "active" if self.is_running else "inactive",
            "swarms": {
                swarm_id: {
                    "project_id": swarm.project_id,
                    "status": swarm.status,
                    "workers": swarm.max_workers,
                    "load": swarm.current_load,
                    "last_seen": time.time() - swarm.last_heartbeat
                }
                for swarm_id, swarm in self.swarms.items()
            },
            "metrics": {
                "total_swarms": self.metrics.total_swarms,
                "active_swarms": self.metrics.active_swarms,
                "resource_requests": self.metrics.resource_requests,
                "successful_migrations": self.metrics.successful_migrations,
                "average_response_time": self.metrics.average_response_time
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _heartbeat_monitor(self):
        """Monitor swarm heartbeats"""
        while self.is_running:
            try:
                current_time = time.time()
                
                for swarm_id, swarm in self.swarms.items():
                    time_since_heartbeat = current_time - swarm.last_heartbeat
                    
                    if time_since_heartbeat > self.heartbeat_timeout:
                        if swarm.status == "active":
                            swarm.status = "inactive"
                            self.logger.warning(f"Swarm {swarm_id} marked inactive - no heartbeat")
                            self.metrics.active_swarms -= 1
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.exception(f"Heartbeat monitor error: {e}")
    
    async def _discovery_service(self):
        """Handle swarm discovery broadcasts"""
        while self.is_running:
            try:
                # Broadcast federation presence
                discovery_message = {
                    "type": "FEDERATION_ANNOUNCE",
                    "federation_id": self.federation_id,
                    "endpoint": f"federation://{self.federation_id}",
                    "capabilities": {
                        "max_swarms": 100,
                        "protocols": ["v1.0"],
                        "features": ["resource_sharing", "agent_migration", "global_optimization"]
                    },
                    "timestamp": time.time()
                }
                
                # In real implementation, broadcast via network
                await self._broadcast_message(discovery_message)
                
                await asyncio.sleep(60)  # Announce every minute
                
            except Exception as e:
                self.logger.exception(f"Discovery service error: {e}")
    
    async def _find_best_swarm(self, agent_type: str) -> Optional[SwarmInfo]:
        """Find the best swarm for a specific agent type"""
        candidates = [
            swarm for swarm in self.swarms.values()
            if swarm.status == "active" and 
            agent_type in swarm.worker_types and
            swarm.current_load < 0.8
        ]
        
        if not candidates:
            return None
        
        # Sort by load and available memory
        candidates.sort(key=lambda s: (s.current_load, -s.memory_available))
        return candidates[0]
    
    async def _send_resource_request(self, request: ResourceRequest):
        """Send resource request to target swarm"""
        message = {
            "type": "RESOURCE_REQUEST",
            "request_id": request.request_id,
            "from_swarm": request.from_swarm,
            "to_swarm": request.to_swarm,
            "agent_type": request.agent_type,
            "duration": request.duration,
            "priority": request.priority,
            "task_context": request.task_context,
            "timestamp": time.time()
        }
        
        # In real implementation, send via network to target swarm
        await self._send_message(request.to_swarm, message)
    
    async def _initialize_federation_memory(self):
        """Initialize federation memory namespace"""
        federation_config = {
            "federation_id": self.federation_id,
            "created_at": datetime.now().isoformat(),
            "protocol_version": "1.0",
            "features": ["resource_sharing", "agent_migration"]
        }
        
        await self.memory_manager.store_memory(
            "federation:config",
            federation_config,
            category="federation",
            metadata={"persistent": True}
        )
    
    # Message handlers
    async def _handle_discovery(self, message: Dict[str, Any]):
        """Handle discovery message"""
        await self.register_swarm(message)
    
    async def _handle_heartbeat(self, message: Dict[str, Any]):
        """Handle heartbeat message"""
        swarm_id = message["swarm_id"]
        if swarm_id in self.swarms:
            self.swarms[swarm_id].last_heartbeat = time.time()
            self.swarms[swarm_id].current_load = message.get("current_load", 0)
    
    async def _handle_resource_request(self, message: Dict[str, Any]):
        """Handle incoming resource request"""
        # Process resource request
        pass
    
    async def _handle_resource_response(self, message: Dict[str, Any]):
        """Handle resource response"""
        request_id = message["correlation_id"]
        if request_id in self.resource_requests:
            request = self.resource_requests[request_id]
            request.status = message["status"]
    
    async def _handle_consensus_proposal(self, message: Dict[str, Any]):
        """Handle consensus proposal"""
        # Implement multi-queen consensus
        pass
    
    async def _handle_agent_migration(self, message: Dict[str, Any]):
        """Handle agent migration"""
        # Implement agent state transfer
        pass
    
    # Network simulation (to be replaced with real networking)
    async def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all swarms"""
        # In real implementation, use network broadcast
        pass
    
    async def _send_message(self, target: str, message: Dict[str, Any]):
        """Send message to specific swarm"""
        # In real implementation, use network send
        pass