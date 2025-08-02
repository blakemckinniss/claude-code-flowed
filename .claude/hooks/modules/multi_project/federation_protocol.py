"""
Federation Protocol - Implements the multi-swarm coordination protocol

Handles message passing, consensus building, and state synchronization
between swarms in the federation.
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import struct


class MessageType(Enum):
    """Federation message types"""
    DISCOVERY = "DISCOVERY"
    HEARTBEAT = "HEARTBEAT"
    RESOURCE_REQUEST = "RESOURCE_REQUEST"
    RESOURCE_RESPONSE = "RESOURCE_RESPONSE"
    CONSENSUS_PROPOSAL = "CONSENSUS_PROPOSAL"
    CONSENSUS_VOTE = "CONSENSUS_VOTE"
    CONSENSUS_RESULT = "CONSENSUS_RESULT"
    AGENT_MIGRATION = "AGENT_MIGRATION"
    STATE_SYNC = "STATE_SYNC"
    STATE_DIGEST = "STATE_DIGEST"


@dataclass
class FederationMessage:
    """Standard federation message format"""
    message_id: str
    message_type: MessageType
    sender: str
    recipient: str  # 'broadcast' for all
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    signature: Optional[str] = None
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        data = {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "signature": self.signature
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'FederationMessage':
        """Deserialize message from bytes"""
        msg_data = json.loads(data.decode('utf-8'))
        return cls(
            message_id=msg_data["message_id"],
            message_type=MessageType(msg_data["message_type"]),
            sender=msg_data["sender"],
            recipient=msg_data["recipient"],
            payload=msg_data["payload"],
            timestamp=msg_data["timestamp"],
            correlation_id=msg_data.get("correlation_id"),
            signature=msg_data.get("signature")
        )


@dataclass
class ConsensusProposal:
    """Represents a consensus proposal"""
    proposal_id: str
    proposer: str
    subject: str
    proposal: Dict[str, Any]
    voting_deadline: float
    votes: Dict[str, str] = field(default_factory=dict)  # voter -> vote
    weights: Dict[str, float] = field(default_factory=dict)  # voter -> weight
    status: str = "pending"  # pending, approved, rejected
    
    def add_vote(self, voter: str, vote: str, weight: float = 1.0):
        """Add a vote to the proposal"""
        self.votes[voter] = vote
        self.weights[voter] = weight
    
    def calculate_result(self, required_majority: float = 0.51) -> str:
        """Calculate consensus result"""
        total_weight = sum(self.weights.values())
        approve_weight = sum(w for v, w in zip(self.votes.values(), self.weights.values()) 
                           if self.votes[v] == "APPROVE")
        
        if total_weight > 0:
            approval_ratio = approve_weight / total_weight
            self.status = "approved" if approval_ratio >= required_majority else "rejected"
        
        return self.status


class FederationProtocol:
    """
    Implements the federation protocol for multi-swarm coordination.
    Handles all protocol-level operations and message routing.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Message handling
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        
        # Consensus management
        self._active_proposals: Dict[str, ConsensusProposal] = {}
        self._consensus_threshold = 0.51  # Simple majority
        
        # State management
        self._local_state: Dict[str, Any] = {}
        self._state_version = 0
        self._state_digests: Dict[str, str] = {}  # node_id -> digest
        
        # Network simulation (to be replaced with real networking)
        self._peers: Set[str] = set()
        self._message_queue = asyncio.Queue()
        
        # Protocol configuration
        self._heartbeat_interval = 30
        self._sync_interval = 60
        self._message_timeout = 10
        
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self._message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self._message_handlers[MessageType.STATE_DIGEST] = self._handle_state_digest
        self._message_handlers[MessageType.CONSENSUS_VOTE] = self._handle_consensus_vote
        self._message_handlers[MessageType.CONSENSUS_RESULT] = self._handle_consensus_result
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler"""
        self._message_handlers[message_type] = handler
    
    async def send_message(self, message: FederationMessage) -> Optional[Any]:
        """Send a message and optionally wait for response"""
        # Sign message
        message.signature = self._sign_message(message)
        
        # If expecting response, create future
        future = None
        if message.correlation_id:
            future = asyncio.Future()
            self._pending_responses[message.correlation_id] = future
        
        # Send message (simulated)
        await self._network_send(message)
        
        # Wait for response if needed
        if future:
            try:
                result = await asyncio.wait_for(future, timeout=self._message_timeout)
                return result
            except asyncio.TimeoutError:
                del self._pending_responses[message.correlation_id]
                return None
        
        return None
    
    async def broadcast_message(self, message: FederationMessage):
        """Broadcast message to all peers"""
        message.recipient = "broadcast"
        message.signature = self._sign_message(message)
        await self._network_broadcast(message)
    
    async def propose_consensus(self, subject: str, proposal: Dict[str, Any], 
                               timeout: int = 60) -> ConsensusProposal:
        """Propose a consensus decision"""
        proposal_obj = ConsensusProposal(
            proposal_id=f"prop-{int(time.time())}-{self.node_id[:8]}",
            proposer=self.node_id,
            subject=subject,
            proposal=proposal,
            voting_deadline=time.time() + timeout
        )
        
        self._active_proposals[proposal_obj.proposal_id] = proposal_obj
        
        # Broadcast proposal
        message = FederationMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.CONSENSUS_PROPOSAL,
            sender=self.node_id,
            recipient="broadcast",
            payload={
                "proposal_id": proposal_obj.proposal_id,
                "subject": subject,
                "proposal": proposal,
                "voting_deadline": proposal_obj.voting_deadline
            },
            timestamp=time.time()
        )
        
        await self.broadcast_message(message)
        
        # Wait for voting to complete
        await asyncio.sleep(timeout)
        
        # Calculate result
        proposal_obj.calculate_result(self._consensus_threshold)
        
        # Broadcast result
        await self._broadcast_consensus_result(proposal_obj)
        
        return proposal_obj
    
    async def vote_on_proposal(self, proposal_id: str, vote: str, reasoning: str = ""):
        """Vote on a consensus proposal"""
        if proposal_id not in self._active_proposals:
            return
        
        proposal = self._active_proposals[proposal_id]
        
        message = FederationMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.CONSENSUS_VOTE,
            sender=self.node_id,
            recipient=proposal.proposer,
            payload={
                "proposal_id": proposal_id,
                "vote": vote,
                "weight": 1.0,
                "reasoning": reasoning
            },
            timestamp=time.time()
        )
        
        await self.send_message(message)
    
    async def sync_state(self, target_node: Optional[str] = None):
        """Synchronize state with peers"""
        # Calculate state digest
        digest = self._calculate_state_digest()
        self._state_digests[self.node_id] = digest
        
        # Send digest to peers
        message = FederationMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.STATE_DIGEST,
            sender=self.node_id,
            recipient=target_node or "broadcast",
            payload={
                "digest": digest,
                "version": self._state_version,
                "state_size": len(json.dumps(self._local_state))
            },
            timestamp=time.time()
        )
        
        if target_node:
            await self.send_message(message)
        else:
            await self.broadcast_message(message)
    
    async def request_state_sync(self, from_node: str):
        """Request full state sync from a peer"""
        message = FederationMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.STATE_SYNC,
            sender=self.node_id,
            recipient=from_node,
            payload={"request_type": "full_sync"},
            timestamp=time.time(),
            correlation_id=self._generate_message_id()
        )
        
        response = await self.send_message(message)
        if response:
            await self._apply_state_update(response)
    
    def update_local_state(self, key: str, value: Any):
        """Update local state"""
        self._local_state[key] = value
        self._state_version += 1
    
    def get_local_state(self, key: Optional[str] = None) -> Any:
        """Get local state"""
        if key:
            return self._local_state.get(key)
        return self._local_state.copy()
    
    # Message handlers
    async def _handle_heartbeat(self, message: FederationMessage):
        """Handle heartbeat message"""
        self._peers.add(message.sender)
    
    async def _handle_state_digest(self, message: FederationMessage):
        """Handle state digest message"""
        sender_digest = message.payload["digest"]
        self._state_digests[message.sender] = sender_digest
        
        # Check if we need to sync
        if sender_digest != self._state_digests.get(self.node_id):
            # States differ, may need to sync
            pass
    
    async def _handle_consensus_vote(self, message: FederationMessage):
        """Handle consensus vote"""
        proposal_id = message.payload["proposal_id"]
        if proposal_id in self._active_proposals:
            proposal = self._active_proposals[proposal_id]
            proposal.add_vote(
                message.sender,
                message.payload["vote"],
                message.payload.get("weight", 1.0)
            )
    
    async def _handle_consensus_result(self, message: FederationMessage):
        """Handle consensus result"""
        proposal_id = message.payload["proposal_id"]
        if proposal_id in self._active_proposals:
            self._active_proposals[proposal_id].status = message.payload["result"]
    
    # Helper methods
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg-{int(time.time())}-{self.node_id[:8]}-{hash(time.time()) % 10000}"
    
    def _sign_message(self, message: FederationMessage) -> str:
        """Sign a message (simplified)"""
        data = f"{message.message_id}:{message.sender}:{message.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _calculate_state_digest(self) -> str:
        """Calculate digest of current state"""
        state_str = json.dumps(self._local_state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]
    
    async def _apply_state_update(self, state_data: Dict[str, Any]):
        """Apply state update from peer"""
        # Merge state using CRDT semantics
        for key, value in state_data.items():
            if key not in self._local_state:
                self._local_state[key] = value
            else:
                # Simple last-write-wins for now
                if isinstance(value, dict) and "timestamp" in value:
                    if value["timestamp"] > self._local_state[key].get("timestamp", 0):
                        self._local_state[key] = value
        
        self._state_version += 1
    
    async def _broadcast_consensus_result(self, proposal: ConsensusProposal):
        """Broadcast consensus result"""
        message = FederationMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.CONSENSUS_RESULT,
            sender=self.node_id,
            recipient="broadcast",
            payload={
                "proposal_id": proposal.proposal_id,
                "result": proposal.status,
                "votes": proposal.votes,
                "weights": proposal.weights
            },
            timestamp=time.time()
        )
        
        await self.broadcast_message(message)
    
    # Network simulation (to be replaced with real implementation)
    async def _network_send(self, message: FederationMessage):
        """Simulate network send"""
        # In real implementation, send over network
        await self._message_queue.put(message)
    
    async def _network_broadcast(self, message: FederationMessage):
        """Simulate network broadcast"""
        # In real implementation, broadcast over network
        for peer in self._peers:
            msg_copy = FederationMessage(
                message_id=message.message_id,
                message_type=message.message_type,
                sender=message.sender,
                recipient=peer,
                payload=message.payload,
                timestamp=message.timestamp,
                correlation_id=message.correlation_id,
                signature=message.signature
            )
            await self._message_queue.put(msg_copy)
    
    async def process_incoming_messages(self):
        """Process incoming messages (for simulation)"""
        while True:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                
                # Check if this is a response to a pending request
                if message.correlation_id and message.correlation_id in self._pending_responses:
                    future = self._pending_responses[message.correlation_id]
                    future.set_result(message.payload)
                    del self._pending_responses[message.correlation_id]
                    continue
                
                # Route to appropriate handler
                handler = self._message_handlers.get(message.message_type)
                if handler:
                    await handler(message)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing message: {e}")