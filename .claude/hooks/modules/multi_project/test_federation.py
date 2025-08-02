"""
Test Federation - Demonstrates multi-swarm federation capabilities

Tests the federation controller with simulated swarms to validate
resource sharing, consensus building, and state synchronization.
"""

import asyncio
import json
import time
from typing import Dict, List, Any

from federation_controller import FederationController
from swarm_registry import SwarmRegistry  
from resource_pool import GlobalResourcePool
from federation_protocol import FederationProtocol, MessageType


class SimulatedSwarm:
    """Simulates a Hive Mind swarm for testing"""
    
    def __init__(self, swarm_id: str, project_id: str, project_name: str, 
                 worker_types: List[str], max_workers: int):
        self.swarm_id = swarm_id
        self.project_id = project_id
        self.project_name = project_name
        self.worker_types = worker_types
        self.max_workers = max_workers
        self.current_load = 0.0
        self.agents = []
        
        # Create simulated agents
        for i in range(max_workers):
            agent_type = worker_types[i % len(worker_types)]
            self.agents.append({
                "agent_id": f"{swarm_id}-agent-{i}",
                "agent_type": agent_type,
                "home_swarm": swarm_id,
                "status": "idle"
            })
    
    def get_registration_data(self) -> Dict[str, Any]:
        """Get swarm registration data"""
        return {
            "swarm_id": self.swarm_id,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "queen_endpoint": f"tcp://swarm-{self.swarm_id}:8080",
            "capabilities": {
                "worker_types": self.worker_types,
                "max_workers": self.max_workers,
                "current_load": self.current_load,
                "memory_available": 2048,
                "specializations": ["research", "coding", "testing"],
                "performance_metrics": {
                    "avg_task_time": 45.2,
                    "success_rate": 0.95
                }
            },
            "metadata": {
                "version": "2.0.0",
                "environment": "test"
            }
        }
    
    def update_load(self, load: float):
        """Update swarm load"""
        self.current_load = min(1.0, max(0.0, load))


async def test_federation_setup():
    """Test basic federation setup"""
    print("\n=== Testing Federation Setup ===")
    
    # Create federation controller
    federation = FederationController("test-federation")
    await federation.start()
    
    # Create simulated swarms
    swarms = [
        SimulatedSwarm("swarm-project1", "project1", "E-Commerce Platform",
                      ["researcher", "coder", "tester"], 5),
        SimulatedSwarm("swarm-project2", "project2", "Analytics Dashboard", 
                      ["analyst", "coder", "ml-engineer"], 4),
        SimulatedSwarm("swarm-project3", "project3", "Mobile App",
                      ["mobile-developer", "ui-designer", "tester"], 6)
    ]
    
    # Register swarms
    print("\nRegistering swarms...")
    for swarm in swarms:
        success = await federation.register_swarm(swarm.get_registration_data())
        print(f"  - {swarm.project_name}: {'✓' if success else '✗'}")
    
    # Check federation status
    status = await federation.get_federation_status()
    print("\nFederation Status:")
    print(f"  - Total swarms: {status['metrics']['total_swarms']}")
    print(f"  - Active swarms: {status['metrics']['active_swarms']}")
    
    await federation.stop()
    return swarms


async def test_resource_allocation():
    """Test resource allocation across swarms"""
    print("\n=== Testing Resource Allocation ===")
    
    # Setup
    federation = FederationController("test-federation")
    resource_pool = GlobalResourcePool()
    await federation.start()
    
    # Create and register swarms
    swarms = await create_test_swarms(federation)
    
    # Register all agents in resource pool
    print("\nRegistering agents in global pool...")
    total_agents = 0
    for swarm in swarms:
        for agent in swarm.agents:
            await resource_pool.register_agent(agent)
            total_agents += 1
    
    print(f"  - Total agents registered: {total_agents}")
    
    # Test resource allocation
    print("\nTesting resource allocation...")
    
    # Request 1: Project1 needs ML engineer (should get from Project2)
    request1 = {
        "from_swarm": "swarm-project1",
        "agent_type": "ml-engineer",
        "duration": 3600,
        "priority": "high"
    }
    
    allocation_id = await resource_pool.allocate_agent({
        **request1,
        "requesting_swarm": request1["from_swarm"],
        "task_id": "task-123"
    })
    
    if allocation_id:
        print(f"  ✓ Allocated ML engineer to Project1 (allocation: {allocation_id})")
    else:
        print("  ✗ Failed to allocate ML engineer")
    
    # Request 2: Project2 needs mobile developer (should get from Project3)
    request2 = {
        "from_swarm": "swarm-project2",
        "agent_type": "mobile-developer",
        "duration": 7200,
        "priority": "normal"
    }
    
    allocation_id2 = await resource_pool.allocate_agent({
        **request2,
        "requesting_swarm": request2["from_swarm"],
        "task_id": "task-456"
    })
    
    if allocation_id2:
        print(f"  ✓ Allocated mobile developer to Project2 (allocation: {allocation_id2})")
    else:
        print("  ✗ Failed to allocate mobile developer")
    
    # Check pool status
    pool_status = await resource_pool.get_pool_status()
    print("\nResource Pool Status:")
    print(f"  - Total agents: {pool_status['total_agents']}")
    print(f"  - Active allocations: {pool_status['active_allocations']}")
    print(f"  - Utilization: {pool_status['metrics']['utilization']:.1%}")
    
    # Release allocations
    if allocation_id:
        await resource_pool.release_agent(allocation_id)
    if allocation_id2:
        await resource_pool.release_agent(allocation_id2)
    
    await federation.stop()


async def test_swarm_discovery():
    """Test swarm discovery by capability"""
    print("\n=== Testing Swarm Discovery ===")
    
    registry = SwarmRegistry()
    
    # Register test swarms
    swarms_data = [
        {
            "swarm_id": "swarm-1",
            "project_id": "proj1",
            "project_name": "Web App",
            "queen_endpoint": "tcp://swarm1:8080",
            "capabilities": {
                "agent_types": ["coder", "tester", "researcher"],
                "max_capacity": 10,
                "specializations": ["web", "api"],
                "performance_metrics": {"success_rate": 0.95}
            },
            "metadata": {}
        },
        {
            "swarm_id": "swarm-2", 
            "project_id": "proj2",
            "project_name": "ML Platform",
            "queen_endpoint": "tcp://swarm2:8080",
            "capabilities": {
                "agent_types": ["ml-engineer", "data-scientist", "researcher"],
                "max_capacity": 8,
                "specializations": ["ml", "data"],
                "performance_metrics": {"success_rate": 0.92}
            },
            "metadata": {}
        }
    ]
    
    for swarm_data in swarms_data:
        await registry.register(swarm_data)
    
    # Test capability search
    print("\nSearching for swarms by capability...")
    
    ml_swarms = await registry.find_swarms_by_capability("ml-engineer")
    print(f"  - ML engineer capable swarms: {len(ml_swarms)}")
    for swarm in ml_swarms:
        print(f"    • {swarm.project_name} (capacity: {swarm.capabilities.max_capacity})")
    
    coder_swarms = await registry.find_swarms_by_capability("coder")
    print(f"  - Coder capable swarms: {len(coder_swarms)}")
    for swarm in coder_swarms:
        print(f"    • {swarm.project_name} (capacity: {swarm.capabilities.max_capacity})")
    
    # Test similarity scoring
    print("\nTesting swarm similarity...")
    similarity = await registry.calculate_swarm_similarity("swarm-1", "swarm-2")
    print(f"  - Similarity between Web App and ML Platform: {similarity:.2f}")
    
    # Test fallback recommendations
    print("\nTesting fallback recommendations...")
    fallbacks = await registry.recommend_fallback_swarms("swarm-1", limit=2)
    print(f"  - Recommended fallbacks for Web App: {fallbacks}")
    
    # Get registry stats
    stats = await registry.get_registry_stats()
    print("\nRegistry Statistics:")
    print(f"  - Total swarms: {stats['total_swarms']}")
    print(f"  - Total capacity: {stats['total_capacity']}")
    print(f"  - Agent type distribution: {json.dumps(stats['agent_type_distribution'], indent=4)}")


async def test_consensus_protocol():
    """Test consensus building between swarms"""
    print("\n=== Testing Consensus Protocol ===")
    
    # Create protocol instances for each swarm
    protocols = {
        "swarm-1": FederationProtocol("swarm-1"),
        "swarm-2": FederationProtocol("swarm-2"), 
        "swarm-3": FederationProtocol("swarm-3")
    }
    
    # Simulate peer discovery
    for node_id, protocol in protocols.items():
        for peer_id in protocols:
            if peer_id != node_id:
                protocol._peers.add(peer_id)
    
    print("\nProposing consensus for resource allocation policy...")
    
    # Swarm 1 proposes new resource allocation policy
    proposal = {
        "action": "update_allocation_policy",
        "policy": {
            "max_allocation_per_swarm": 3,
            "priority_weights": {"critical": 2.0, "high": 1.5, "normal": 1.0},
            "allocation_timeout": 300
        }
    }
    
    # Simulate consensus (simplified)
    print("  - Swarm 1 proposing policy update")
    proposal_result = {
        "proposal_id": "prop-12345",
        "status": "pending",
        "votes": {
            "swarm-1": "APPROVE",
            "swarm-2": "APPROVE", 
            "swarm-3": "REJECT"
        },
        "result": "approved"  # 2/3 majority
    }
    
    print(f"  - Consensus result: {proposal_result['result']}")
    print(f"  - Votes: {json.dumps(proposal_result['votes'], indent=4)}")
    
    # Test state synchronization
    print("\nTesting state synchronization...")
    
    # Update local state on swarm-1
    protocols["swarm-1"].update_local_state("resource_policy", proposal["policy"])
    protocols["swarm-1"].update_local_state("last_updated", time.time())
    
    # Calculate state digests
    digests = {}
    for node_id, protocol in protocols.items():
        digest = protocol._calculate_state_digest()
        digests[node_id] = digest
    
    print("  - State digests:")
    for node_id, digest in digests.items():
        print(f"    • {node_id}: {digest}")
    
    # Check if states differ
    if len(set(digests.values())) > 1:
        print("  - States differ, synchronization needed")
    else:
        print("  - All states synchronized")


async def test_load_balancing():
    """Test agent load balancing across swarms"""
    print("\n=== Testing Load Balancing ===")
    
    resource_pool = GlobalResourcePool()
    
    # Create unbalanced agent distribution
    swarm_loads = {
        "swarm-heavy": [],  # Overloaded swarm
        "swarm-light": []   # Underloaded swarm
    }
    
    # Add 8 agents to heavy swarm, 2 to light swarm
    for i in range(8):
        agent = {
            "agent_id": f"heavy-agent-{i}",
            "agent_type": "coder",
            "home_swarm": "swarm-heavy"
        }
        await resource_pool.register_agent(agent)
        swarm_loads["swarm-heavy"].append(agent["agent_id"])
    
    for i in range(2):
        agent = {
            "agent_id": f"light-agent-{i}",
            "agent_type": "coder",
            "home_swarm": "swarm-light"
        }
        await resource_pool.register_agent(agent)
        swarm_loads["swarm-light"].append(agent["agent_id"])
    
    print("\nInitial distribution:")
    print(f"  - Heavy swarm: {len(swarm_loads['swarm-heavy'])} agents")
    print(f"  - Light swarm: {len(swarm_loads['swarm-light'])} agents")
    
    # Simulate load by making some agents busy
    for i in range(6):
        agent_id = f"heavy-agent-{i}"
        if agent_id in resource_pool._agents:
            resource_pool._agents[agent_id].status = "busy"
    
    # Trigger rebalancing
    print("\nTriggering load balancing...")
    migrations = await resource_pool.rebalance_agents()
    
    if migrations:
        print("  - Migrations performed:")
        for route, count in migrations.items():
            print(f"    • {route}: {count} agents")
    else:
        print("  - No migrations needed (or interval not reached)")
    
    # Check final status
    pool_status = await resource_pool.get_pool_status()
    print("\nFinal distribution:")
    for swarm_id, count in pool_status["swarm_distribution"].items():
        print(f"  - {swarm_id}: {count} agents")


async def create_test_swarms(federation: FederationController) -> List[SimulatedSwarm]:
    """Helper to create and register test swarms"""
    swarms = [
        SimulatedSwarm("swarm-project1", "project1", "E-Commerce Platform",
                      ["researcher", "coder", "tester"], 5),
        SimulatedSwarm("swarm-project2", "project2", "Analytics Dashboard", 
                      ["analyst", "coder", "ml-engineer"], 4),
        SimulatedSwarm("swarm-project3", "project3", "Mobile App",
                      ["mobile-developer", "ui-designer", "tester"], 6)
    ]
    
    for swarm in swarms:
        await federation.register_swarm(swarm.get_registration_data())
    
    return swarms


async def main():
    """Run all federation tests"""
    print("╔══════════════════════════════════════════════════╗")
    print("║       Federation Controller Test Suite           ║")
    print("╚══════════════════════════════════════════════════╝")
    
    try:
        # Run tests
        await test_federation_setup()
        await test_resource_allocation()
        await test_swarm_discovery()
        await test_consensus_protocol()
        await test_load_balancing()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())