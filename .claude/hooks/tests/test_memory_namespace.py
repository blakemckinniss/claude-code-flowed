#!/usr/bin/env python3
"""Test memory namespace isolation and functionality"""

import asyncio
import json
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.memory.project_memory_manager import get_memory_manager
from modules.memory.hook_memory_integration import get_hook_memory_integration
from modules.memory.retrieval_patterns import get_retrieval_patterns

async def test_memory_namespace():
    """Test memory namespace functionality"""
    print("🧪 Testing Memory Namespace Integration\n")
    
    # Test 1: Check project configuration
    print("1️⃣ Testing project configuration...")
    memory_manager = get_memory_manager()
    print(f"   Project namespace: {memory_manager.namespace}")
    print(f"   Config loaded: {bool(memory_manager.config)}")
    print(f"   Project name: {memory_manager.config.get('project', {}).get('name', 'Unknown')}")
    
    # Test 2: Store test memory
    print("\n2️⃣ Testing memory storage...")
    test_key = "test/memory_integration"
    test_value = {
        "test": True,
        "timestamp": "2025-08-01T00:00:00Z",
        "description": "Memory namespace test"
    }
    
    success = await memory_manager.store_memory(
        key=test_key,
        value=test_value,
        category="test",
        ttl=3600
    )
    print(f"   Store operation: {'✅ Success' if success else '❌ Failed'}")
    
    # Test 3: Retrieve test memory
    print("\n3️⃣ Testing memory retrieval...")
    retrieved = await memory_manager.retrieve_memory(test_key)
    if retrieved:
        print("   Retrieve operation: ✅ Success")
        print(f"   Retrieved data: {json.dumps(retrieved.get('value', {}), indent=2)}")
    else:
        print("   Retrieve operation: ❌ Failed")
    
    # Test 4: Test semantic analysis
    print("\n4️⃣ Testing semantic context analysis...")
    test_content = "This is a test of the error handling system with optimization patterns"
    analysis = memory_manager.analyze_semantic_context(test_content)
    print(f"   Relevance score: {analysis['relevance_score']:.2f}")
    print(f"   Categories detected: {', '.join(analysis['categories'])}")
    print(f"   Should store: {'✅ Yes' if analysis['should_store'] else '❌ No'}")
    
    # Test 5: Test search functionality
    print("\n5️⃣ Testing memory search...")
    search_results = await memory_manager.search_memories("test")
    print(f"   Search results: {len(search_results)} found")
    
    # Test 6: Test hook integration
    print("\n6️⃣ Testing hook memory integration...")
    hook_integration = get_hook_memory_integration()
    
    # Simulate pre-tool capture
    await hook_integration.capture_pre_tool_memory(
        "TestTool",
        {"action": "test", "target": "memory_system"}
    )
    
    # Simulate post-tool capture
    await hook_integration.capture_post_tool_memory(
        "TestTool",
        {"success": True, "result": "Test completed"},
        0.123
    )
    
    session_stats = hook_integration.get_session_stats()
    print(f"   Session ID: {session_stats['session_id']}")
    print(f"   Memories created: {session_stats['memories_created']}")
    print(f"   Namespace: {session_stats['namespace']}")
    
    # Test 7: Test retrieval patterns
    print("\n7️⃣ Testing retrieval patterns...")
    retrieval = get_retrieval_patterns()
    
    # Test contextual retrieval
    contextual_memories = await retrieval.get_contextual_memories(
        "Write",
        {"file_path": "/test/file.py", "description": "Update test file"}
    )
    print(f"   Contextual memories found: {len(contextual_memories)}")
    
    # Test error prevention
    error_memories = await retrieval.get_error_prevention_memories(
        "Bash",
        {"command": "rm -rf /"}
    )
    print(f"   Error prevention memories: {len(error_memories)}")
    
    print("\n✅ Memory namespace testing complete!")
    
    # Cleanup
    stats = memory_manager.get_memory_stats()
    print("\n📊 Final statistics:")
    print(f"   Cached items: {stats['cached_items']}")
    print(f"   Categories: {', '.join(stats['categories'])}")
    print(f"   Semantic enabled: {'✅' if stats['semantic_enabled'] else '❌'}")

if __name__ == "__main__":
    asyncio.run(test_memory_namespace())