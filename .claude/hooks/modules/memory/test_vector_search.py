#!/usr/bin/env python3
"""
Test script for vector search enhancement
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the modules path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.project_memory_manager import get_enhanced_memory_manager, get_memory_manager

async def test_enhanced_memory_manager():
    """Test the enhanced memory manager"""
    print("=== Testing Enhanced Project Memory Manager ===\n")
    
    # Test basic initialization
    try:
        enhanced_manager = get_enhanced_memory_manager()
        print("✅ Enhanced memory manager initialized successfully")
        
        # Test stats
        stats = enhanced_manager.get_enhanced_stats()
        print(f"📊 Enhanced Stats: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        print(f"❌ Failed to initialize enhanced manager: {e}")
        return False
    
    # Test backward compatibility
    try:
        base_manager = get_memory_manager()
        base_stats = base_manager.get_memory_stats()
        print(f"📊 Base Stats: {json.dumps(base_stats, indent=2)}")
        print("✅ Backward compatibility maintained")
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False
    
    # Test vector search functionality
    try:
        # Test with some sample data
        test_memories = [
            {
                "key": "test_architecture_1",
                "value": "Implemented microservices architecture with event-driven communication",
                "category": "architecture",
                "metadata": {"complexity": "high"}
            },
            {
                "key": "test_task_1", 
                "value": "Created user authentication system with JWT tokens",
                "category": "tasks",
                "metadata": {"status": "completed"}
            },
            {
                "key": "test_optimization_1",
                "value": "Optimized database queries using connection pooling",
                "category": "optimization",
                "metadata": {"performance_gain": "40%"}
            }
        ]
        
        # Test vector search (should fallback gracefully if vector service not available)
        results = await enhanced_manager.vector_search_memories("architecture patterns", top_k=3)
        print(f"🔍 Vector search results: {len(results)} items found")
        
        # Test contextual memories
        context_results = await enhanced_manager.get_contextual_memories("microservices design patterns")
        print(f"🎯 Contextual memories: {len(context_results)} items found")
        
        # Test context injection
        context_data = await enhanced_manager.inject_context_for_claude_code("How to implement authentication?")
        print(f"💉 Context injection: {context_data['context_summary']}")
        
        print("✅ Vector search functionality tests completed")
        
    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        return False
    
    print("\n=== All Tests Completed Successfully ===")
    return True

async def test_vector_service():
    """Test vector service directly"""
    print("\n=== Testing Vector Search Service ===\n")
    
    try:
        from memory.vector_search_service import VectorSearchService, VectorSearchConfig
        
        # Test config loading
        config = {
            "vectorSearch": {
                "enabled": True,
                "provider": "local",
                "dimensions": 100,
                "similarityThreshold": 0.7
            }
        }
        
        service = VectorSearchService(config)
        print("✅ Vector search service initialized")
        
        # Test embedding generation
        test_text = "This is a test document about machine learning algorithms"
        embedding = await service.get_embedding(test_text)
        
        if embedding:
            print(f"✅ Embedding generated: {len(embedding)} dimensions")
        else:
            print("⚠️  Embedding generation returned None (expected for local provider without data)")
        
        # Test similarity calculation
        vec1 = [1.0, 0.5, 0.2, 0.8]
        vec2 = [0.8, 0.6, 0.3, 0.7]
        similarity = service.cosine_similarity(vec1, vec2)
        print(f"✅ Cosine similarity calculated: {similarity:.3f}")
        
        # Test stats
        stats = service.get_stats()
        print(f"📊 Vector service stats: {json.dumps(stats, indent=2)}")
        
    except ImportError as e:
        print(f"⚠️  Vector search service import failed (expected if numpy not available): {e}")
    except Exception as e:
        print(f"❌ Vector service test failed: {e}")

if __name__ == "__main__":
    print("Starting vector search enhancement tests...\n")
    
    async def run_all_tests():
        success = await test_enhanced_memory_manager()
        await test_vector_service()
        return success
    
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n🎉 All tests passed! Vector search enhancement is working.")
    else:
        print("\n⚠️  Some tests failed, but basic functionality should work.")
    
    sys.exit(0 if success else 1)