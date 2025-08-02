#!/usr/bin/env python3
"""
Demonstration of Claude Code integration with Enhanced Project Memory Manager
Shows how context injection works for improved development assistance
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the modules path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.project_memory_manager import get_enhanced_memory_manager

async def demo_claude_code_integration():
    """Demonstrate Claude Code context injection"""
    print("=== Claude Code Integration with Enhanced Memory Manager ===\n")
    
    enhanced_manager = get_enhanced_memory_manager()
    
    # Simulate storing some development context
    print("📝 Storing development context memories...")
    
    sample_memories = [
        {
            "key": "authentication_pattern",
            "value": {
                "description": "JWT-based authentication with refresh tokens",
                "implementation": "Used fastapi-users with custom JWT backend",
                "files": ["auth/jwt_backend.py", "auth/models.py"],
                "challenges": "Token refresh logic and session management"
            },
            "category": "architecture",
            "metadata": {"complexity": "medium", "last_used": "2025-08-01"}
        },
        {
            "key": "database_optimization",
            "value": {
                "description": "Optimized user queries using database indexes",
                "solution": "Added composite index on (user_id, created_at)",
                "performance_gain": "60% faster queries",
                "files": ["migrations/001_add_user_indexes.py"]
            },
            "category": "optimization",
            "metadata": {"impact": "high", "measured": True}
        },
        {
            "key": "sparc_methodology",
            "value": {
                "description": "Applied SPARC methodology for feature development",
                "phases": ["specification", "pseudocode", "architecture", "refinement", "completion"],
                "benefits": "Better code quality and fewer bugs",
                "usage": "Used for user management feature"
            },
            "category": "patterns",
            "metadata": {"methodology": "sparc", "success_rate": "95%"}
        },
        {
            "key": "error_handling_pattern",
            "value": {
                "description": "Centralized error handling with custom exceptions",
                "implementation": "FastAPI exception handlers with structured responses",
                "files": ["core/exceptions.py", "core/error_handlers.py"],
                "pattern": "Domain-specific exceptions with HTTP status mapping"
            },
            "category": "architecture",
            "metadata": {"reusable": True, "documented": True}
        }
    ]
    
    # Store sample memories
    for memory in sample_memories:
        success = await enhanced_manager.store_memory(
            memory["key"],
            memory["value"],
            memory["category"],
            metadata=memory["metadata"]
        )
        if success:
            print(f"  ✅ Stored: {memory['key']}")
        else:
            print(f"  ⚠️  Failed to store: {memory['key']}")
    
    print("\n🔍 Testing context injection for different scenarios...\n")
    
    # Test scenarios for Claude Code integration
    test_scenarios = [
        {
            "user_prompt": "How do I implement user authentication in FastAPI?",
            "expected_context": "authentication patterns and architecture"
        },
        {
            "user_prompt": "My database queries are slow, how can I optimize them?",
            "expected_context": "optimization patterns and performance improvements"
        },
        {
            "user_prompt": "What's the best way to structure error handling?",
            "expected_context": "error handling patterns and architecture"
        },
        {
            "user_prompt": "Should I use SPARC methodology for this feature?",
            "expected_context": "development methodologies and patterns"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"🎯 Scenario {i}: {scenario['user_prompt']}")
        
        # Get context injection data
        context_data = await enhanced_manager.inject_context_for_claude_code(scenario['user_prompt'])
        
        print(f"   📊 Context injection enabled: {context_data['injection_enabled']}")
        print(f"   📝 Summary: {context_data['context_summary']}")
        print(f"   🔗 Relevant memories: {len(context_data['relevant_memories'])}")
        
        # Show relevant memories found
        for memory in context_data['relevant_memories']:
            print(f"     - {memory.get('key', 'Unknown')} ({memory.get('category', 'general')})")
        
        print()
    
    # Demonstrate vector search capabilities
    print("🔍 Testing vector search functionality...\n")
    
    search_queries = [
        "authentication and security",
        "performance optimization techniques",
        "error handling best practices",
        "development methodology"
    ]
    
    for query in search_queries:
        print(f"Query: '{query}'")
        results = await enhanced_manager.vector_search_memories(query, top_k=3)
        
        if results:
            print(f"  Found {len(results)} similar memories:")
            for memory, similarity in results:
                print(f"    - {memory.get('key', 'Unknown')} (similarity: {similarity:.3f})")
        else:
            print("  No similar memories found (using fallback search)")
        print()
    
    # Show enhanced statistics
    print("📊 Enhanced Memory Manager Statistics:")
    stats = enhanced_manager.get_enhanced_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n✨ Integration demonstration completed!")
    print("\n💡 How this helps Claude Code:")
    print("   1. Automatic context injection based on user prompts")
    print("   2. Semantic search finds relevant past solutions")
    print("   3. Maintains project-specific knowledge across sessions")
    print("   4. Improves code suggestions with historical context")
    print("   5. Enables pattern recognition for better architecture advice")

if __name__ == "__main__":
    asyncio.run(demo_claude_code_integration())