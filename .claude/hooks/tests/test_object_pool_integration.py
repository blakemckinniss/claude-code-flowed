#!/usr/bin/env python3
"""Test object pool integration with validation workflow."""

import sys
import os
import json
import time
import threading
from pathlib import Path

# Add the hooks directory to the path
hooks_dir = Path(__file__).parent
sys.path.insert(0, str(hooks_dir))

from modules.pre_tool.manager import PreToolAnalysisManager
from modules.optimization.object_pool import get_object_pools, get_pool_stats


def test_object_pool_integration():
    """Test object pool integration with the validation manager."""
    print("🧪 Testing Object Pool Integration with Validation Workflow...")
    
    # Initialize the manager (this should initialize object pools)
    manager = PreToolAnalysisManager()
    print(f"✅ Manager initialized with {len(manager.validators)} validators")
    
    # Get initial pool stats
    initial_stats = get_pool_stats()
    print("📊 Initial Pool Stats:")
    for pool_name, stats in initial_stats.items():
        print(f"   {pool_name}: {stats.current_size} objects, {stats.created_objects} created")
    
    # Test validation with various tools to exercise object pooling
    test_tools = [
        ("TodoWrite", {"todos": [{"content": "Test task", "status": "pending", "priority": "medium", "id": "test-1"}]}),
        ("Read", {"file_path": "/test/path.txt"}),
        ("Task", {"subagent_type": "coder", "description": "Test task", "prompt": "Test prompt"}),
        ("Bash", {"command": "echo test", "description": "Test command"}),
        ("Write", {"file_path": "/test/new_file.txt", "content": "Test content"})
    ]
    
    validation_count = 0
    successful_validations = 0
    
    print(f"\n🔄 Running {len(test_tools)} validation tests...")
    
    for tool_name, tool_input in test_tools:
        try:
            print(f"   Testing {tool_name}...")
            
            # Run validation (this should use object pools)
            result = manager.validate_tool_usage(tool_name, tool_input)
            validation_count += 1
            
            if result is None:
                print(f"     ✅ {tool_name}: No blocking issues")
                successful_validations += 1
            else:
                print(f"     ⚠️  {tool_name}: {result.get('severity', 'UNKNOWN')} - {result.get('message', 'No message')[:100]}...")
                successful_validations += 1
                
        except Exception as e:
            print(f"     ❌ {tool_name}: Error - {e}")
    
    # Get final pool stats
    final_stats = get_pool_stats()
    print("\n📊 Final Pool Stats:")
    for pool_name, stats in final_stats.items():
        initial = initial_stats.get(pool_name)
        borrowed_increase = stats.borrowed_objects - (initial.borrowed_objects if initial else 0)
        returned_increase = stats.returned_objects - (initial.returned_objects if initial else 0)
        
        print(f"   {pool_name}:")
        print(f"     Current Size: {stats.current_size}")
        print(f"     Objects Borrowed: +{borrowed_increase} (total: {stats.borrowed_objects})")
        print(f"     Objects Returned: +{returned_increase} (total: {stats.returned_objects})")
        print(f"     Hit Rate: {stats.get_hit_rate():.1%}")
        print(f"     Efficiency: {stats.get_efficiency_score():.1%}")
    
    # Test pool performance under concurrent load
    print("\n⚡ Testing Concurrent Pool Performance...")
    
    def concurrent_validation_test(thread_id: int, iterations: int):
        """Run validations concurrently to test thread safety."""
        local_success = 0
        for i in range(iterations):
            try:
                tool_name = f"Test_{thread_id}_{i}"
                tool_input = {"test_data": f"thread_{thread_id}_iteration_{i}"}
                manager.validate_tool_usage(tool_name, tool_input)
                local_success += 1
            except Exception as e:
                print(f"     Thread {thread_id} error on iteration {i}: {e}")
        return local_success
    
    # Run concurrent validations
    num_threads = 4
    iterations_per_thread = 10
    threads = []
    start_time = time.time()
    
    for thread_id in range(num_threads):
        thread = threading.Thread(
            target=concurrent_validation_test,
            args=(thread_id, iterations_per_thread)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    concurrent_time = time.time() - start_time
    total_concurrent_operations = num_threads * iterations_per_thread
    
    print(f"   Completed {total_concurrent_operations} concurrent validations in {concurrent_time:.3f}s")
    print(f"   Average: {concurrent_time/total_concurrent_operations*1000:.2f}ms per validation")
    
    # Get post-concurrent stats
    concurrent_stats = get_pool_stats()
    print("\n📊 Post-Concurrent Pool Stats:")
    for pool_name, stats in concurrent_stats.items():
        final = final_stats.get(pool_name)
        borrowed_increase = stats.borrowed_objects - (final.borrowed_objects if final else 0)
        
        print(f"   {pool_name}:")
        print(f"     Additional Borrows: +{borrowed_increase}")
        print(f"     Current Hit Rate: {stats.get_hit_rate():.1%}")
        print(f"     Current Efficiency: {stats.get_efficiency_score():.1%}")
    
    # Get manager status including object pool performance
    print("\n📋 Manager Status with Object Pool Integration:")
    try:
        status = manager.get_validator_status()
        print(f"   Total Validations: {status['total_validations']}")
        print(f"   Parallel Validation: {status['parallel_validation_enabled']}")
        
        if "object_pool_performance" in status:
            pool_perf = status["object_pool_performance"]
            print("   Object Pool Performance:")
            for pool_name, perf in pool_perf.items():
                print(f"     {pool_name}: {perf['hit_rate']:.1%} hit rate, {perf['efficiency']:.1%} efficiency")
        
        if "cache_performance" in status:
            cache_perf = status["cache_performance"]
            print(f"   Cache Performance: {cache_perf.get('hit_rate', 0):.1%} hit rate")
            
    except Exception as e:
        print(f"   ❌ Error getting manager status: {e}")
    
    # Final summary
    print("\n✅ Object Pool Integration Test Summary:")
    print(f"   Sequential Validations: {validation_count} ({successful_validations} successful)")
    print(f"   Concurrent Validations: {total_concurrent_operations}")
    print(f"   Total Operations: {validation_count + total_concurrent_operations}")
    
    # Check if object pools were actually used
    pools_used = any(stats.borrowed_objects > 0 for stats in concurrent_stats.values())
    if pools_used:
        print("   🎯 Object pools successfully integrated and used!")
        
        # Calculate memory efficiency gains
        total_borrows = sum(stats.borrowed_objects for stats in concurrent_stats.values())
        total_creates = sum(stats.created_objects for stats in concurrent_stats.values())
        if total_borrows > 0:
            reuse_rate = (total_borrows - total_creates) / total_borrows
            print(f"   💾 Memory Efficiency: {reuse_rate:.1%} object reuse rate")
            if reuse_rate > 0.5:
                print("   🚀 Excellent memory efficiency achieved!")
            elif reuse_rate > 0.2:
                print("   👍 Good memory efficiency achieved!")
    else:
        print("   ⚠️  Object pools were not used - may need configuration check")
    
    return pools_used and successful_validations > 0


if __name__ == "__main__":
    try:
        success = test_object_pool_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)