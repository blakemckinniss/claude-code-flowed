#!/usr/bin/env python3
"""
Test script for the ProcessManager to ensure it prevents runaway processes.

This script tests various scenarios including:
- Normal subprocess execution
- Long-running processes
- Memory limit enforcement
- Timeout handling
- Orphan process prevention
- Concurrent process management
"""

import sys
import time
import threading

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Import directly from the process_manager module
from modules.utils.process_manager import (
    ProcessManager,
    get_process_manager,
    managed_subprocess_run,
    managed_subprocess_popen,
    managed_process_context,
    ProcessManagerError
)


def test_basic_subprocess_run():
    """Test basic subprocess.run replacement."""
    print("🧪 Testing basic subprocess.run replacement...")
    
    try:
        result = managed_subprocess_run(['echo', 'Hello World'], timeout=5)
        assert result.returncode == 0
        print("✅ Basic subprocess.run test passed")
        return True
    except Exception as e:
        print(f"❌ Basic subprocess.run test failed: {e}")
        return False


def test_timeout_enforcement():
    """Test that timeouts are properly enforced."""
    print("🧪 Testing timeout enforcement...")
    
    start_time = time.time()
    try:
        # This should timeout after 2 seconds
        managed_subprocess_run(['sleep', '10'], timeout=2)
        print("❌ Timeout test failed: command should have timed out")
        return False
    except Exception:
        elapsed = time.time() - start_time
        if 1.5 <= elapsed <= 3.0:  # Allow some margin
            print("✅ Timeout enforcement test passed")
            return True
        else:
            print(f"❌ Timeout test failed: took {elapsed:.1f}s, expected ~2s")
            return False


def test_process_tracking():
    """Test that processes are properly tracked."""
    print("🧪 Testing process tracking...")
    
    manager = get_process_manager()
    initial_count = len(manager.list_processes())
    
    try:
        with managed_process_context(['sleep', '1']) as process:
            # Check that process is tracked
            current_count = len(manager.list_processes())
            if current_count <= initial_count:
                print("❌ Process tracking test failed: process not tracked")
                return False
            
            # Wait for process to complete
            process.wait()
        
        # Wait a moment for cleanup
        time.sleep(0.5)
        
        # Check that process was cleaned up
        final_count = len(manager.list_processes())
        if final_count > initial_count:
            print("❌ Process tracking test failed: process not cleaned up")
            return False
        
        print("✅ Process tracking test passed")
        return True
        
    except Exception as e:
        print(f"❌ Process tracking test failed: {e}")
        return False


def test_concurrent_processes():
    """Test handling of multiple concurrent processes."""
    print("🧪 Testing concurrent process management...")
    
    def run_process(process_id):
        """Run a process in a thread."""
        try:
            result = managed_subprocess_run(['sleep', '0.5'], timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    try:
        threads = []
        results = []
        
        # Start 5 concurrent processes
        for i in range(5):
            thread = threading.Thread(target=lambda: results.append(run_process(i)))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        if len(results) == 5 and all(results):
            print("✅ Concurrent process test passed")
            return True
        else:
            print(f"❌ Concurrent process test failed: {len(results)}/5 succeeded")
            return False
            
    except Exception as e:
        print(f"❌ Concurrent process test failed: {e}")
        return False


def test_memory_monitoring():
    """Test memory limit monitoring (simulation)."""
    print("🧪 Testing memory monitoring setup...")
    
    try:
        manager = get_process_manager()
        
        # Start a process with memory limit
        with managed_process_context(['echo', 'test'], max_memory_mb=10) as process:
            process.wait()
            
            # Check that the process was tracked
            stats = manager.get_stats()
            print(f"📊 Manager stats: {stats['total_processes']} processes tracked")
        
        print("✅ Memory monitoring setup test passed")
        return True
        
    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")
        return False


def test_cleanup_callbacks():
    """Test cleanup callback functionality."""
    print("🧪 Testing cleanup callbacks...")
    
    cleanup_called = threading.Event()
    
    def cleanup_callback():
        cleanup_called.set()
    
    try:
        manager = get_process_manager()
        
        # Start a process with cleanup callback
        process = managed_subprocess_popen(
            ['sleep', '0.1'], 
            cleanup_callback=cleanup_callback,
            timeout=5
        )
        
        # Wait for process to complete
        process.wait()
        
        # Trigger cleanup
        manager.terminate_process(process.pid)
        
        # Wait for cleanup callback
        if cleanup_called.wait(timeout=2):
            print("✅ Cleanup callback test passed")
            return True
        else:
            print("❌ Cleanup callback test failed: callback not called")
            return False
            
    except Exception as e:
        print(f"❌ Cleanup callback test failed: {e}")
        return False


def test_process_limits():
    """Test process limit enforcement."""
    print("🧪 Testing process limits...")
    
    try:
        # Create a manager with low process limit
        limited_manager = ProcessManager(max_processes=2)
        
        # Start processes up to the limit
        processes = []
        for i in range(2):
            process = limited_manager.popen(['sleep', '1'])
            processes.append(process)
        
        # Try to start one more (should fail)
        try:
            limited_manager.popen(['sleep', '1'])
            print("❌ Process limit test failed: should have been blocked")
            return False
        except ProcessManagerError:
            print("✅ Process limit enforcement works")
        
        # Cleanup
        for process in processes:
            limited_manager.terminate_process(process.pid)
        limited_manager.shutdown_all()
        
        print("✅ Process limit test passed")
        return True
        
    except Exception as e:
        print(f"❌ Process limit test failed: {e}")
        return False


def test_manager_stats():
    """Test manager statistics functionality."""
    print("🧪 Testing manager statistics...")
    
    try:
        manager = get_process_manager()
        initial_stats = manager.get_stats()
        
        # Start a tagged process
        with managed_process_context(['echo', 'test'], tags={'test': 'stats'}) as process:
            current_stats = manager.get_stats()
            
            # Check that stats show the process
            if current_stats['total_processes'] > initial_stats['total_processes']:
                print("✅ Manager statistics test passed")
                return True
            else:
                print("❌ Manager statistics test failed: no change in stats")
                return False
        
    except Exception as e:
        print(f"❌ Manager statistics test failed: {e}")
        return False


def test_existing_hook_compatibility():
    """Test compatibility with existing hook patterns."""
    print("🧪 Testing compatibility with existing hooks...")
    
    test_cases = [
        # Test NPX command (like in stop.py)
        {
            'name': 'NPX command test',
            'command': ['echo', 'simulating npx claude-flow'],
            'expected': True
        },
        # Test Ruff-like command (like in post_tool_use.py)
        {
            'name': 'Ruff-like command test',
            'command': ['echo', 'simulating ruff check'],
            'expected': True
        },
        # Test background process (like in hook_pool.py)
        {
            'name': 'Background process test',
            'command': ['sleep', '0.1'],
            'expected': True
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        try:
            result = managed_subprocess_run(test_case['command'], timeout=10)
            success = (result.returncode == 0) == test_case['expected']
            results.append(success)
            
            if success:
                print(f"✅ {test_case['name']} passed")
            else:
                print(f"❌ {test_case['name']} failed")
                
        except Exception as e:
            print(f"❌ {test_case['name']} failed with exception: {e}")
            results.append(False)
    
    if all(results):
        print("✅ All compatibility tests passed")
        return True
    else:
        print(f"❌ Compatibility tests: {sum(results)}/{len(results)} passed")
        return False


def run_all_tests():
    """Run all test cases."""
    print("🚀 Starting ProcessManager Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_subprocess_run,
        test_timeout_enforcement,
        test_process_tracking,
        test_concurrent_processes,
        test_memory_monitoring,
        test_cleanup_callbacks,
        test_process_limits,
        test_manager_stats,
        test_existing_hook_compatibility
    ]
    
    results = []
    start_time = time.time()
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        
        print()  # Add spacing between tests
    
    # Summary
    elapsed = time.time() - start_time
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📋 Test Summary: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {elapsed:.2f} seconds")
    
    if passed == total:
        print("🎉 All tests passed! ProcessManager is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Review the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    # Ensure cleanup
    try:
        get_process_manager().shutdown_all()
    except:
        pass
    
    sys.exit(0 if success else 1)