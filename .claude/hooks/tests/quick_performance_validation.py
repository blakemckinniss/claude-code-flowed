#!/usr/bin/env python3
"""
Quick Performance Validation for Intelligent Feedback System
============================================================

Lightweight performance validation that quickly tests the <100ms target
for stderr feedback generation. Designed for rapid validation.
"""

import sys
import json
import time
import statistics
import subprocess
from pathlib import Path

# Hook path
HOOKS_DIR = Path(__file__).parent.parent
HOOK_PATH = HOOKS_DIR / "post_tool_use.py"

def quick_performance_test():
    """Run quick performance validation."""
    
    print("⚡ QUICK PERFORMANCE VALIDATION - Intelligent Feedback System")
    print("=" * 70)
    print("Target: <100ms stderr feedback generation")
    print(f"Hook: {HOOK_PATH}")
    
    # Test scenarios (lightweight)
    test_scenarios = [
        ("Simple Read", {
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/test.py"},
            "tool_response": {"success": True, "content": "print('hello')"},
            "start_time": time.time()
        }),
        
        ("Write Operation", {
            "tool_name": "Write", 
            "tool_input": {
                "file_path": "/tmp/test_write.py",
                "content": "import os\nprint('test')"
            },
            "tool_response": {"success": True},
            "start_time": time.time()
        }),
        
        ("MCP Tool", {
            "tool_name": "mcp__zen__chat",
            "tool_input": {"prompt": "test", "model": "anthropic/claude-3.5-haiku"},
            "tool_response": {"success": True},
            "start_time": time.time()
        })
    ]
    
    results = []
    
    print("\n🔄 Running Performance Tests...")
    print("-" * 40)
    
    for scenario_name, input_data in test_scenarios:
        print(f"Testing {scenario_name}... ", end="", flush=True)
        
        # Run 5 iterations for quick validation
        times = []
        success_count = 0
        
        for _ in range(5):
            start_time = time.perf_counter()
            
            try:
                process = subprocess.run(
                    [sys.executable, str(HOOK_PATH)],
                    input=json.dumps(input_data),
                    text=True,
                    capture_output=True,
                    timeout=3
                )
                
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # ms
                
                if process.returncode in [0, 2]:  # Success or guidance
                    times.append(execution_time)
                    success_count += 1
                    
            except subprocess.TimeoutExpired:
                times.append(3000)  # Timeout = 3000ms
            except Exception:
                times.append(1000)  # Error = 1000ms
        
        # Calculate metrics
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            
            # Performance assessment
            if avg_time < 50:
                status = "✅ EXCELLENT"
            elif avg_time < 100:
                status = "✅ GOOD"
            elif avg_time < 200:
                status = "⚠️  SLOW"
            else:
                status = "❌ FAILED"
            
            print(f"{avg_time:.1f}ms avg ({min_time:.1f}-{max_time:.1f}ms) {status}")
            
            results.append({
                'scenario': scenario_name,
                'avg_time_ms': avg_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'success_rate': success_count / 5,
                'target_100ms_met': avg_time < 100,
                'target_50ms_met': avg_time < 50
            })
        else:
            print("❌ ALL FAILED")
            results.append({
                'scenario': scenario_name,
                'avg_time_ms': float('inf'),
                'target_100ms_met': False,
                'target_50ms_met': False
            })
    
    # Overall assessment
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 30)
    
    valid_results = [r for r in results if r['avg_time_ms'] != float('inf')]
    
    if valid_results:
        overall_avg = statistics.mean(r['avg_time_ms'] for r in valid_results)
        target_100ms_count = sum(1 for r in valid_results if r['target_100ms_met'])
        target_50ms_count = sum(1 for r in valid_results if r['target_50ms_met'])
        
        print(f"Overall Average: {overall_avg:.1f}ms")
        print(f"Sub-100ms Target: {target_100ms_count}/{len(valid_results)} scenarios ({target_100ms_count/len(valid_results):.1%})")
        print(f"Sub-50ms Target: {target_50ms_count}/{len(valid_results)} scenarios ({target_50ms_count/len(valid_results):.1%})")
        
        print("\n🎯 FINAL ASSESSMENT")
        print("-" * 25)
        
        if overall_avg < 50:
            grade = "A+ - EXCEPTIONAL"
            message = "🎉 Outstanding! Sub-50ms performance achieved!"
        elif overall_avg < 100:
            grade = "A - EXCELLENT" 
            message = "🚀 Excellent! Sub-100ms target met successfully!"
        elif overall_avg < 200:
            grade = "B - GOOD"
            message = "✅ Good performance, minor optimizations recommended"
        else:
            grade = "C - NEEDS IMPROVEMENT"
            message = "⚠️  Performance target not met - optimization required"
        
        print(f"Grade: {grade}")
        print(f"Result: {message}")
        
        # Success criteria
        success = target_100ms_count >= len(valid_results) * 0.8  # 80% must meet 100ms target
        
        print(f"\n{'✅ PERFORMANCE VALIDATION PASSED' if success else '❌ PERFORMANCE VALIDATION FAILED'}")
        
        return success, overall_avg, results
    else:
        print("❌ ALL TESTS FAILED - Hook execution issues")
        return False, float('inf'), results

if __name__ == "__main__":
    success, avg_time, results = quick_performance_test()
    
    # Save quick results
    results_file = HOOKS_DIR / "tests" / "quick_performance_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "success": success,
                "overall_avg_ms": avg_time,
                "results": results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    sys.exit(0 if success else 1)