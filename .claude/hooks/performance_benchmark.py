#!/usr/bin/env python3
"""Performance Benchmark Tool for Hook System Optimization.

This tool measures hook execution performance and validates
the sub-100ms target achievement.
"""

import json
import subprocess
import sys
import time
import statistics
from typing import Dict, Any, List, Tuple
from pathlib import Path
import concurrent.futures
import threading

def run_hook_benchmark(hook_script: str, test_data: Dict[str, Any], 
                      iterations: int = 10) -> Dict[str, Any]:
    """Benchmark a hook script with multiple iterations."""
    execution_times = []
    stderr_lengths = []
    return_codes = []
    errors = []
    
    for i in range(iterations):
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                [sys.executable, hook_script],
                input=json.dumps(test_data),
                text=True,
                capture_output=True,
                timeout=10
            )
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            
            execution_times.append(execution_time_ms)
            stderr_lengths.append(len(result.stderr))
            return_codes.append(result.returncode)
            
        except subprocess.TimeoutExpired:
            errors.append(f"Iteration {i+1}: Timeout")
            execution_times.append(10000)  # 10s timeout
            stderr_lengths.append(0)
            return_codes.append(-1)
        except Exception as e:
            errors.append(f"Iteration {i+1}: {e}")
            execution_times.append(float('inf'))
            stderr_lengths.append(0)
            return_codes.append(-1)
    
    # Calculate statistics
    valid_times = [t for t in execution_times if t != float('inf')]
    
    return {
        "iterations": iterations,
        "execution_times_ms": execution_times,
        "mean_execution_time_ms": statistics.mean(valid_times) if valid_times else float('inf'),
        "median_execution_time_ms": statistics.median(valid_times) if valid_times else float('inf'),
        "p95_execution_time_ms": statistics.quantiles(valid_times, n=20)[18] if len(valid_times) > 19 else max(valid_times) if valid_times else float('inf'),
        "p99_execution_time_ms": statistics.quantiles(valid_times, n=100)[98] if len(valid_times) > 99 else max(valid_times) if valid_times else float('inf'),
        "min_execution_time_ms": min(valid_times) if valid_times else float('inf'),
        "max_execution_time_ms": max(valid_times) if valid_times else float('inf'),
        "std_dev_ms": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
        "mean_stderr_length": statistics.mean(stderr_lengths),
        "success_rate": sum(1 for code in return_codes if code == 0) / len(return_codes),
        "errors": errors,
        "target_met": statistics.mean(valid_times) < 100.0 if valid_times else False
    }

def create_test_scenarios() -> List[Tuple[str, Dict[str, Any]]]:
    """Create test scenarios for benchmarking."""
    return [
        ("Simple Read", {
            "tool_name": "Read",
            "tool_input": {"file_path": "/test.py"},
            "tool_response": {"success": True, "content": "print('hello')"}
        }),
        ("Bash Command", {
            "tool_name": "Bash", 
            "tool_input": {"command": "echo hello"},
            "tool_response": {"success": True, "stdout": "hello"}
        }),
        ("Task Agent", {
            "tool_name": "Task",
            "tool_input": {"agent": "test-agent", "task": "simple task"},
            "tool_response": {"success": True}
        }),
        ("Hook File Violation", {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/home/devcontainers/flowed/.claude/hooks/test.py",
                "content": "import sys\nsys.path.append('/test')\nprint('hello')"
            },
            "tool_response": {"success": True}
        }),
        ("Python File with Issues", {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/home/devcontainers/flowed/test_ruff.py", 
                "content": "import unused\ndef bad_function( ):\n  x=1+2\n  return x"
            },
            "tool_response": {"success": True}
        }),
        ("Error Case", {
            "tool_name": "Read",
            "tool_input": {"file_path": "/nonexistent.py"},
            "tool_response": {"success": False, "error": "File not found"}
        })
    ]

def run_parallel_benchmark(hook_script: str, scenarios: List[Tuple[str, Dict[str, Any]]], 
                          concurrent_executions: int = 5) -> Dict[str, Any]:
    """Run benchmark with concurrent executions to test scalability."""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_executions) as executor:
        for scenario_name, test_data in scenarios:
            print(f"Running parallel benchmark for: {scenario_name}")
            
            # Submit multiple concurrent executions
            futures = []
            for _ in range(concurrent_executions):
                future = executor.submit(run_hook_benchmark, hook_script, test_data, 3)
                futures.append(future)
            
            # Collect results
            concurrent_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    concurrent_results.append(result)
                except Exception as e:
                    concurrent_results.append({"error": str(e)})
            
            # Aggregate results
            all_times = []
            for result in concurrent_results:
                if "execution_times_ms" in result:
                    all_times.extend(result["execution_times_ms"])
            
            valid_times = [t for t in all_times if t != float('inf')]
            
            results[scenario_name] = {
                "concurrent_executions": concurrent_executions,
                "total_executions": len(all_times),
                "mean_execution_time_ms": statistics.mean(valid_times) if valid_times else float('inf'),
                "p95_execution_time_ms": statistics.quantiles(valid_times, n=20)[18] if len(valid_times) > 19 else max(valid_times) if valid_times else float('inf'),
                "target_met_under_load": statistics.mean(valid_times) < 100.0 if valid_times else False,
                "success_rate": len(valid_times) / len(all_times) if all_times else 0
            }
    
    return results

def main():
    """Run comprehensive performance benchmark."""
    print("🚀 Hook Performance Optimization Benchmark")
    print("=" * 60)
    
    hooks_dir = Path("/home/devcontainers/flowed/.claude/hooks")
    
    # Test hooks to benchmark
    hooks_to_test = [
        ("Original", hooks_dir / "post_tool_use.py"),
        ("Optimized", hooks_dir / "post_tool_use_optimized.py")
    ]
    
    scenarios = create_test_scenarios()
    
    all_results = {}
    
    for hook_name, hook_path in hooks_to_test:
        if not hook_path.exists():
            print(f"⚠️  {hook_name} hook not found: {hook_path}")
            continue
            
        print(f"\n📊 Benchmarking {hook_name} Hook")
        print("-" * 40)
        
        hook_results = {}
        
        # Sequential benchmark
        for scenario_name, test_data in scenarios:
            print(f"Testing scenario: {scenario_name}... ", end="", flush=True)
            
            result = run_hook_benchmark(str(hook_path), test_data, iterations=5)
            hook_results[scenario_name] = result
            
            # Print quick result
            mean_time = result["mean_execution_time_ms"]
            target_met = "✅" if result["target_met"] else "❌"
            print(f"{mean_time:.2f}ms {target_met}")
        
        # Parallel benchmark
        print("\n🔄 Running parallel load test...")
        parallel_results = run_parallel_benchmark(str(hook_path), scenarios[:3], concurrent_executions=3)
        hook_results["parallel_load_test"] = parallel_results
        
        all_results[hook_name] = hook_results
    
    # Generate performance report
    print("\n" + "=" * 80)
    print("📈 PERFORMANCE OPTIMIZATION REPORT")  
    print("=" * 80)
    
    for hook_name, results in all_results.items():
        print(f"\n🏷️  {hook_name} Hook Results:")
        print("-" * 50)
        
        sequential_results = {k: v for k, v in results.items() if k != "parallel_load_test"}
        
        # Calculate aggregate metrics
        all_means = [r["mean_execution_time_ms"] for r in sequential_results.values() if r["mean_execution_time_ms"] != float('inf')]
        all_p95s = [r["p95_execution_time_ms"] for r in sequential_results.values() if r["p95_execution_time_ms"] != float('inf')]
        success_rates = [r["success_rate"] for r in sequential_results.values()]
        targets_met = [r["target_met"] for r in sequential_results.values()]
        
        if all_means:
            avg_mean = statistics.mean(all_means)
            avg_p95 = statistics.mean(all_p95s) if all_p95s else float('inf')
            overall_success = statistics.mean(success_rates)
            target_achievement = sum(targets_met) / len(targets_met)
            
            print(f"  Average Response Time: {avg_mean:.2f}ms")
            print(f"  Average P95: {avg_p95:.2f}ms") 
            print(f"  Success Rate: {overall_success:.1%}")
            print(f"  Sub-100ms Target Achievement: {target_achievement:.1%}")
            
            # Performance grade
            if avg_mean < 50:
                grade = "A+ (Excellent)"
            elif avg_mean < 100:
                grade = "A (Target Met)"
            elif avg_mean < 200:
                grade = "B (Good)"
            elif avg_mean < 500:
                grade = "C (Acceptable)"
            else:
                grade = "D (Needs Improvement)"
            
            print(f"  Performance Grade: {grade}")
        
        # Parallel results
        if "parallel_load_test" in results:
            print("\n  🔄 Parallel Load Test Results:")
            parallel_data = results["parallel_load_test"]
            for scenario, data in parallel_data.items():
                mean_time = data["mean_execution_time_ms"]
                target_met = "✅" if data["target_met_under_load"] else "❌"
                print(f"    {scenario}: {mean_time:.2f}ms under load {target_met}")
    
    # Comparison if both hooks available
    if len(all_results) > 1:
        print("\n⚡ OPTIMIZATION IMPACT")
        print("-" * 50)
        
        hooks = list(all_results.keys())
        original_results = all_results[hooks[0]]
        optimized_results = all_results[hooks[1]]
        
        original_means = [r["mean_execution_time_ms"] for r in original_results.values() if isinstance(r, dict) and "mean_execution_time_ms" in r and r["mean_execution_time_ms"] != float('inf')]
        optimized_means = [r["mean_execution_time_ms"] for r in optimized_results.values() if isinstance(r, dict) and "mean_execution_time_ms" in r and r["mean_execution_time_ms"] != float('inf')]
        
        if original_means and optimized_means:
            original_avg = statistics.mean(original_means)
            optimized_avg = statistics.mean(optimized_means)
            
            speedup = original_avg / optimized_avg if optimized_avg > 0 else float('inf')
            improvement = ((original_avg - optimized_avg) / original_avg) * 100
            
            print(f"  Average speedup: {speedup:.2f}x")
            print(f"  Performance improvement: {improvement:.1f}%")
            print(f"  Time reduction: {original_avg - optimized_avg:.2f}ms")
    
    print("\n🎯 SUMMARY")
    print("-" * 50)
    print("Target: Sub-100ms stderr feedback generation")
    
    # Check if target achieved
    target_achieved = False
    if "Optimized" in all_results:
        optimized_data = all_results["Optimized"]
        means = [r["mean_execution_time_ms"] for r in optimized_data.values() if isinstance(r, dict) and "mean_execution_time_ms" in r and r["mean_execution_time_ms"] != float('inf')]
        if means and statistics.mean(means) < 100:
            target_achieved = True
    
    if target_achieved:
        print("✅ SUCCESS: Sub-100ms target achieved!")
        print("🚀 Hook system is now optimized for lightning-fast feedback.")
    else:
        print("❌ Target not fully achieved - further optimization needed.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
