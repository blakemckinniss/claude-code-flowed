#!/usr/bin/env python3
"""
Performance Benchmark Suite for Intelligent Feedback System
===========================================================

Comprehensive performance benchmarking framework that measures:
- Stderr feedback generation time (target: <100ms)
- Memory efficiency during feedback generation
- Throughput under concurrent load
- Performance regression detection
- Real-world scenario simulation

Designed specifically for the intelligent feedback system validation.
"""

import sys
import os
import json
import time
import statistics
import subprocess
import concurrent.futures
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import traceback

# Add hooks modules to path
HOOKS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(HOOKS_DIR / "modules"))

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    test_name: str
    execution_time_ms: float
    memory_usage_mb: float
    stderr_length: int
    feedback_generated: bool
    target_100ms_met: bool
    target_50ms_met: bool
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class BenchmarkResult:
    """Benchmark result for a specific test scenario."""
    scenario_name: str
    iterations: int
    metrics: List[PerformanceMetrics]
    
    # Statistical analysis
    mean_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    
    # Performance targets
    target_100ms_achievement: float  # Percentage
    target_50ms_achievement: float   # Percentage
    
    # Resource efficiency
    mean_memory_mb: float
    peak_memory_mb: float
    mean_cpu_percent: float
    
    # Quality metrics
    feedback_generation_rate: float
    success_rate: float
    consistency_score: float  # 1.0 = perfectly consistent timing

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    performance_grade: str
    scenarios: List[BenchmarkResult]
    regression_detected: bool
    recommendations: List[str]


class PerformanceBenchmarker:
    """Advanced performance benchmarking system."""
    
    def __init__(self):
        self.hooks_dir = HOOKS_DIR
        self.hook_path = self.hooks_dir / "post_tool_use.py"
        self.baseline_data = self._load_baseline_data()
        self.results = []
        
    def _load_baseline_data(self) -> Dict[str, float]:
        """Load baseline performance data for regression detection."""
        baseline_file = self.hooks_dir / "tests" / "performance_baseline.json"
        try:
            if baseline_file.exists():
                with open(baseline_file) as f:
                    return json.load(f)
        except Exception:
            pass
        
        # Default baseline expectations
        return {
            "simple_read_ms": 25.0,
            "complex_write_ms": 50.0,
            "hook_violation_ms": 75.0,
            "error_handling_ms": 30.0,
            "mcp_tool_ms": 40.0
        }
    
    def _create_test_scenarios(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Create comprehensive test scenarios for benchmarking."""
        return [
            ("Simple Read", {
                "tool_name": "Read",
                "tool_input": {"file_path": "/tmp/test.py"},
                "tool_response": {"success": True, "content": "print('hello')"},
                "start_time": time.time()
            }),
            
            ("Complex Write Operation", {
                "tool_name": "Write", 
                "tool_input": {
                    "file_path": "/tmp/complex_file.py",
                    "content": "import os\nimport sys\nfrom pathlib import Path\n\ndef complex_function():\n    pass\n"
                },
                "tool_response": {"success": True},
                "start_time": time.time()
            }),
            
            ("Hook Violation Detection", {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": "/home/devcontainers/flowed/.claude/hooks/violation_test.py",
                    "content": "import sys\nsys.path.insert(0, '/dangerous/path')\nprint('violation')"
                },
                "tool_response": {"success": True},
                "start_time": time.time()
            }),
            
            ("MCP Tool Usage", {
                "tool_name": "mcp__zen__chat",
                "tool_input": {
                    "prompt": "Analyze this simple code pattern",
                    "model": "anthropic/claude-3.5-haiku"
                },
                "tool_response": {"success": True, "response": "Analysis complete"},
                "start_time": time.time()
            }),
            
            ("Error Handling", {
                "tool_name": "Read",
                "tool_input": {"file_path": "/nonexistent/file.py"},
                "tool_response": {"success": False, "error": "File not found"},
                "start_time": time.time()
            }),
            
            ("Large File Operation", {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": "/tmp/large_file.py",
                    "content": "# Large file simulation\n" + "# " + "x" * 1000 + "\n" * 100
                },
                "tool_response": {"success": True},
                "start_time": time.time()
            }),
            
            ("Multiple Tool Sequence", {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": "/tmp/sequence_test.py",
                    "old_string": "old_content",
                    "new_string": "new_content"
                },
                "tool_response": {"success": True},
                "start_time": time.time(),
                "_sequence_marker": True  # Indicates this is part of a sequence
            }),
            
            ("Timeout Simulation", {
                "tool_name": "WebSearch",
                "tool_input": {"query": "timeout test"},
                "tool_response": {"success": False, "error": "Request timeout"},
                "start_time": time.time()
            })
        ]
    
    @contextmanager
    def _monitor_resources(self):
        """Monitor CPU and memory usage during execution."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        process.cpu_percent()
        
        # Reset CPU measurement
        time.sleep(0.1)
        start_cpu = process.cpu_percent()
        
        measurements = {
            'memory_samples': [initial_memory],
            'cpu_samples': [start_cpu],
            'peak_memory': initial_memory
        }
        
        # Background monitoring
        monitoring = True
        def monitor():
            while monitoring:
                try:
                    mem = process.memory_info().rss / 1024 / 1024
                    cpu = process.cpu_percent()
                    measurements['memory_samples'].append(mem)
                    measurements['cpu_samples'].append(cpu)
                    measurements['peak_memory'] = max(measurements['peak_memory'], mem)
                    time.sleep(0.01)  # 10ms sampling
                except:
                    break
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        
        try:
            yield measurements
        finally:
            monitoring = False
            monitor_thread.join(timeout=0.1)
    
    def benchmark_single_execution(self, test_name: str, input_data: Dict[str, Any]) -> PerformanceMetrics:
        """Benchmark a single hook execution with detailed metrics."""
        
        with self._monitor_resources() as resources:
            start_time = time.perf_counter()
            
            try:
                process = subprocess.Popen(
                    [sys.executable, str(self.hook_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(self.hooks_dir)
                )
                
                stdout, stderr = process.communicate(
                    input=json.dumps(input_data),
                    timeout=5.0
                )
                
                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000
                
                # Analyze results
                success = process.returncode in [0, 2]
                feedback_generated = len(stderr.strip()) > 0
                stderr_length = len(stderr)
                
                return PerformanceMetrics(
                    test_name=test_name,
                    execution_time_ms=execution_time_ms,
                    memory_usage_mb=resources['peak_memory'],
                    stderr_length=stderr_length,
                    feedback_generated=feedback_generated,
                    target_100ms_met=execution_time_ms < 100,
                    target_50ms_met=execution_time_ms < 50,
                    cpu_usage_percent=statistics.mean(resources['cpu_samples']) if resources['cpu_samples'] else 0,
                    success=success
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                return PerformanceMetrics(
                    test_name=test_name,
                    execution_time_ms=5000,  # Timeout
                    memory_usage_mb=resources['peak_memory'],
                    stderr_length=0,
                    feedback_generated=False,
                    target_100ms_met=False,
                    target_50ms_met=False,
                    cpu_usage_percent=0,
                    success=False,
                    error_message="Execution timeout"
                )
                
            except Exception as e:
                return PerformanceMetrics(
                    test_name=test_name,
                    execution_time_ms=0,
                    memory_usage_mb=resources['peak_memory'],
                    stderr_length=0,
                    feedback_generated=False,
                    target_100ms_met=False,
                    target_50ms_met=False,
                    cpu_usage_percent=0,
                    success=False,
                    error_message=str(e)
                )
    
    def benchmark_scenario(self, scenario_name: str, input_data: Dict[str, Any], 
                         iterations: int = 20) -> BenchmarkResult:
        """Benchmark a specific scenario with statistical analysis."""
        
        print(f"  🔄 Benchmarking {scenario_name} ({iterations} iterations)...")
        
        metrics = []
        for i in range(iterations):
            metric = self.benchmark_single_execution(f"{scenario_name}_{i}", input_data)
            metrics.append(metric)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"    Progress: {i + 1}/{iterations} iterations complete")
        
        # Statistical analysis
        execution_times = [m.execution_time_ms for m in metrics if m.success]
        
        if not execution_times:
            # All executions failed
            return BenchmarkResult(
                scenario_name=scenario_name,
                iterations=iterations,
                metrics=metrics,
                mean_time_ms=float('inf'),
                median_time_ms=float('inf'),
                p95_time_ms=float('inf'),
                p99_time_ms=float('inf'),
                min_time_ms=float('inf'),
                max_time_ms=float('inf'),
                std_dev_ms=0,
                target_100ms_achievement=0,
                target_50ms_achievement=0,
                mean_memory_mb=0,
                peak_memory_mb=0,
                mean_cpu_percent=0,
                feedback_generation_rate=0,
                success_rate=0,
                consistency_score=0
            )
        
        # Calculate statistics
        mean_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Percentiles
        sorted_times = sorted(execution_times)
        p95_index = int(0.95 * len(sorted_times))
        p99_index = int(0.99 * len(sorted_times))
        p95_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
        p99_time = sorted_times[min(p99_index, len(sorted_times) - 1)]
        
        # Target achievements
        target_100ms_count = sum(1 for m in metrics if m.target_100ms_met and m.success)
        target_50ms_count = sum(1 for m in metrics if m.target_50ms_met and m.success)
        successful_count = sum(1 for m in metrics if m.success)
        
        target_100ms_achievement = target_100ms_count / successful_count if successful_count > 0 else 0
        target_50ms_achievement = target_50ms_count / successful_count if successful_count > 0 else 0
        
        # Resource usage
        successful_metrics = [m for m in metrics if m.success]
        mean_memory = statistics.mean(m.memory_usage_mb for m in successful_metrics) if successful_metrics else 0
        peak_memory = max(m.memory_usage_mb for m in successful_metrics) if successful_metrics else 0
        mean_cpu = statistics.mean(m.cpu_usage_percent for m in successful_metrics) if successful_metrics else 0
        
        # Quality metrics
        feedback_count = sum(1 for m in successful_metrics if m.feedback_generated)
        feedback_rate = feedback_count / len(successful_metrics) if successful_metrics else 0
        success_rate = len(successful_metrics) / len(metrics)
        
        # Consistency score (1.0 = perfectly consistent)
        consistency_score = 1.0 - (std_dev / mean_time) if mean_time > 0 else 0
        consistency_score = max(0, min(1.0, consistency_score))
        
        return BenchmarkResult(
            scenario_name=scenario_name,
            iterations=iterations,
            metrics=metrics,
            mean_time_ms=mean_time,
            median_time_ms=median_time,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            target_100ms_achievement=target_100ms_achievement,
            target_50ms_achievement=target_50ms_achievement,
            mean_memory_mb=mean_memory,
            peak_memory_mb=peak_memory,
            mean_cpu_percent=mean_cpu,
            feedback_generation_rate=feedback_rate,
            success_rate=success_rate,
            consistency_score=consistency_score
        )
    
    def benchmark_concurrent_load(self, scenario_name: str, input_data: Dict[str, Any],
                                concurrent_users: int = 5, requests_per_user: int = 10) -> Dict[str, Any]:
        """Benchmark concurrent load performance."""
        
        print(f"  🔄 Load testing {scenario_name} ({concurrent_users} concurrent users, {requests_per_user} requests each)...")
        
        def execute_user_requests():
            user_metrics = []
            for _ in range(requests_per_user):
                metric = self.benchmark_single_execution(scenario_name, input_data)
                user_metrics.append(metric)
            return user_metrics
        
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(execute_user_requests) for _ in range(concurrent_users)]
            all_results = []
            
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                except Exception as e:
                    print(f"    Warning: User request failed: {e}")
        
        total_time = time.perf_counter() - start_time
        
        # Analyze concurrent performance
        successful_requests = [r for r in all_results if r.success]
        
        if not successful_requests:
            return {
                "concurrent_users": concurrent_users,
                "total_requests": len(all_results),
                "successful_requests": 0,
                "success_rate": 0,
                "mean_response_time_ms": float('inf'),
                "throughput_rps": 0,
                "load_test_passed": False
            }
        
        mean_response_time = statistics.mean(r.execution_time_ms for r in successful_requests)
        throughput = len(successful_requests) / total_time
        success_rate = len(successful_requests) / len(all_results)
        
        # Load test passes if mean response time under load < 200ms and success rate > 80%
        load_test_passed = mean_response_time < 200 and success_rate > 0.8
        
        return {
            "concurrent_users": concurrent_users,
            "total_requests": len(all_results),
            "successful_requests": len(successful_requests),
            "success_rate": success_rate,
            "mean_response_time_ms": mean_response_time,
            "throughput_rps": throughput,
            "load_test_passed": load_test_passed
        }
    
    def detect_performance_regression(self, results: List[BenchmarkResult]) -> Tuple[bool, List[str]]:
        """Detect performance regressions against baseline."""
        
        regression_detected = False
        issues = []
        
        for result in results:
            scenario_key = result.scenario_name.lower().replace(" ", "_") + "_ms"
            baseline_time = self.baseline_data.get(scenario_key)
            
            if baseline_time and result.mean_time_ms > baseline_time * 1.5:  # 50% regression threshold
                regression_detected = True
                regression_percent = ((result.mean_time_ms - baseline_time) / baseline_time) * 100
                issues.append(f"{result.scenario_name}: {regression_percent:+.1f}% regression ({result.mean_time_ms:.1f}ms vs {baseline_time:.1f}ms baseline)")
        
        return regression_detected, issues
    
    def generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance optimization recommendations."""
        
        recommendations = []
        
        # Overall performance analysis
        mean_times = [r.mean_time_ms for r in results if r.success_rate > 0]
        if mean_times:
            overall_mean = statistics.mean(mean_times)
            
            if overall_mean > 100:
                recommendations.append("🚨 Critical: Average response time exceeds 100ms target - implement caching optimizations")
            elif overall_mean > 50:
                recommendations.append("⚠️  Warning: Average response time exceeds 50ms - consider async processing improvements")
            
            # Consistency analysis
            inconsistent_scenarios = [r for r in results if r.consistency_score < 0.7]
            if inconsistent_scenarios:
                recommendations.append(f"📊 Improve timing consistency for {len(inconsistent_scenarios)} scenarios - reduce variance")
        
        # Memory usage analysis
        high_memory_scenarios = [r for r in results if r.peak_memory_mb > 100]
        if high_memory_scenarios:
            recommendations.append(f"💾 Optimize memory usage for {len(high_memory_scenarios)} scenarios - current peak: {max(r.peak_memory_mb for r in high_memory_scenarios):.1f}MB")
        
        # Success rate analysis
        unreliable_scenarios = [r for r in results if r.success_rate < 0.95]
        if unreliable_scenarios:
            recommendations.append(f"🔧 Improve reliability for {len(unreliable_scenarios)} scenarios - implement better error handling")
        
        # Feedback generation analysis
        low_feedback_scenarios = [r for r in results if r.feedback_generation_rate < 0.5]
        if low_feedback_scenarios:
            recommendations.append(f"💬 Enhance feedback generation for {len(low_feedback_scenarios)} scenarios - improve contextual analysis")
        
        if not recommendations:
            recommendations.append("✅ Excellent performance across all metrics - system is optimally tuned")
        
        return recommendations
    
    def calculate_performance_grade(self, results: List[BenchmarkResult]) -> Tuple[float, str]:
        """Calculate overall performance grade."""
        
        if not results:
            return 0.0, "F"
        
        # Weighted scoring
        scores = []
        
        for result in results:
            if result.success_rate == 0:
                scores.append(0)
                continue
                
            # Time performance (40% weight)
            if result.mean_time_ms < 25:
                time_score = 100
            elif result.mean_time_ms < 50:
                time_score = 90
            elif result.mean_time_ms < 100:
                time_score = 80
            elif result.mean_time_ms < 200:
                time_score = 70
            else:
                time_score = max(0, 100 - (result.mean_time_ms - 200) / 10)
            
            # Success rate (30% weight)
            success_score = result.success_rate * 100
            
            # Consistency (20% weight)
            consistency_score = result.consistency_score * 100
            
            # Feedback quality (10% weight)
            feedback_score = result.feedback_generation_rate * 100
            
            scenario_score = (
                time_score * 0.4 +
                success_score * 0.3 +
                consistency_score * 0.2 +
                feedback_score * 0.1
            )
            
            scores.append(scenario_score)
        
        overall_score = statistics.mean(scores)
        
        # Grade assignment
        if overall_score >= 95:
            grade = "A+"
        elif overall_score >= 90:
            grade = "A"
        elif overall_score >= 85:
            grade = "A-"
        elif overall_score >= 80:
            grade = "B+"
        elif overall_score >= 75:
            grade = "B"
        elif overall_score >= 70:
            grade = "B-"
        elif overall_score >= 65:
            grade = "C+"
        elif overall_score >= 60:
            grade = "C"
        elif overall_score >= 55:
            grade = "C-"
        else:
            grade = "D" if overall_score >= 50 else "F"
        
        return overall_score, grade
    
    def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """Run comprehensive performance benchmark suite."""
        
        print("🚀 INTELLIGENT FEEDBACK SYSTEM - PERFORMANCE BENCHMARK SUITE")
        print("=" * 80)
        print("Target: <100ms stderr feedback generation")
        print(f"Hook: {self.hook_path}")
        print("=" * 80)
        
        scenarios = self._create_test_scenarios()
        benchmark_results = []
        
        # Phase 1: Individual scenario benchmarks
        print("\n📊 Phase 1: Individual Scenario Benchmarks")
        print("-" * 50)
        
        for scenario_name, input_data in scenarios:
            try:
                result = self.benchmark_scenario(scenario_name, input_data, iterations=15)
                benchmark_results.append(result)
                
                # Progress report
                status = "✅ PASSED" if result.target_100ms_achievement > 0.8 else "⚠️  SLOW" if result.target_100ms_achievement > 0.5 else "❌ FAILED"
                print(f"    {scenario_name}: {result.mean_time_ms:.1f}ms avg ({result.target_100ms_achievement:.0%} <100ms) {status}")
                
            except Exception as e:
                print(f"    {scenario_name}: ❌ FAILED - {e}")
                # Create failed result
                failed_result = BenchmarkResult(
                    scenario_name=scenario_name,
                    iterations=0,
                    metrics=[],
                    mean_time_ms=float('inf'),
                    median_time_ms=float('inf'),
                    p95_time_ms=float('inf'),
                    p99_time_ms=float('inf'),
                    min_time_ms=float('inf'),
                    max_time_ms=float('inf'),
                    std_dev_ms=0,
                    target_100ms_achievement=0,
                    target_50ms_achievement=0,
                    mean_memory_mb=0,
                    peak_memory_mb=0,
                    mean_cpu_percent=0,
                    feedback_generation_rate=0,
                    success_rate=0,
                    consistency_score=0
                )
                benchmark_results.append(failed_result)
        
        # Phase 2: Load testing
        print("\n🔄 Phase 2: Concurrent Load Testing")
        print("-" * 50)
        
        # Test a representative scenario under load
        representative_scenario = scenarios[0]  # Simple Read
        load_result = self.benchmark_concurrent_load(
            representative_scenario[0], 
            representative_scenario[1],
            concurrent_users=3,
            requests_per_user=8
        )
        
        print(f"    Load Test: {load_result['mean_response_time_ms']:.1f}ms avg, {load_result['throughput_rps']:.1f} RPS")
        print(f"    Result: {'✅ PASSED' if load_result['load_test_passed'] else '❌ FAILED'}")
        
        # Phase 3: Analysis and grading
        print("\n📈 Phase 3: Performance Analysis")
        print("-" * 50)
        
        successful_results = [r for r in benchmark_results if r.success_rate > 0]
        failed_results = [r for r in benchmark_results if r.success_rate == 0]
        
        # Regression detection
        regression_detected, regression_issues = self.detect_performance_regression(successful_results)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(successful_results)
        
        # Calculate grade
        overall_score, grade = self.calculate_performance_grade(benchmark_results)
        
        # Create final result
        suite_result = BenchmarkSuite(
            total_tests=len(benchmark_results),
            passed_tests=len(successful_results),
            failed_tests=len(failed_results),
            overall_score=overall_score,
            performance_grade=grade,
            scenarios=benchmark_results,
            regression_detected=regression_detected,
            recommendations=recommendations
        )
        
        # Save detailed results
        self._save_benchmark_results(suite_result)
        
        # Print final report
        self._print_benchmark_report(suite_result, load_result, regression_issues)
        
        return suite_result
    
    def _save_benchmark_results(self, suite: BenchmarkSuite) -> None:
        """Save benchmark results to file."""
        
        results_file = self.hooks_dir / "tests" / "performance_benchmark_results.json"
        
        # Convert to serializable format
        results_data = {
            "timestamp": time.time(),
            "suite_summary": asdict(suite),
            "detailed_metrics": [
                {
                    "scenario": result.scenario_name,
                    "metrics": [asdict(m) for m in result.metrics]
                }
                for result in suite.scenarios
            ]
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\n💾 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")
    
    def _print_benchmark_report(self, suite: BenchmarkSuite, load_result: Dict[str, Any], 
                               regression_issues: List[str]) -> None:
        """Print comprehensive benchmark report."""
        
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)
        
        print("\n🎯 OVERALL PERFORMANCE")
        print("-" * 25)
        print(f"Score: {suite.overall_score:.1f}/100 ({suite.performance_grade})")
        print(f"Tests: {suite.passed_tests}/{suite.total_tests} passed")
        print("Target Achievement: <100ms feedback generation")
        
        print("\n📈 SCENARIO BREAKDOWN")
        print("-" * 25)
        
        for result in suite.scenarios:
            if result.success_rate > 0:
                target_status = "✅" if result.target_100ms_achievement > 0.8 else "⚠️ " if result.target_100ms_achievement > 0.5 else "❌"
                print(f"• {result.scenario_name}: {result.mean_time_ms:.1f}ms avg (P95: {result.p95_time_ms:.1f}ms) {target_status}")
                print(f"  Success: {result.success_rate:.1%}, Feedback: {result.feedback_generation_rate:.1%}, Consistency: {result.consistency_score:.2f}")
            else:
                print(f"• {result.scenario_name}: ❌ FAILED")
        
        print("\n🔄 LOAD TEST RESULTS")
        print("-" * 25)
        print(f"Concurrent Performance: {load_result['mean_response_time_ms']:.1f}ms avg")
        print(f"Throughput: {load_result['throughput_rps']:.1f} requests/second")
        print(f"Load Test: {'✅ PASSED' if load_result['load_test_passed'] else '❌ FAILED'}")
        
        if regression_issues:
            print("\n🚨 REGRESSION DETECTED")
            print("-" * 25)
            for issue in regression_issues:
                print(f"• {issue}")
        
        print("\n💡 RECOMMENDATIONS")
        print("-" * 25)
        for rec in suite.recommendations:
            print(f"• {rec}")
        
        # Performance summary
        successful_scenarios = [r for r in suite.scenarios if r.success_rate > 0]
        if successful_scenarios:
            fastest_time = min(r.mean_time_ms for r in successful_scenarios)
            slowest_time = max(r.mean_time_ms for r in successful_scenarios)
            avg_time = statistics.mean(r.mean_time_ms for r in successful_scenarios)
            
            target_100ms_overall = statistics.mean(r.target_100ms_achievement for r in successful_scenarios)
            target_50ms_overall = statistics.mean(r.target_50ms_achievement for r in successful_scenarios)
            
            print("\n🏆 PERFORMANCE HIGHLIGHTS")
            print("-" * 30)
            print(f"Fastest Response: {fastest_time:.1f}ms")
            print(f"Slowest Response: {slowest_time:.1f}ms") 
            print(f"Average Response: {avg_time:.1f}ms")
            print(f"Sub-100ms Achievement: {target_100ms_overall:.1%}")
            print(f"Sub-50ms Achievement: {target_50ms_overall:.1%}")
            
            print("\n🎖️  FINAL ASSESSMENT")
            print("-" * 25)
            if suite.performance_grade in ["A+", "A", "A-"]:
                print("✅ EXCELLENT - Feedback system meets all performance targets")
            elif suite.performance_grade in ["B+", "B", "B-"]:
                print("⚠️  GOOD - Minor optimizations recommended")
            elif suite.performance_grade in ["C+", "C", "C-"]:
                print("🔧 NEEDS IMPROVEMENT - Performance targets not consistently met")
            else:
                print("❌ POOR - Significant performance issues require attention")
        
        print("=" * 80)


def run_performance_benchmark_suite() -> BenchmarkSuite:
    """Run the complete performance benchmark suite."""
    benchmarker = PerformanceBenchmarker()
    return benchmarker.run_comprehensive_benchmark()


if __name__ == "__main__":
    run_performance_benchmark_suite()