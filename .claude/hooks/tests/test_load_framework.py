#!/usr/bin/env python3
"""Load Testing Framework for ZEN Co-pilot Multi-Project Orchestration.

This module provides comprehensive load testing for:
- Multi-project orchestration scalability
- Concurrent ZenConsultant operations
- Memory system performance under load
- Hook system scalability validation
- Resource utilization monitoring
- Throughput and latency benchmarking
"""

import time
import json
import threading
import psutil
import statistics
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
import queue
import sys

# Set up hook paths
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

from modules.core.zen_consultant import ZenConsultant, ComplexityLevel


@dataclass
class LoadTestScenario:
    """Load test scenario configuration."""
    name: str
    description: str
    concurrent_projects: int
    operations_per_project: int
    operation_types: List[str]
    duration_seconds: int
    ramp_up_seconds: int
    expected_throughput: float
    max_response_time_ms: float
    max_memory_usage_percent: float


@dataclass
class LoadTestMetrics:
    """Load test metrics container."""
    timestamp: float
    operations_completed: int
    operations_failed: int
    response_times_ms: List[float]
    memory_usage_mb: float
    cpu_usage_percent: float
    active_threads: int
    throughput_ops_per_sec: float


@dataclass
class LoadTestResult:
    """Complete load test result."""
    scenario: str
    success: bool
    duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    peak_memory_mb: float
    peak_cpu_percent: float
    avg_throughput_ops_per_sec: float
    peak_throughput_ops_per_sec: float
    bottlenecks: List[str]
    recommendations: List[str]


class MultiProjectLoadTester:
    """Comprehensive load testing suite for multi-project orchestration."""
    
    def __init__(self):
        self.zen_consultant = ZenConsultant()
        self.metrics_queue = queue.Queue()
        self.metrics_collection_active = False
        self.load_test_results: List[LoadTestResult] = []
        self._initialize_test_scenarios()
        
    def _initialize_test_scenarios(self) -> None:
        """Initialize load test scenarios."""
        self.test_scenarios = [
            LoadTestScenario(
                name="light_load",
                description="Light load with 5 concurrent projects",
                concurrent_projects=5,
                operations_per_project=20,
                operation_types=["simple_directive", "medium_directive"],
                duration_seconds=60,
                ramp_up_seconds=10,
                expected_throughput=10.0,
                max_response_time_ms=100.0,
                max_memory_usage_percent=30.0
            ),
            
            LoadTestScenario(
                name="moderate_load",
                description="Moderate load with 15 concurrent projects",
                concurrent_projects=15,
                operations_per_project=50,
                operation_types=["simple_directive", "medium_directive", "complex_directive"],
                duration_seconds=120,
                ramp_up_seconds=20,
                expected_throughput=25.0,
                max_response_time_ms=200.0,
                max_memory_usage_percent=40.0
            ),
            
            LoadTestScenario(
                name="heavy_load",
                description="Heavy load with 30 concurrent projects",
                concurrent_projects=30,
                operations_per_project=100,
                operation_types=["simple_directive", "medium_directive", "complex_directive", "enterprise_directive"],
                duration_seconds=300,
                ramp_up_seconds=30,
                expected_throughput=40.0,
                max_response_time_ms=500.0,
                max_memory_usage_percent=50.0
            ),
            
            LoadTestScenario(
                name="stress_test",
                description="Stress test with 50 concurrent projects",
                concurrent_projects=50,
                operations_per_project=200,
                operation_types=["simple_directive", "medium_directive", "complex_directive", "enterprise_directive"],
                duration_seconds=600,
                ramp_up_seconds=60,
                expected_throughput=50.0,
                max_response_time_ms=1000.0,
                max_memory_usage_percent=60.0
            ),
            
            LoadTestScenario(
                name="burst_load",
                description="Burst load testing with rapid ramp-up",
                concurrent_projects=25,
                operations_per_project=50,
                operation_types=["simple_directive", "medium_directive", "complex_directive"],
                duration_seconds=60,
                ramp_up_seconds=5,  # Rapid ramp-up
                expected_throughput=30.0,
                max_response_time_ms=300.0,
                max_memory_usage_percent=45.0
            ),
            
            LoadTestScenario(
                name="endurance_test",
                description="Endurance test with sustained load",
                concurrent_projects=20,
                operations_per_project=1000,
                operation_types=["simple_directive", "medium_directive", "complex_directive"],
                duration_seconds=1800,  # 30 minutes
                ramp_up_seconds=60,
                expected_throughput=20.0,
                max_response_time_ms=250.0,
                max_memory_usage_percent=35.0
            )
        ]
        
    def _generate_test_prompt(self, operation_type: str) -> str:
        """Generate test prompt based on operation type."""
        prompts = {
            "simple_directive": [
                "Fix CSS styling bug in login form",
                "Add logging to user registration",
                "Update configuration parameter",
                "Create unit test for helper function",
                "Optimize database query performance"
            ],
            "medium_directive": [
                "Refactor authentication module for better security",
                "Implement REST API for user management",
                "Design database schema for product catalog", 
                "Create monitoring dashboard for system metrics",
                "Implement caching layer for improved performance"
            ],
            "complex_directive": [
                "Design microservices architecture with service mesh",
                "Implement machine learning recommendation engine",
                "Build real-time analytics pipeline with stream processing",
                "Create multi-tenant SaaS platform architecture",
                "Implement distributed tracing and observability"
            ],
            "enterprise_directive": [
                "Design enterprise-scale platform with compliance and audit trails",
                "Implement multi-cloud deployment with disaster recovery",
                "Build comprehensive security framework with zero-trust architecture",
                "Create enterprise data platform with governance and lineage",
                "Implement global-scale distributed system with consensus protocols"
            ]
        }
        
        return random.choice(prompts.get(operation_type, prompts["simple_directive"]))
        
    def _project_worker(self, project_id: int, scenario: LoadTestScenario, 
                       operations_queue: queue.Queue, results_queue: queue.Queue) -> None:
        """Worker function for simulating a single project's operations."""
        project_consultant = ZenConsultant()
        project_operations = 0
        project_failures = 0
        
        try:
            while project_operations < scenario.operations_per_project:
                try:
                    # Get operation from queue
                    operation_type = operations_queue.get(timeout=1.0)
                    
                    # Generate test prompt
                    prompt = self._generate_test_prompt(operation_type)
                    
                    # Execute operation with timing
                    start_time = time.time()
                    project_consultant.get_concise_directive(prompt)
                    end_time = time.time()
                    
                    response_time_ms = (end_time - start_time) * 1000
                    
                    # Record metrics
                    metrics = LoadTestMetrics(
                        timestamp=time.time(),
                        operations_completed=1,
                        operations_failed=0,
                        response_times_ms=[response_time_ms],
                        memory_usage_mb=psutil.Process().memory_info().rss / (1024 * 1024),
                        cpu_usage_percent=psutil.cpu_percent(),
                        active_threads=threading.active_count(),
                        throughput_ops_per_sec=1.0 / (end_time - start_time)
                    )
                    
                    results_queue.put(metrics)
                    project_operations += 1
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.01)
                    
                except queue.Empty:
                    break
                except Exception:
                    project_failures += 1
                    
                    # Record failure metrics
                    failure_metrics = LoadTestMetrics(
                        timestamp=time.time(),
                        operations_completed=0,
                        operations_failed=1,
                        response_times_ms=[],
                        memory_usage_mb=psutil.Process().memory_info().rss / (1024 * 1024),
                        cpu_usage_percent=psutil.cpu_percent(),
                        active_threads=threading.active_count(),
                        throughput_ops_per_sec=0.0
                    )
                    
                    results_queue.put(failure_metrics)
                    
        except Exception as e:
            print(f"Project {project_id} worker error: {e}")
            
    def _populate_operations_queue(self, operations_queue: queue.Queue, scenario: LoadTestScenario) -> None:
        """Populate operations queue with work items."""
        total_operations = scenario.concurrent_projects * scenario.operations_per_project
        
        for _ in range(total_operations):
            operation_type = random.choice(scenario.operation_types)
            operations_queue.put(operation_type)
            
    def _collect_metrics(self, results_queue: queue.Queue, collected_metrics: List[LoadTestMetrics]) -> None:
        """Collect metrics from worker threads."""
        while self.metrics_collection_active:
            try:
                metrics = results_queue.get(timeout=0.1)
                collected_metrics.append(metrics)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Metrics collection error: {e}")
                
    def run_load_test_scenario(self, scenario: LoadTestScenario) -> LoadTestResult:
        """Run a single load test scenario."""
        print(f"🚀 Running Load Test Scenario: {scenario.name}")
        print(f"   Projects: {scenario.concurrent_projects}, Operations: {scenario.operations_per_project}")
        
        # Initialize queues and metrics collection
        operations_queue = queue.Queue()
        results_queue = queue.Queue()
        collected_metrics: List[LoadTestMetrics] = []
        
        # Populate operations queue
        self._populate_operations_queue(operations_queue, scenario)
        
        # Start metrics collection
        self.metrics_collection_active = True
        metrics_thread = threading.Thread(
            target=self._collect_metrics,
            args=(results_queue, collected_metrics)
        )
        metrics_thread.start()
        
        # Record test start
        test_start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Create and start project workers with ramp-up
        project_threads = []
        ramp_up_delay = scenario.ramp_up_seconds / scenario.concurrent_projects
        
        for project_id in range(scenario.concurrent_projects):
            thread = threading.Thread(
                target=self._project_worker,
                args=(project_id, scenario, operations_queue, results_queue)
            )
            thread.start()
            project_threads.append(thread)
            
            # Ramp-up delay
            if ramp_up_delay > 0:
                time.sleep(ramp_up_delay)
                
        # Monitor test execution
        time.time()
        peak_memory = initial_memory
        peak_cpu = 0.0
        
        while time.time() - test_start_time < scenario.duration_seconds:
            # Monitor system resources
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            current_cpu = psutil.cpu_percent()
            
            peak_memory = max(peak_memory, current_memory)
            peak_cpu = max(peak_cpu, current_cpu)
            
            # Check if all operations are complete
            if operations_queue.empty() and all(not t.is_alive() for t in project_threads):
                break
                
            time.sleep(1.0)
            
        # Wait for remaining threads to complete
        remaining_timeout = max(0, scenario.duration_seconds - (time.time() - test_start_time))
        for thread in project_threads:
            thread.join(timeout=remaining_timeout)
            
        # Stop metrics collection
        self.metrics_collection_active = False
        metrics_thread.join(timeout=5.0)
        
        test_end_time = time.time()
        actual_duration = test_end_time - test_start_time
        
        # Analyze collected metrics
        return self._analyze_load_test_results(scenario, collected_metrics, actual_duration, peak_memory, peak_cpu)
        
    def _analyze_load_test_results(self, scenario: LoadTestScenario, metrics: List[LoadTestMetrics],
                                 duration: float, peak_memory: float, peak_cpu: float) -> LoadTestResult:
        """Analyze load test results and generate comprehensive report."""
        
        if not metrics:
            return LoadTestResult(
                scenario=scenario.name,
                success=False,
                duration_seconds=duration,
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                avg_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                max_response_time_ms=0.0,
                peak_memory_mb=peak_memory,
                peak_cpu_percent=peak_cpu,
                avg_throughput_ops_per_sec=0.0,
                peak_throughput_ops_per_sec=0.0,
                bottlenecks=["No metrics collected"],
                recommendations=["Investigate metrics collection failure"]
            )
            
        # Aggregate metrics
        total_operations = sum(m.operations_completed for m in metrics)
        total_failures = sum(m.operations_failed for m in metrics)
        successful_operations = total_operations - total_failures
        
        # Response time analysis
        all_response_times = []
        for m in metrics:
            all_response_times.extend(m.response_times_ms)
            
        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            max_response_time = max(all_response_times)
            
            # Calculate percentiles
            sorted_times = sorted(all_response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_response_time
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_response_time
        else:
            avg_response_time = 0.0
            max_response_time = 0.0
            p95_response_time = 0.0
            p99_response_time = 0.0
            
        # Throughput analysis
        if duration > 0:
            avg_throughput = total_operations / duration
            
            # Calculate peak throughput (highest throughput in any 1-second window)
            throughput_samples = [m.throughput_ops_per_sec for m in metrics if m.throughput_ops_per_sec > 0]
            peak_throughput = max(throughput_samples) if throughput_samples else 0.0
        else:
            avg_throughput = 0.0
            peak_throughput = 0.0
            
        # Determine success criteria
        success_criteria = [
            avg_throughput >= scenario.expected_throughput * 0.8,  # 80% of expected throughput
            p95_response_time <= scenario.max_response_time_ms,
            peak_memory <= (psutil.virtual_memory().total / (1024 * 1024)) * (scenario.max_memory_usage_percent / 100),
            total_failures / max(total_operations, 1) <= 0.05  # Less than 5% failure rate
        ]
        
        overall_success = all(success_criteria)
        
        # Identify bottlenecks
        bottlenecks = []
        if avg_throughput < scenario.expected_throughput * 0.8:
            bottlenecks.append(f"Throughput below target: {avg_throughput:.1f} < {scenario.expected_throughput:.1f}")
        if p95_response_time > scenario.max_response_time_ms:
            bottlenecks.append(f"Response time exceeds limit: P95 {p95_response_time:.1f}ms > {scenario.max_response_time_ms}ms")
        if peak_cpu > 90:
            bottlenecks.append(f"High CPU utilization: {peak_cpu:.1f}%")
        if peak_memory > (psutil.virtual_memory().total / (1024 * 1024)) * 0.8:
            bottlenecks.append(f"High memory utilization: {peak_memory:.1f}MB")
        if total_failures > 0:
            bottlenecks.append(f"Operation failures detected: {total_failures}/{total_operations}")
            
        # Generate recommendations
        recommendations = self._generate_load_test_recommendations(scenario, bottlenecks, metrics)
        
        return LoadTestResult(
            scenario=scenario.name,
            success=overall_success,
            duration_seconds=duration,
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=total_failures,
            avg_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            max_response_time_ms=max_response_time,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            avg_throughput_ops_per_sec=avg_throughput,
            peak_throughput_ops_per_sec=peak_throughput,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )
        
    def _generate_load_test_recommendations(self, scenario: LoadTestScenario, 
                                          bottlenecks: List[str], metrics: List[LoadTestMetrics]) -> List[str]:
        """Generate recommendations based on load test results."""
        recommendations = []
        
        if any("Throughput below target" in b for b in bottlenecks):
            recommendations.append("Consider horizontal scaling or performance optimization")
            recommendations.append("Review algorithm complexity and optimize critical paths")
            
        if any("Response time exceeds" in b for b in bottlenecks):
            recommendations.append("Implement response time optimization techniques")
            recommendations.append("Consider asynchronous processing for heavy operations")
            
        if any("High CPU utilization" in b for b in bottlenecks):
            recommendations.append("Optimize CPU-intensive operations")
            recommendations.append("Consider CPU scaling or load balancing")
            
        if any("High memory utilization" in b for b in bottlenecks):
            recommendations.append("Implement memory optimization and garbage collection tuning")
            recommendations.append("Consider memory scaling or caching strategies")
            
        if any("Operation failures" in b for b in bottlenecks):
            recommendations.append("Improve error handling and retry mechanisms")
            recommendations.append("Investigate root causes of operation failures")
            
        if not bottlenecks:
            recommendations.append("Load test passed successfully - system performing within acceptable limits")
            recommendations.append("Consider testing higher load levels to find system limits")
            
        return recommendations
        
    def run_comprehensive_load_tests(self, selected_scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive load tests across multiple scenarios."""
        print("⚡ Running Comprehensive Load Testing Suite")
        print("=" * 50)
        
        # Filter scenarios if specified
        if selected_scenarios:
            scenarios_to_run = [s for s in self.test_scenarios if s.name in selected_scenarios]
        else:
            scenarios_to_run = self.test_scenarios
            
        # Run each scenario
        for scenario in scenarios_to_run:
            result = self.run_load_test_scenario(scenario)
            self.load_test_results.append(result)
            
            # Print scenario results
            if result.success:
                print(f"  ✅ {scenario.name}: PASSED")
            else:
                print(f"  ❌ {scenario.name}: FAILED")
                
            print(f"     Throughput: {result.avg_throughput_ops_per_sec:.1f} ops/sec")
            print(f"     P95 Response: {result.p95_response_time_ms:.1f}ms")
            print(f"     Peak Memory: {result.peak_memory_mb:.1f}MB")
            
        # Generate comprehensive analysis
        return self._generate_comprehensive_load_report()
        
    def _generate_comprehensive_load_report(self) -> Dict[str, Any]:
        """Generate comprehensive load testing report."""
        if not self.load_test_results:
            return {"error": "No load test results available"}
            
        # Overall statistics
        total_scenarios = len(self.load_test_results)
        passed_scenarios = sum(1 for r in self.load_test_results if r.success)
        
        # Performance metrics aggregation
        avg_throughputs = [r.avg_throughput_ops_per_sec for r in self.load_test_results]
        peak_throughputs = [r.peak_throughput_ops_per_sec for r in self.load_test_results]
        p95_response_times = [r.p95_response_time_ms for r in self.load_test_results]
        peak_memories = [r.peak_memory_mb for r in self.load_test_results]
        
        # Find performance limits
        max_concurrent_projects = max(
            next(s.concurrent_projects for s in self.test_scenarios if s.name == r.scenario)
            for r in self.load_test_results if r.success
        ) if any(r.success for r in self.load_test_results) else 0
        
        max_throughput = max(peak_throughputs) if peak_throughputs else 0
        
        # Collect all bottlenecks and recommendations
        all_bottlenecks = []
        all_recommendations = []
        
        for result in self.load_test_results:
            all_bottlenecks.extend(result.bottlenecks)
            all_recommendations.extend(result.recommendations)
            
        unique_bottlenecks = list(set(all_bottlenecks))
        unique_recommendations = list(set(all_recommendations))
        
        # Determine overall system status
        if passed_scenarios == total_scenarios:
            system_status = "EXCELLENT - All load tests passed"
        elif passed_scenarios >= total_scenarios * 0.8:
            system_status = "GOOD - Most load tests passed"
        elif passed_scenarios >= total_scenarios * 0.5:
            system_status = "FAIR - Some load tests failed"
        else:
            system_status = "POOR - Most load tests failed"
            
        return {
            "timestamp": time.time(),
            "system_status": system_status,
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "pass_rate": passed_scenarios / total_scenarios,
            "performance_summary": {
                "max_concurrent_projects": max_concurrent_projects,
                "max_throughput_ops_per_sec": max_throughput,
                "avg_throughput_range": f"{min(avg_throughputs):.1f}-{max(avg_throughputs):.1f}" if avg_throughputs else "N/A",
                "p95_response_time_range": f"{min(p95_response_times):.1f}-{max(p95_response_times):.1f}ms" if p95_response_times else "N/A",
                "peak_memory_range": f"{min(peak_memories):.1f}-{max(peak_memories):.1f}MB" if peak_memories else "N/A"
            },
            "scenario_results": [
                {
                    "scenario": r.scenario,
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "total_operations": r.total_operations,
                    "avg_throughput_ops_per_sec": r.avg_throughput_ops_per_sec,
                    "p95_response_time_ms": r.p95_response_time_ms,
                    "peak_memory_mb": r.peak_memory_mb,
                    "bottlenecks": r.bottlenecks
                } for r in self.load_test_results
            ],
            "bottlenecks": unique_bottlenecks,
            "recommendations": unique_recommendations,
            "scalability_assessment": self._assess_scalability()
        }
        
    def _assess_scalability(self) -> Dict[str, Any]:
        """Assess system scalability based on load test results."""
        if not self.load_test_results:
            return {"assessment": "No data available"}
            
        # Analyze throughput scaling
        throughput_data = []
        for result in self.load_test_results:
            scenario = next(s for s in self.test_scenarios if s.name == result.scenario)
            throughput_data.append({
                "concurrent_projects": scenario.concurrent_projects,
                "throughput": result.avg_throughput_ops_per_sec,
                "success": result.success
            })
            
        # Sort by concurrent projects
        throughput_data.sort(key=lambda x: x["concurrent_projects"])
        
        # Determine scaling characteristics
        if len(throughput_data) >= 2:
            # Check if throughput increases with load (good scaling)
            throughput_trend = []
            for i in range(1, len(throughput_data)):
                prev = throughput_data[i-1]
                curr = throughput_data[i]
                if prev["success"] and curr["success"]:
                    ratio = curr["throughput"] / prev["throughput"] if prev["throughput"] > 0 else 0
                    throughput_trend.append(ratio)
                    
            if throughput_trend:
                avg_scaling_ratio = statistics.mean(throughput_trend)
                if avg_scaling_ratio >= 1.5:
                    scaling_assessment = "EXCELLENT - Super-linear scaling observed"
                elif avg_scaling_ratio >= 1.2:
                    scaling_assessment = "GOOD - Above-linear scaling"
                elif avg_scaling_ratio >= 0.8:
                    scaling_assessment = "FAIR - Near-linear scaling"
                else:
                    scaling_assessment = "POOR - Sub-linear scaling"
            else:
                scaling_assessment = "UNKNOWN - Insufficient data"
        else:
            scaling_assessment = "UNKNOWN - Need multiple load levels"
            
        # Find breaking point
        max_successful_load = 0
        for data in throughput_data:
            if data["success"]:
                max_successful_load = max(max_successful_load, data["concurrent_projects"])
                
        return {
            "assessment": scaling_assessment,
            "max_concurrent_projects": max_successful_load,
            "throughput_data": throughput_data,
            "scaling_efficiency": statistics.mean(throughput_trend) if throughput_trend else 0.0
        }


def run_load_test_suite(scenarios: Optional[List[str]] = None):
    """Run complete load test suite and save results."""
    print("⚡ ZEN Co-pilot System - Load Testing Framework")
    print("=" * 60)
    
    tester = MultiProjectLoadTester()
    
    # Run comprehensive load tests
    if scenarios:
        print(f"Running selected scenarios: {', '.join(scenarios)}")
    else:
        print("Running all load test scenarios...")
        
    report = tester.run_comprehensive_load_tests(scenarios)
    
    # Save report
    report_path = Path("/home/devcontainers/flowed/.claude/hooks/load_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    # Print summary
    print("\n⚡ LOAD TEST RESULTS SUMMARY")
    print("-" * 40)
    print(f"🎯 System Status: {report['system_status']}")
    print(f"✅ Scenarios Passed: {report['passed_scenarios']}/{report['total_scenarios']} ({report['pass_rate']:.1%})")
    
    perf = report['performance_summary']
    print(f"🚀 Max Concurrent Projects: {perf['max_concurrent_projects']}")
    print(f"⚡ Max Throughput: {perf['max_throughput_ops_per_sec']:.1f} ops/sec")
    print(f"📊 Response Time Range: {perf['p95_response_time_range']}")
    print(f"🧠 Memory Usage Range: {perf['peak_memory_range']}")
    
    # Scalability assessment
    scalability = report['scalability_assessment']
    print("\n📈 SCALABILITY ASSESSMENT")
    print("-" * 30)
    print(f"• Assessment: {scalability['assessment']}")
    print(f"• Max Load Handled: {scalability['max_concurrent_projects']} projects")
    print(f"• Scaling Efficiency: {scalability['scaling_efficiency']:.2f}")
    
    print(f"\n📋 Full report saved to: {report_path}")
    
    # Print top recommendations
    if report['recommendations']:
        print("\n🎯 TOP RECOMMENDATIONS")
        print("-" * 25)
        for rec in report['recommendations'][:5]:
            print(f"• {rec}")
            
    return report


if __name__ == "__main__":
    # Run with selected scenarios for testing
    # Uncomment to run specific scenarios:
    # run_load_test_suite(["light_load", "moderate_load"])
    
    # Run all scenarios
    run_load_test_suite()