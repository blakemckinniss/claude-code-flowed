"""
Comprehensive Test Framework Architecture for Hook System
=========================================================

This module provides the foundational testing architecture for the expanded
stderr feedback system with automated performance validation.

Test Layers:
1. Unit Tests - Individual analyzer components with mock data
2. Integration Tests - End-to-end hook pipeline testing 
3. Performance Benchmarks - stderr generation time and memory analysis
4. Validation Framework - Progressive rollout with success metrics
"""

import asyncio
import json
import time
import tracemalloc
import unittest
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union
from unittest.mock import Mock, MagicMock, patch
import logging
import os
import sys
from pathlib import Path

# Test framework configuration
TEST_CONFIG = {
    "performance": {
        "max_stderr_generation_time_ms": 50,  # 50ms max for stderr generation
        "max_memory_usage_mb": 10,           # 10MB max memory per test
        "benchmark_iterations": 100,          # Iterations for performance tests
        "timeout_seconds": 30                 # Test timeout
    },
    "validation": {
        "success_threshold": 0.95,           # 95% success rate required
        "error_tolerance": 0.02,             # 2% error tolerance
        "rollout_stages": ["dev", "staging", "production"]
    },
    "coverage": {
        "min_unit_coverage": 90,             # 90% minimum unit test coverage
        "min_integration_coverage": 80,      # 80% minimum integration coverage
        "critical_path_coverage": 100        # 100% coverage for critical paths
    }
}


@dataclass
class TestResult:
    """Standardized test result structure."""
    test_name: str
    status: str  # "passed", "failed", "skipped", "error"
    execution_time_ms: float
    memory_usage_mb: float
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    operation_name: str
    avg_execution_time_ms: float
    max_execution_time_ms: float
    min_execution_time_ms: float
    memory_peak_mb: float
    memory_avg_mb: float
    operations_per_second: float
    success_rate: float


class TestMetricsCollector:
    """Collects and analyzes test metrics."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.benchmarks: List[PerformanceBenchmark] = []
        self.start_time = time.time()
    
    def add_result(self, result: TestResult) -> None:
        """Add test result to collection."""
        self.results.append(result)
    
    def add_benchmark(self, benchmark: PerformanceBenchmark) -> None:
        """Add performance benchmark to collection."""
        self.benchmarks.append(benchmark)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "passed"])
        failed_tests = len([r for r in self.results if r.status == "failed"])
        
        total_time = sum(r.execution_time_ms for r in self.results)
        avg_memory = sum(r.memory_usage_mb for r in self.results) / total_tests if total_tests > 0 else 0
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_execution_time_ms": total_time,
                "average_memory_usage_mb": avg_memory,
                "test_duration_seconds": time.time() - self.start_time
            },
            "performance": {
                "benchmarks": len(self.benchmarks),
                "avg_operation_time_ms": sum(b.avg_execution_time_ms for b in self.benchmarks) / len(self.benchmarks) if self.benchmarks else 0,
                "max_operation_time_ms": max((b.max_execution_time_ms for b in self.benchmarks), default=0),
                "avg_memory_peak_mb": sum(b.memory_peak_mb for b in self.benchmarks) / len(self.benchmarks) if self.benchmarks else 0
            },
            "validation": {
                "meets_performance_threshold": all(
                    b.avg_execution_time_ms <= TEST_CONFIG["performance"]["max_stderr_generation_time_ms"]
                    for b in self.benchmarks
                ),
                "meets_memory_threshold": all(
                    b.memory_peak_mb <= TEST_CONFIG["performance"]["max_memory_usage_mb"]
                    for b in self.benchmarks
                ),
                "meets_success_threshold": passed_tests / total_tests >= TEST_CONFIG["validation"]["success_threshold"] if total_tests > 0 else False
            }
        }


class BaseTestCase(unittest.TestCase):
    """Enhanced base test case with performance and memory tracking."""
    
    def setUp(self):
        """Set up test case with performance tracking."""
        self.start_time = time.time()
        tracemalloc.start()
        self.metrics_collector = TestMetricsCollector()
        
        # Mock stderr capture for testing
        self.stderr_capture = []
        self.original_stderr = sys.stderr
        
    def tearDown(self):
        """Clean up and collect metrics."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = (time.time() - self.start_time) * 1000  # Convert to ms
        memory_usage = peak / 1024 / 1024  # Convert to MB
        
        # Create test result
        result = TestResult(
            test_name=self._testMethodName,
            status="passed",  # Will be updated if test fails
            execution_time_ms=execution_time,
            memory_usage_mb=memory_usage
        )
        self.metrics_collector.add_result(result)
        
        sys.stderr = self.original_stderr
    
    @contextmanager
    def capture_stderr(self):
        """Context manager to capture stderr output."""
        class StderrCapture:
            def __init__(self, capture_list):
                self.capture_list = capture_list
            
            def write(self, text):
                self.capture_list.append(text)
            
            def flush(self):
                pass
        
        sys.stderr = StderrCapture(self.stderr_capture)
        try:
            yield self.stderr_capture
        finally:
            sys.stderr = self.original_stderr
    
    def assertPerformanceWithin(self, operation_func: Callable, max_time_ms: float, max_memory_mb: float):
        """Assert that operation completes within performance thresholds."""
        tracemalloc.start()
        start_time = time.time()
        
        try:
            result = operation_func()
            execution_time = (time.time() - start_time) * 1000
            current, peak = tracemalloc.get_traced_memory()
            memory_usage = peak / 1024 / 1024
            
            self.assertLessEqual(
                execution_time, max_time_ms,
                f"Operation took {execution_time:.2f}ms, expected <= {max_time_ms}ms"
            )
            self.assertLessEqual(
                memory_usage, max_memory_mb,
                f"Operation used {memory_usage:.2f}MB, expected <= {max_memory_mb}MB"
            )
            
            return result
        finally:
            tracemalloc.stop()


class MockToolExecutionData:
    """Provides realistic mock data for tool execution testing."""
    
    @staticmethod
    def get_zen_chat_execution() -> Dict[str, Any]:
        """Mock ZEN chat tool execution data."""
        return {
            "tool_name": "mcp__zen__chat",
            "tool_input": {
                "prompt": "Help me refactor this component for better maintainability",
                "thinking_mode": "high",
                "context": "React component with mixed concerns"
            },
            "tool_response": {
                "status": "success",
                "analysis": {
                    "complexity": "moderate",
                    "recommendations": ["extract custom hooks", "separate concerns", "add prop validation"],
                    "coordination_needed": True
                },
                "guidance": "Proceed with Flow Worker coordination for implementation",
                "execution_time_ms": 234
            }
        }
    
    @staticmethod
    def get_flow_swarm_execution() -> Dict[str, Any]:
        """Mock Claude Flow swarm execution data."""
        return {
            "tool_name": "mcp__claude-flow__swarm_init",
            "tool_input": {
                "topology": "hierarchical",
                "maxAgents": 5,
                "strategy": "parallel",
                "task": "refactor-component"
            },
            "tool_response": {
                "status": "success",
                "swarm_id": "swarm_12345",
                "agents_spawned": 3,
                "coordination_plan": {
                    "lead_agent": "refactor-specialist",
                    "support_agents": ["test-generator", "type-checker"]
                },
                "memory_namespace": "refactor/component/session_001"
            }
        }
    
    @staticmethod
    def get_native_tool_sequence() -> List[Dict[str, Any]]:
        """Mock sequence of native tool executions."""
        return [
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "/src/components/UserProfile.tsx"},
                "tool_response": {"content": "// React component code..."}
            },
            {
                "tool_name": "Write",
                "tool_input": {"file_path": "/src/components/UserProfile.tsx", "content": "// Refactored code..."},
                "tool_response": {"status": "success"}
            },
            {
                "tool_name": "Bash",
                "tool_input": {"command": "npm test UserProfile.test.tsx"},
                "tool_response": {"exit_code": 0, "output": "Tests passed"}
            }
        ]
    
    @staticmethod
    def get_problematic_sequence() -> List[Dict[str, Any]]:
        """Mock problematic tool sequence that should trigger drift detection."""
        return [
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "/src/components/Form.tsx"},
                "tool_response": {"content": "// Component with issues..."}
            },
            {
                "tool_name": "Write",
                "tool_input": {"file_path": "/src/components/Form.tsx", "content": "// Quick fix..."},
                "tool_response": {"status": "success"}
            },
            {
                "tool_name": "Write",
                "tool_input": {"file_path": "/src/components/Button.tsx", "content": "// Another quick fix..."},
                "tool_response": {"status": "success"}
            },
            {
                "tool_name": "Bash",
                "tool_input": {"command": "git add -A && git commit -m 'quick fixes'"},
                "tool_response": {"exit_code": 0}
            }
        ]


class TestDataGenerator:
    """Generates comprehensive test data for various scenarios."""
    
    def __init__(self):
        self.scenario_templates = {
            "ideal_workflow": self._generate_ideal_workflow,
            "bypassed_zen": self._generate_bypassed_zen,
            "excessive_native": self._generate_excessive_native,
            "fragmented_workflow": self._generate_fragmented_workflow,
            "memory_coordination": self._generate_memory_coordination
        }
    
    def generate_scenario(self, scenario_name: str, variations: int = 5) -> List[Dict[str, Any]]:
        """Generate test scenarios with variations."""
        if scenario_name not in self.scenario_templates:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenarios = []
        for i in range(variations):
            scenario_data = self.scenario_templates[scenario_name](variation=i)
            scenarios.append(scenario_data)
        
        return scenarios
    
    def _generate_ideal_workflow(self, variation: int = 0) -> Dict[str, Any]:
        """Generate ideal workflow test data."""
        base_workflow = [
            MockToolExecutionData.get_zen_chat_execution(),
            MockToolExecutionData.get_flow_swarm_execution(),
            *MockToolExecutionData.get_native_tool_sequence()
        ]
        
        # Add variation-specific modifications
        if variation == 1:
            # Add memory coordination
            base_workflow.insert(2, {
                "tool_name": "mcp__claude-flow__memory_usage",
                "tool_input": {"action": "store", "key": "workflow/refactor"},
                "tool_response": {"status": "success", "memory_id": "mem_001"}
            })
        elif variation == 2:
            # Add filesystem coordination
            base_workflow.insert(-1, {
                "tool_name": "mcp__filesystem__batch_operation",
                "tool_input": {"operations": ["read", "write", "validate"]},
                "tool_response": {"status": "success", "batch_id": "batch_001"}
            })
        
        return {
            "scenario": "ideal_workflow",
            "variation": variation,
            "tool_sequence": base_workflow,
            "expected_drift": None,
            "expected_guidance": None
        }
    
    def _generate_bypassed_zen(self, variation: int = 0) -> Dict[str, Any]:
        """Generate bypassed ZEN scenario."""
        workflow = [
            MockToolExecutionData.get_flow_swarm_execution(),
            *MockToolExecutionData.get_native_tool_sequence()
        ]
        
        return {
            "scenario": "bypassed_zen",
            "variation": variation,
            "tool_sequence": workflow,
            "expected_drift": "BYPASSED_ZEN",
            "expected_guidance": "Queen ZEN must command before worker deployment"
        }
    
    def _generate_excessive_native(self, variation: int = 0) -> Dict[str, Any]:
        """Generate excessive native tool usage scenario."""
        workflow = MockToolExecutionData.get_native_tool_sequence() * (3 + variation)
        
        return {
            "scenario": "excessive_native",
            "variation": variation,
            "tool_sequence": workflow,
            "expected_drift": "NO_MCP_COORDINATION",
            "expected_guidance": "MCP coordination mandatory for complex workflows"
        }
    
    def _generate_fragmented_workflow(self, variation: int = 0) -> Dict[str, Any]:
        """Generate fragmented workflow scenario."""
        workflow = MockToolExecutionData.get_problematic_sequence()
        
        # Add more fragmentation based on variation
        if variation > 0:
            workflow.extend([
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": f"/src/utils/helper{variation}.ts"},
                    "tool_response": {"content": "// Utility code..."}
                },
                {
                    "tool_name": "Write", 
                    "tool_input": {"file_path": f"/src/utils/helper{variation}.ts", "content": "// Modified..."},
                    "tool_response": {"status": "success"}
                }
            ])
        
        return {
            "scenario": "fragmented_workflow",
            "variation": variation,
            "tool_sequence": workflow,
            "expected_drift": "FRAGMENTED_WORKFLOW",
            "expected_guidance": "Consider batching operations through MCP tools"
        }
    
    def _generate_memory_coordination(self, variation: int = 0) -> Dict[str, Any]:
        """Generate memory coordination test scenario."""
        workflow = [
            MockToolExecutionData.get_zen_chat_execution(),
            {
                "tool_name": "mcp__claude-flow__memory_usage",
                "tool_input": {"action": "store", "key": f"test/scenario_{variation}"},
                "tool_response": {"status": "success", "memory_id": f"mem_{variation:03d}"}
            },
            *MockToolExecutionData.get_native_tool_sequence()
        ]
        
        return {
            "scenario": "memory_coordination",
            "variation": variation,
            "tool_sequence": workflow,
            "expected_drift": None,
            "expected_guidance": None
        }


class PerformanceBenchmarkRunner:
    """Runs performance benchmarks for hook system components."""
    
    def __init__(self, iterations: int = None):
        self.iterations = iterations or TEST_CONFIG["performance"]["benchmark_iterations"]
        self.results: List[PerformanceBenchmark] = []
    
    def benchmark_stderr_generation(self, analyzer_class, test_data: List[Dict[str, Any]]) -> PerformanceBenchmark:
        """Benchmark stderr generation time for analyzer."""
        execution_times = []
        memory_peaks = []
        success_count = 0
        
        for i in range(self.iterations):
            tracemalloc.start()
            start_time = time.time()
            
            try:
                # Create analyzer instance
                analyzer = analyzer_class(priority=100)
                
                # Process test data and collect results for validation
                drift_results = []
                for data in test_data:
                    drift_result = analyzer.analyze_drift(
                        data["tool_name"],
                        data["tool_input"], 
                        data["tool_response"]
                    )
                    drift_results.append(drift_result)
                
                execution_time = (time.time() - start_time) * 1000
                current, peak = tracemalloc.get_traced_memory()
                memory_peak = peak / 1024 / 1024
                
                execution_times.append(execution_time)
                memory_peaks.append(memory_peak)
                success_count += 1
                
            except Exception as e:
                # Log error but continue benchmarking
                logging.warning(f"Benchmark iteration {i} failed: {e}")
            finally:
                tracemalloc.stop()
        
        # Calculate benchmark metrics
        if execution_times:
            benchmark = PerformanceBenchmark(
                operation_name=f"{analyzer_class.__name__}_stderr_generation",
                avg_execution_time_ms=sum(execution_times) / len(execution_times),
                max_execution_time_ms=max(execution_times),
                min_execution_time_ms=min(execution_times),
                memory_peak_mb=max(memory_peaks) if memory_peaks else 0,
                memory_avg_mb=sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0,
                operations_per_second=1000 / (sum(execution_times) / len(execution_times)) if execution_times else 0,
                success_rate=success_count / self.iterations
            )
        else:
            # Failed benchmark
            benchmark = PerformanceBenchmark(
                operation_name=f"{analyzer_class.__name__}_stderr_generation",
                avg_execution_time_ms=float('inf'),
                max_execution_time_ms=float('inf'),
                min_execution_time_ms=float('inf'),
                memory_peak_mb=0,
                memory_avg_mb=0,
                operations_per_second=0,
                success_rate=0
            )
        
        self.results.append(benchmark)
        return benchmark
    
    def benchmark_pipeline_integration(self, pipeline_func: Callable, test_scenarios: List[Dict[str, Any]]) -> PerformanceBenchmark:
        """Benchmark full pipeline integration performance."""
        execution_times = []
        memory_peaks = []
        success_count = 0
        
        for i in range(min(self.iterations, len(test_scenarios) * 10)):  # Limit iterations for integration tests
            scenario = test_scenarios[i % len(test_scenarios)]
            
            tracemalloc.start()
            start_time = time.time()
            
            try:
                # Run pipeline with scenario and collect results for validation
                pipeline_func(scenario["tool_sequence"])
                
                execution_time = (time.time() - start_time) * 1000
                current, peak = tracemalloc.get_traced_memory()
                memory_peak = peak / 1024 / 1024
                
                execution_times.append(execution_time)
                memory_peaks.append(memory_peak)
                success_count += 1
                
            except Exception as e:
                logging.warning(f"Pipeline benchmark iteration {i} failed: {e}")
            finally:
                tracemalloc.stop()
        
        # Calculate benchmark metrics
        benchmark = PerformanceBenchmark(
            operation_name="pipeline_integration",
            avg_execution_time_ms=sum(execution_times) / len(execution_times) if execution_times else float('inf'),
            max_execution_time_ms=max(execution_times) if execution_times else float('inf'),
            min_execution_time_ms=min(execution_times) if execution_times else float('inf'),
            memory_peak_mb=max(memory_peaks) if memory_peaks else 0,
            memory_avg_mb=sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0,
            operations_per_second=1000 / (sum(execution_times) / len(execution_times)) if execution_times else 0,
            success_rate=success_count / min(self.iterations, len(test_scenarios) * 10)
        )
        
        self.results.append(benchmark)
        return benchmark
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive benchmark summary."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        return {
            "total_benchmarks": len(self.results),
            "avg_execution_time_ms": sum(b.avg_execution_time_ms for b in self.results) / len(self.results),
            "max_execution_time_ms": max(b.max_execution_time_ms for b in self.results),
            "avg_memory_peak_mb": sum(b.memory_peak_mb for b in self.results) / len(self.results),
            "avg_operations_per_second": sum(b.operations_per_second for b in self.results) / len(self.results),
            "overall_success_rate": sum(b.success_rate for b in self.results) / len(self.results),
            "performance_threshold_met": all(
                b.avg_execution_time_ms <= TEST_CONFIG["performance"]["max_stderr_generation_time_ms"]
                for b in self.results
            ),
            "memory_threshold_met": all(
                b.memory_peak_mb <= TEST_CONFIG["performance"]["max_memory_usage_mb"]
                for b in self.results
            )
        }


class ValidationFramework:
    """Progressive rollout testing framework with success metrics."""
    
    def __init__(self):
        self.stages = TEST_CONFIG["validation"]["rollout_stages"]
        self.success_threshold = TEST_CONFIG["validation"]["success_threshold"]
        self.current_stage = 0
        self.stage_results: Dict[str, List[TestResult]] = {}
    
    def validate_stage(self, stage_name: str, test_suite: Callable) -> Dict[str, Any]:
        """Validate a rollout stage with comprehensive testing."""
        logging.info(f"Starting validation for stage: {stage_name}")
        
        stage_start_time = time.time()
        stage_results = []
        
        try:
            # Run test suite for this stage
            results = test_suite()
            stage_results.extend(results)
            
            # Analyze results
            passed_tests = len([r for r in results if r.status == "passed"])
            total_tests = len(results)
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            # Calculate metrics
            avg_execution_time = sum(r.execution_time_ms for r in results) / total_tests if total_tests > 0 else 0
            max_execution_time = max((r.execution_time_ms for r in results), default=0)
            avg_memory_usage = sum(r.memory_usage_mb for r in results) / total_tests if total_tests > 0 else 0
            
            # Determine if stage passes validation
            stage_passed = (
                success_rate >= self.success_threshold and
                avg_execution_time <= TEST_CONFIG["performance"]["max_stderr_generation_time_ms"] and
                avg_memory_usage <= TEST_CONFIG["performance"]["max_memory_usage_mb"]
            )
            
            validation_result = {
                "stage": stage_name,
                "status": "passed" if stage_passed else "failed",
                "metrics": {
                    "success_rate": success_rate,
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "avg_execution_time_ms": avg_execution_time,
                    "max_execution_time_ms": max_execution_time,
                    "avg_memory_usage_mb": avg_memory_usage,
                    "stage_duration_seconds": time.time() - stage_start_time
                },
                "thresholds_met": {
                    "success_rate": success_rate >= self.success_threshold,
                    "performance": avg_execution_time <= TEST_CONFIG["performance"]["max_stderr_generation_time_ms"],
                    "memory": avg_memory_usage <= TEST_CONFIG["performance"]["max_memory_usage_mb"]
                },
                "recommendation": "proceed" if stage_passed else "investigate_and_fix"
            }
            
            self.stage_results[stage_name] = stage_results
            logging.info(f"Stage {stage_name} validation: {'PASSED' if stage_passed else 'FAILED'}")
            
            return validation_result
            
        except Exception as e:
            logging.exception(f"Stage {stage_name} validation failed with error: {e}")
            return {
                "stage": stage_name,
                "status": "error",
                "error": str(e),
                "metrics": {},
                "recommendation": "fix_critical_issues"
            }
    
    def run_progressive_rollout(self, test_suites: Dict[str, Callable]) -> Dict[str, Any]:
        """Run progressive rollout validation across all stages."""
        rollout_results = {}
        overall_success = True
        
        for stage in self.stages:
            if stage not in test_suites:
                logging.warning(f"No test suite provided for stage: {stage}")
                continue
            
            stage_result = self.validate_stage(stage, test_suites[stage])
            rollout_results[stage] = stage_result
            
            # Stop rollout if stage fails
            if stage_result["status"] != "passed":
                overall_success = False
                logging.error(f"Rollout stopped at stage {stage} due to validation failure")
                break
        
        return {
            "rollout_status": "success" if overall_success else "failed",
            "stages_completed": len(rollout_results),
            "total_stages": len(self.stages),
            "stage_results": rollout_results,
            "overall_metrics": self._calculate_overall_metrics(),
            "recommendations": self._generate_recommendations(rollout_results)
        }
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics across all stages."""
        all_results = []
        for stage_results in self.stage_results.values():
            all_results.extend(stage_results)
        
        if not all_results:
            return {"error": "No test results available"}
        
        total_tests = len(all_results)
        passed_tests = len([r for r in all_results if r.status == "passed"])
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "overall_success_rate": passed_tests / total_tests,
            "avg_execution_time_ms": sum(r.execution_time_ms for r in all_results) / total_tests,
            "avg_memory_usage_mb": sum(r.memory_usage_mb for r in all_results) / total_tests,
            "meets_success_threshold": passed_tests / total_tests >= self.success_threshold
        }
    
    def _generate_recommendations(self, rollout_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on rollout results."""
        recommendations = []
        
        for stage, result in rollout_results.items():
            if result["status"] == "failed":
                if not result["thresholds_met"]["success_rate"]:
                    recommendations.append(f"Improve test reliability in {stage} stage")
                if not result["thresholds_met"]["performance"]:
                    recommendations.append(f"Optimize performance in {stage} stage")
                if not result["thresholds_met"]["memory"]:
                    recommendations.append(f"Reduce memory usage in {stage} stage")
            elif result["status"] == "error":
                recommendations.append(f"Fix critical errors in {stage} stage")
        
        if not recommendations:
            recommendations.append("All stages passed - ready for full deployment")
        
        return recommendations


# Export main components for use in other test modules
__all__ = [
    'BaseTestCase',
    'TestResult', 
    'PerformanceBenchmark',
    'TestMetricsCollector',
    'MockToolExecutionData',
    'TestDataGenerator',
    'PerformanceBenchmarkRunner',
    'ValidationFramework',
    'TEST_CONFIG'
]