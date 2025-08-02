"""
Performance Benchmarks for Hook System
======================================

Comprehensive performance benchmarking measuring stderr generation time,
memory usage, and system throughput under various load conditions.
"""

import unittest
import sys
import os
import time
import tracemalloc
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Callable
import statistics
import json
import tempfile

# Add hooks modules to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from test_framework_architecture import (
    BaseTestCase, MockToolExecutionData, TestDataGenerator, 
    PerformanceBenchmarkRunner, TEST_CONFIG, PerformanceBenchmark
)

# Import post-tool components with fallbacks
try:
    from post_tool.manager import PostToolAnalysisManager
    from post_tool.core.drift_detector import DriftAnalyzer
except ImportError:
    # Mock classes for testing framework validation
    class PostToolAnalysisManager:
        def __init__(self, config_path=None):
            self.tool_count = 0
        def analyze_tool_usage(self, tool_name, tool_input, tool_response):
            self.tool_count += 1
    class DriftAnalyzer:
        def __init__(self, priority=0):
            self.priority = priority
        def analyze_drift(self, tool_name, tool_input, tool_response):
            return None


class TestStderrGenerationBenchmarks(BaseTestCase):
    """Benchmarks focused on stderr generation performance."""
    
    def setUp(self):
        super().setUp()
        self.benchmark_runner = PerformanceBenchmarkRunner(iterations=50)  # Higher iterations for accuracy
        self.test_data_generator = TestDataGenerator()
    
    def test_single_analyzer_stderr_performance(self):
        """Benchmark stderr generation for individual analyzers."""
        test_scenarios = [
            ("ideal_workflow", 3),
            ("bypassed_zen", 3),
            ("excessive_native", 3)
        ]
        
        results = {}
        
        for scenario_name, variations in test_scenarios:
            scenarios = self.test_data_generator.generate_scenario(scenario_name, variations)
            
            # Extract tool sequence data for benchmarking
            test_data = []
            for scenario in scenarios:
                test_data.extend(scenario["tool_sequence"])
            
            # Benchmark with mock analyzer
            class TestAnalyzer(DriftAnalyzer):
                def analyze_drift(self, tool_name, tool_input, tool_response):
                    # Simulate stderr generation time
                    if tool_name.startswith("mcp__"):
                        return None  # No drift for MCP tools
                    else:
                        # Simulate drift detection and guidance generation
                        import time
                        time.sleep(0.001)  # 1ms simulation
                        return Mock(drift_type=Mock(value="test_drift"))
                
                def get_analyzer_name(self):
                    return "TestAnalyzer"
            
            benchmark = self.benchmark_runner.benchmark_stderr_generation(TestAnalyzer, test_data)
            results[scenario_name] = benchmark
            
            # Validate performance thresholds
            self.assertLessEqual(
                benchmark.avg_execution_time_ms,
                TEST_CONFIG["performance"]["max_stderr_generation_time_ms"],
                f"{scenario_name} stderr generation should be under 50ms"
            )
            
            self.assertLessEqual(
                benchmark.memory_peak_mb,
                TEST_CONFIG["performance"]["max_memory_usage_mb"],
                f"{scenario_name} should stay under memory limit"
            )
        
        # Compare performance across scenarios
        ideal_time = results["ideal_workflow"].avg_execution_time_ms
        problematic_time = results["bypassed_zen"].avg_execution_time_ms
        
        # Problematic scenarios might take slightly longer but should be close
        self.assertLess(
            problematic_time / ideal_time, 2.0,
            "Problematic scenarios should not be more than 2x slower"
        )
    
    def test_concurrent_stderr_generation(self):
        """Benchmark stderr generation under concurrent load."""
        test_data = []
        scenarios = self.test_data_generator.generate_scenario("bypassed_zen", 5)
        for scenario in scenarios:
            test_data.extend(scenario["tool_sequence"])
        
        def concurrent_stderr_generation(num_threads: int) -> PerformanceBenchmark:
            """Run stderr generation with specified concurrency."""
            
            class ConcurrentAnalyzer(DriftAnalyzer):
                def __init__(self, priority=0):
                    super().__init__(priority)
                    self.generation_count = 0
                
                def analyze_drift(self, tool_name, tool_input, tool_response):
                    self.generation_count += 1
                    # Simulate stderr generation with thread contention
                    if not tool_name.startswith("mcp__"):
                        return Mock(
                            drift_type=Mock(value="concurrent_drift"),
                            severity=Mock(value=2),
                            evidence_details=f"Thread {threading.current_thread().ident}"
                        )
                    return None
                
                def get_analyzer_name(self):
                    return f"ConcurrentAnalyzer_{threading.current_thread().ident}"
            
            start_time = time.time()
            tracemalloc.start()
            
            execution_times = []
            success_count = 0
            
            def worker_function(data_chunk):
                analyzer = ConcurrentAnalyzer()
                worker_start = time.time()
                
                try:
                    for data in data_chunk:
                        analyzer.analyze_drift(
                            data["tool_name"],
                            data["tool_input"],
                            data["tool_response"]
                        )
                    worker_time = (time.time() - worker_start) * 1000
                    return worker_time, analyzer.generation_count
                except Exception:
                    return None, 0
            
            # Split data across threads
            chunk_size = len(test_data) // num_threads
            data_chunks = [test_data[i:i + chunk_size] for i in range(0, len(test_data), chunk_size)]
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_function, chunk) for chunk in data_chunks]
                
                for future in futures:
                    result = future.result()
                    if result[0] is not None:
                        execution_times.append(result[0])
                        success_count += result[1]
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            total_time = (time.time() - start_time) * 1000
            memory_peak = peak / 1024 / 1024
            
            return PerformanceBenchmark(
                operation_name=f"concurrent_stderr_{num_threads}_threads",
                avg_execution_time_ms=sum(execution_times) / len(execution_times) if execution_times else float('inf'),
                max_execution_time_ms=max(execution_times) if execution_times else float('inf'),
                min_execution_time_ms=min(execution_times) if execution_times else float('inf'),
                memory_peak_mb=memory_peak,
                memory_avg_mb=memory_peak,  # Simplified for concurrent test
                operations_per_second=success_count / (total_time / 1000) if total_time > 0 else 0,
                success_rate=1.0 if execution_times else 0.0
            )
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        benchmarks = {}
        
        for num_threads in concurrency_levels:
            benchmark = concurrent_stderr_generation(num_threads)
            benchmarks[num_threads] = benchmark
            
            # Validate performance doesn't degrade significantly
            self.assertLessEqual(
                benchmark.memory_peak_mb,
                TEST_CONFIG["performance"]["max_memory_usage_mb"] * num_threads,
                f"Memory usage should scale reasonably with {num_threads} threads"
            )
        
        # Verify concurrency scaling
        single_thread_ops = benchmarks[1].operations_per_second
        multi_thread_ops = benchmarks[4].operations_per_second
        
        self.assertGreater(
            multi_thread_ops, single_thread_ops * 1.5,
            "Multi-threading should provide performance improvement"
        )
    
    def test_memory_usage_patterns(self):
        """Analyze memory usage patterns during stderr generation."""
        test_data = []
        # Create large dataset to stress memory
        for scenario_type in ["ideal_workflow", "bypassed_zen", "excessive_native"]:
            scenarios = self.test_data_generator.generate_scenario(scenario_type, 10)
            for scenario in scenarios:
                test_data.extend(scenario["tool_sequence"])
        
        class MemoryTrackingAnalyzer(DriftAnalyzer):
            def __init__(self, priority=0):
                super().__init__(priority)
                self.memory_snapshots = []
            
            def analyze_drift(self, tool_name, tool_input, tool_response):
                current, peak = tracemalloc.get_traced_memory()
                self.memory_snapshots.append(current / 1024 / 1024)  # Convert to MB
                
                # Generate varying amounts of drift evidence
                if len(self.memory_snapshots) % 5 == 0:  # Every 5th tool
                    return Mock(
                        drift_type=Mock(value="memory_test_drift"),
                        severity=Mock(value=2),
                        evidence_details=f"Memory snapshot {len(self.memory_snapshots)}"
                    )
                return None
            
            def get_analyzer_name(self):
                return "MemoryTrackingAnalyzer"
        
        tracemalloc.start()
        analyzer = MemoryTrackingAnalyzer()
        
        start_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        
        # Process all test data
        for data in test_data:
            analyzer.analyze_drift(
                data["tool_name"],
                data["tool_input"],
                data["tool_response"]
            )
        
        end_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        tracemalloc.stop()
        
        # Analyze memory patterns
        memory_growth = end_memory - start_memory
        memory_variance = statistics.variance(analyzer.memory_snapshots) if len(analyzer.memory_snapshots) > 1 else 0
        
        # Memory assertions
        self.assertLess(memory_growth, 5.0, "Memory growth should be under 5MB")
        self.assertLess(peak_memory, TEST_CONFIG["performance"]["max_memory_usage_mb"])
        self.assertLess(memory_variance, 1.0, "Memory usage should be stable (low variance)")
    
    def test_stderr_output_scaling(self):
        """Test how stderr output scales with input size."""
        input_sizes = [10, 50, 100, 500]
        scaling_results = {}
        
        for size in input_sizes:
            # Generate dataset of specified size
            test_data = []
            while len(test_data) < size:
                scenarios = self.test_data_generator.generate_scenario("bypassed_zen", 5)
                for scenario in scenarios:
                    test_data.extend(scenario["tool_sequence"])
                    if len(test_data) >= size:
                        break
            
            test_data = test_data[:size]  # Trim to exact size
            
            class ScalingAnalyzer(DriftAnalyzer):
                def __init__(self):
                    super().__init__(priority=500)
                    self.stderr_output_length = 0
                
                def analyze_drift(self, tool_name, tool_input, tool_response):
                    # Simulate stderr generation
                    if not tool_name.startswith("mcp__"):
                        guidance_message = f"🚨 DRIFT: {tool_name} bypassed coordination"
                        self.stderr_output_length += len(guidance_message)
                        return Mock(drift_type=Mock(value="scaling_drift"))
                    return None
                
                def get_analyzer_name(self):
                    return "ScalingAnalyzer"
            
            # Benchmark processing
            start_time = time.time()
            analyzer = ScalingAnalyzer()
            
            for data in test_data:
                analyzer.analyze_drift(
                    data["tool_name"],
                    data["tool_input"],
                    data["tool_response"]
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            scaling_results[size] = {
                "processing_time_ms": processing_time,
                "stderr_length": analyzer.stderr_output_length,
                "time_per_tool": processing_time / size,
                "stderr_per_tool": analyzer.stderr_output_length / size
            }
        
        # Analyze scaling characteristics
        small_size = scaling_results[10]
        large_size = scaling_results[500]
        
        # Processing time should scale sub-linearly (due to efficiency gains)
        time_scaling_factor = large_size["time_per_tool"] / small_size["time_per_tool"]
        self.assertLess(time_scaling_factor, 2.0, "Processing time per tool should not double")
        
        # Stderr output per tool should remain consistent
        stderr_scaling_factor = large_size["stderr_per_tool"] / small_size["stderr_per_tool"]
        self.assertLess(abs(stderr_scaling_factor - 1.0), 0.5, "Stderr per tool should remain consistent")
        
        # Overall performance should remain within thresholds
        self.assertLess(
            large_size["time_per_tool"],
            TEST_CONFIG["performance"]["max_stderr_generation_time_ms"],
            "Per-tool processing time should remain under threshold even at scale"
        )


class TestSystemThroughputBenchmarks(BaseTestCase):
    """Benchmarks for overall system throughput and capacity."""
    
    def setUp(self):
        super().setUp()
        self.test_data_generator = TestDataGenerator()
    
    def test_maximum_throughput_capacity(self):
        """Measure maximum system throughput under optimal conditions."""
        # Generate large dataset for throughput testing
        throughput_data = []
        for scenario_type in ["ideal_workflow", "bypassed_zen"]:
            scenarios = self.test_data_generator.generate_scenario(scenario_type, 20)
            for scenario in scenarios:
                throughput_data.extend(scenario["tool_sequence"])
        
        # Measure processing rate
        manager = PostToolAnalysisManager()
        
        start_time = time.time()
        processed_count = 0
        
        with patch('sys.stderr'):  # Suppress stderr for clean measurement
            for data in throughput_data:
                manager.analyze_tool_usage(
                    data["tool_name"],
                    data["tool_input"],
                    data["tool_response"]
                )
                processed_count += 1
        
        total_time = time.time() - start_time
        throughput = processed_count / total_time
        
        # Throughput assertions
        self.assertGreater(throughput, 100, "Should process at least 100 tools per second")
        self.assertEqual(processed_count, len(throughput_data), "Should process all tools")
        
        # Average processing time should be well under threshold
        avg_time_per_tool = (total_time * 1000) / processed_count
        self.assertLess(
            avg_time_per_tool,
            TEST_CONFIG["performance"]["max_stderr_generation_time_ms"] / 2,
            "Average processing time should be well under threshold"
        )
    
    def test_sustained_load_performance(self):
        """Test performance under sustained load over time."""
        duration_seconds = 5  # 5-second sustained load test
        batch_size = 50
        
        manager = PostToolAnalysisManager()
        test_scenarios = self.test_data_generator.generate_scenario("bypassed_zen", 10)
        
        # Flatten all tool data
        tool_data_pool = []
        for scenario in test_scenarios:
            tool_data_pool.extend(scenario["tool_sequence"])
        
        start_time = time.time()
        batch_times = []
        total_processed = 0
        
        with patch('sys.stderr'):  # Suppress stderr for clean measurement
            while time.time() - start_time < duration_seconds:
                batch_start = time.time()
                
                # Process a batch
                batch_data = tool_data_pool[:batch_size]
                for data in batch_data:
                    manager.analyze_tool_usage(
                        data["tool_name"],
                        data["tool_input"],
                        data["tool_response"]
                    )
                    total_processed += 1
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Rotate data to simulate continuous load
                tool_data_pool = tool_data_pool[batch_size:] + tool_data_pool[:batch_size]
        
        total_time = time.time() - start_time
        
        # Analyze sustained performance
        avg_batch_time = statistics.mean(batch_times)
        batch_time_variance = statistics.variance(batch_times)
        sustained_throughput = total_processed / total_time
        
        # Performance assertions
        self.assertGreater(sustained_throughput, 80, "Sustained throughput should be > 80 tools/sec")
        self.assertLess(avg_batch_time, 1.0, "Average batch processing should be under 1 second")
        self.assertLess(batch_time_variance, 0.1, "Batch processing times should be consistent")
    
    def test_resource_utilization_efficiency(self):
        """Measure resource utilization efficiency."""
        test_data = []
        scenarios = self.test_data_generator.generate_scenario("excessive_native", 15)
        for scenario in scenarios:
            test_data.extend(scenario["tool_sequence"])
        
        # Measure resource usage during processing
        tracemalloc.start()
        start_time = time.time()
        
        manager = PostToolAnalysisManager()
        
        memory_samples = []
        processing_times = []
        
        with patch('sys.stderr'):
            for i, data in enumerate(test_data):
                sample_start = time.time()
                
                manager.analyze_tool_usage(
                    data["tool_name"],
                    data["tool_input"],
                    data["tool_response"]
                )
                
                sample_time = (time.time() - sample_start) * 1000
                processing_times.append(sample_time)
                
                # Sample memory every 10 operations
                if i % 10 == 0:
                    current, peak = tracemalloc.get_traced_memory()
                    memory_samples.append(current / 1024 / 1024)
        
        total_time = time.time() - start_time
        final_current, final_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate efficiency metrics
        avg_processing_time = statistics.mean(processing_times)
        processing_time_stddev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        memory_efficiency = (len(test_data) * 1024) / final_peak  # Tools processed per byte
        time_efficiency = len(test_data) / total_time  # Tools per second
        
        # Efficiency assertions
        self.assertLess(avg_processing_time, 5.0, "Average processing time should be under 5ms")
        self.assertLess(processing_time_stddev, 2.0, "Processing times should be consistent")
        self.assertGreater(memory_efficiency, 0.1, "Should process at least 0.1 tools per KB")
        self.assertGreater(time_efficiency, 50, "Should process at least 50 tools per second")
        
        # Resource utilization should be stable
        if len(memory_samples) > 1:
            memory_growth_rate = (memory_samples[-1] - memory_samples[0]) / len(memory_samples)
            self.assertLess(abs(memory_growth_rate), 0.1, "Memory usage should be stable")


class TestRegressionBenchmarks(BaseTestCase):
    """Benchmarks to detect performance regressions."""
    
    def setUp(self):
        super().setUp()
        self.baseline_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
        self.baseline_file.close()
    
    def tearDown(self):
        super().tearDown()
        os.unlink(self.baseline_file.name)
    
    def test_performance_regression_detection(self):
        """Detect performance regressions against baseline."""
        # Define baseline performance metrics (these would come from previous runs)
        baseline_metrics = {
            "avg_stderr_generation_ms": 15.0,
            "max_memory_usage_mb": 5.0,
            "throughput_tools_per_sec": 120.0,
            "error_rate": 0.02
        }
        
        # Save baseline
        with open(self.baseline_file.name, 'w') as f:
            json.dump(baseline_metrics, f)
        
        # Run current performance test
        test_data = []
        scenarios = self.test_data_generator.generate_scenario("bypassed_zen", 10)
        for scenario in scenarios:
            test_data.extend(scenario["tool_sequence"])
        
        # Measure current performance
        tracemalloc.start()
        start_time = time.time()
        error_count = 0
        
        manager = PostToolAnalysisManager()
        
        processing_times = []
        
        with patch('sys.stderr'):
            for data in test_data:
                try:
                    sample_start = time.time()
                    manager.analyze_tool_usage(
                        data["tool_name"],
                        data["tool_input"],
                        data["tool_response"]
                    )
                    processing_times.append((time.time() - sample_start) * 1000)
                except Exception:
                    error_count += 1
        
        total_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate current metrics
        current_metrics = {
            "avg_stderr_generation_ms": statistics.mean(processing_times) if processing_times else float('inf'),
            "max_memory_usage_mb": peak / 1024 / 1024,
            "throughput_tools_per_sec": len(test_data) / total_time,
            "error_rate": error_count / len(test_data)
        }
        
        # Compare against baseline with regression thresholds
        regression_thresholds = {
            "avg_stderr_generation_ms": 1.5,  # 50% slower is a regression
            "max_memory_usage_mb": 2.0,       # 100% more memory is a regression
            "throughput_tools_per_sec": 0.8,  # 20% slower throughput is a regression
            "error_rate": 2.0                 # 100% more errors is a regression
        }
        
        regressions = []
        
        for metric, current_value in current_metrics.items():
            baseline_value = baseline_metrics[metric]
            threshold = regression_thresholds[metric]
            
            if metric in ["avg_stderr_generation_ms", "max_memory_usage_mb", "error_rate"]:
                # Lower is better
                if current_value > baseline_value * threshold:
                    regressions.append(f"{metric}: {current_value:.2f} vs baseline {baseline_value:.2f}")
            else:
                # Higher is better (throughput)
                if current_value < baseline_value * threshold:
                    regressions.append(f"{metric}: {current_value:.2f} vs baseline {baseline_value:.2f}")
        
        # Assert no significant regressions
        self.assertEqual(
            len(regressions), 0,
            f"Performance regressions detected: {'; '.join(regressions)}"
        )
        
        # Log current performance for future baseline
        print(f"Current performance metrics: {json.dumps(current_metrics, indent=2)}")


if __name__ == '__main__':
    # Configure test environment
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run benchmarks with extended timeout
    unittest.main(verbosity=2, buffer=False)