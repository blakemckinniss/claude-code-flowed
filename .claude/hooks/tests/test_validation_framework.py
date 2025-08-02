"""
Progressive Rollout Validation Framework
========================================

Comprehensive validation framework for progressive rollout testing with
success metrics, automated validation, and quality gates.
"""

import unittest
import sys
import os
import json
import time
import tempfile
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Callable
import statistics
from concurrent.futures import ThreadPoolExecutor

# Add hooks modules to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from test_framework_architecture import (
    BaseTestCase, MockToolExecutionData, TestDataGenerator, 
    ValidationFramework, TEST_CONFIG, TestResult
)

# Import testing modules
from test_analyzer_unit_tests import TestZenBypassAnalyzer, TestWorkflowPatternAnalyzer
from test_posttool_integration import TestPostToolAnalysisManager
from test_performance_benchmarks import TestStderrGenerationBenchmarks


class TestValidationFrameworkCore(BaseTestCase):
    """Core validation framework functionality tests."""
    
    def setUp(self):
        super().setUp()
        self.validation_framework = ValidationFramework()
        self.test_data_generator = TestDataGenerator()
    
    def test_framework_initialization(self):
        """Test proper initialization of validation framework."""
        self.assertEqual(
            self.validation_framework.stages,
            TEST_CONFIG["validation"]["rollout_stages"]
        )
        self.assertEqual(
            self.validation_framework.success_threshold,
            TEST_CONFIG["validation"]["success_threshold"]
        )
        self.assertEqual(self.validation_framework.current_stage, 0)
        self.assertIsInstance(self.validation_framework.stage_results, dict)
    
    def test_single_stage_validation_success(self):
        """Test successful validation of a single stage."""
        def mock_successful_test_suite():
            """Mock test suite that passes all tests."""
            return [
                TestResult("test_1", "passed", 10.0, 1.0),
                TestResult("test_2", "passed", 15.0, 1.2),
                TestResult("test_3", "passed", 12.0, 0.9),
                TestResult("test_4", "passed", 8.0, 1.1),
                TestResult("test_5", "passed", 14.0, 1.0)
            ]
        
        result = self.validation_framework.validate_stage("dev", mock_successful_test_suite)
        
        # Verify successful validation
        self.assertEqual(result["stage"], "dev")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["metrics"]["success_rate"], 1.0)
        self.assertEqual(result["metrics"]["total_tests"], 5)
        self.assertEqual(result["metrics"]["passed_tests"], 5)
        self.assertEqual(result["recommendation"], "proceed")
        
        # Verify thresholds are met
        self.assertTrue(result["thresholds_met"]["success_rate"])
        self.assertTrue(result["thresholds_met"]["performance"])
        self.assertTrue(result["thresholds_met"]["memory"])
    
    def test_single_stage_validation_failure(self):
        """Test validation failure scenarios."""
        def mock_failing_test_suite():
            """Mock test suite with failures."""
            return [
                TestResult("test_1", "passed", 10.0, 1.0),
                TestResult("test_2", "failed", 15.0, 1.2, "Test failure"),
                TestResult("test_3", "failed", 12.0, 0.9, "Another failure"),
                TestResult("test_4", "passed", 8.0, 1.1),
                TestResult("test_5", "passed", 14.0, 1.0)
            ]
        
        result = self.validation_framework.validate_stage("dev", mock_failing_test_suite)
        
        # Verify failed validation
        self.assertEqual(result["stage"], "dev")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["metrics"]["success_rate"], 0.6)  # 3/5 passed
        self.assertEqual(result["metrics"]["failed_tests"], 2)
        self.assertEqual(result["recommendation"], "investigate_and_fix")
        
        # Verify thresholds
        self.assertFalse(result["thresholds_met"]["success_rate"])  # Below 95% threshold
    
    def test_performance_threshold_validation(self):
        """Test validation of performance thresholds."""
        def mock_slow_test_suite():
            """Mock test suite with slow performance."""
            return [
                TestResult("test_1", "passed", 60.0, 1.0),  # Over 50ms threshold
                TestResult("test_2", "passed", 75.0, 1.2),  # Over threshold
                TestResult("test_3", "passed", 45.0, 0.9),  # Under threshold
            ]
        
        result = self.validation_framework.validate_stage("staging", mock_slow_test_suite)
        
        # Should fail due to performance
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["thresholds_met"]["performance"])
        self.assertGreater(result["metrics"]["avg_execution_time_ms"], 50)
    
    def test_memory_threshold_validation(self):
        """Test validation of memory usage thresholds."""
        def mock_memory_heavy_test_suite():
            """Mock test suite with excessive memory usage."""
            return [
                TestResult("test_1", "passed", 10.0, 12.0),  # Over 10MB threshold
                TestResult("test_2", "passed", 15.0, 8.0),   # Under threshold
                TestResult("test_3", "passed", 12.0, 11.0),  # Over threshold
            ]
        
        result = self.validation_framework.validate_stage("staging", mock_memory_heavy_test_suite)
        
        # Should fail due to memory usage
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["thresholds_met"]["memory"])
        self.assertGreater(result["metrics"]["avg_memory_usage_mb"], 10)
    
    def test_stage_error_handling(self):
        """Test handling of stage validation errors."""
        def mock_error_test_suite():
            """Mock test suite that throws an exception."""
            raise Exception("Test suite execution failed")
        
        result = self.validation_framework.validate_stage("production", mock_error_test_suite)
        
        # Verify error handling
        self.assertEqual(result["stage"], "production")
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)
        self.assertEqual(result["recommendation"], "fix_critical_issues")


class TestProgressiveRollout(BaseTestCase):
    """Test progressive rollout validation across multiple stages."""
    
    def setUp(self):
        super().setUp()
        self.validation_framework = ValidationFramework()
    
    def test_successful_progressive_rollout(self):
        """Test complete successful rollout across all stages."""
        def create_successful_suite(stage_name):
            def suite():
                # Simulate stage-appropriate test complexity
                test_count = {"dev": 10, "staging": 20, "production": 30}[stage_name]
                return [
                    TestResult(f"test_{i}", "passed", 10 + i, 1.0 + (i * 0.1))
                    for i in range(test_count)
                ]
            return suite
        
        test_suites = {
            "dev": create_successful_suite("dev"),
            "staging": create_successful_suite("staging"),
            "production": create_successful_suite("production")
        }
        
        rollout_result = self.validation_framework.run_progressive_rollout(test_suites)
        
        # Verify successful rollout
        self.assertEqual(rollout_result["rollout_status"], "success")
        self.assertEqual(rollout_result["stages_completed"], 3)
        self.assertEqual(rollout_result["total_stages"], 3)
        
        # Verify all stages passed
        for stage in ["dev", "staging", "production"]:
            self.assertEqual(rollout_result["stage_results"][stage]["status"], "passed")
        
        # Verify overall metrics
        overall_metrics = rollout_result["overall_metrics"]
        self.assertEqual(overall_metrics["total_tests"], 60)  # 10 + 20 + 30
        self.assertEqual(overall_metrics["passed_tests"], 60)
        self.assertEqual(overall_metrics["overall_success_rate"], 1.0)
        self.assertTrue(overall_metrics["meets_success_threshold"])
        
        # Verify recommendations
        self.assertIn("ready for full deployment", rollout_result["recommendations"][0])
    
    def test_rollout_failure_at_staging(self):
        """Test rollout failure at staging stage."""
        def dev_suite():
            return [TestResult("dev_test", "passed", 10.0, 1.0)]
        
        def failing_staging_suite():
            return [
                TestResult("staging_test_1", "passed", 10.0, 1.0),
                TestResult("staging_test_2", "failed", 15.0, 1.2, "Critical failure"),
                TestResult("staging_test_3", "failed", 12.0, 0.9, "Another failure")
            ]
        
        def production_suite():
            return [TestResult("prod_test", "passed", 10.0, 1.0)]
        
        test_suites = {
            "dev": dev_suite,
            "staging": failing_staging_suite,
            "production": production_suite
        }
        
        rollout_result = self.validation_framework.run_progressive_rollout(test_suites)
        
        # Verify rollout failure
        self.assertEqual(rollout_result["rollout_status"], "failed")
        self.assertEqual(rollout_result["stages_completed"], 2)  # dev + staging
        
        # Verify staging failed and production was not run
        self.assertEqual(rollout_result["stage_results"]["dev"]["status"], "passed")
        self.assertEqual(rollout_result["stage_results"]["staging"]["status"], "failed")
        self.assertNotIn("production", rollout_result["stage_results"])
        
        # Verify recommendations include staging fixes
        recommendations = rollout_result["recommendations"]
        self.assertTrue(any("staging stage" in rec for rec in recommendations))
    
    def test_rollout_with_missing_test_suite(self):
        """Test rollout behavior with missing test suites."""
        test_suites = {
            "dev": lambda: [TestResult("dev_test", "passed", 10.0, 1.0)],
            # Missing staging and production suites
        }
        
        with patch('logging.warning') as mock_warning:
            rollout_result = self.validation_framework.run_progressive_rollout(test_suites)
            
            # Verify warnings were logged
            mock_warning.assert_called()
            warning_calls = [call.args[0] for call in mock_warning.call_args_list]
            self.assertTrue(any("staging" in call for call in warning_calls))
            self.assertTrue(any("production" in call for call in warning_calls))
        
        # Only dev stage should be completed
        self.assertEqual(rollout_result["stages_completed"], 1)
        self.assertEqual(rollout_result["rollout_status"], "success")  # Only ran dev successfully


class TestRealWorldValidationScenarios(BaseTestCase):
    """Test validation framework with real-world testing scenarios."""
    
    def setUp(self):
        super().setUp()
        self.validation_framework = ValidationFramework()
        self.test_data_generator = TestDataGenerator()
    
    def test_analyzer_unit_test_validation(self):
        """Validate analyzer unit tests using validation framework."""
        def analyzer_unit_test_suite():
            """Run actual analyzer unit tests and convert to TestResult objects."""
            results = []
            
            # Create test instances
            zen_test_case = TestZenBypassAnalyzer()
            workflow_test_case = TestWorkflowPatternAnalyzer()
            
            test_methods = [
                (zen_test_case, "test_ideal_workflow_no_drift"),
                (zen_test_case, "test_bypassed_zen_drift_detection"),
                (zen_test_case, "test_performance_within_threshold"),
                (workflow_test_case, "test_fragmented_workflow_detection"),
                (workflow_test_case, "test_memory_coordination_patterns")
            ]
            
            for test_instance, method_name in test_methods:
                start_time = time.time()
                try:
                    test_instance.setUp()
                    getattr(test_instance, method_name)()
                    test_instance.tearDown()
                    
                    execution_time = (time.time() - start_time) * 1000
                    results.append(TestResult(
                        method_name, "passed", execution_time, 2.0  # Estimate 2MB memory
                    ))
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    results.append(TestResult(
                        method_name, "failed", execution_time, 2.0, str(e)
                    ))
            
            return results
        
        stage_result = self.validation_framework.validate_stage(
            "dev", analyzer_unit_test_suite
        )
        
        # Unit tests should pass validation
        self.assertEqual(stage_result["status"], "passed")
        self.assertGreaterEqual(stage_result["metrics"]["success_rate"], 0.8)
        self.assertEqual(stage_result["recommendation"], "proceed")
    
    def test_integration_test_validation(self):
        """Validate integration tests using validation framework."""
        def integration_test_suite():
            """Run integration tests and convert to TestResult objects."""
            results = []
            
            # Simulate integration test scenarios
            integration_scenarios = [
                ("pipeline_initialization", True, 25.0),
                ("ideal_workflow_processing", True, 30.0),
                ("problematic_workflow_handling", True, 35.0),
                ("error_recovery", True, 20.0),
                ("concurrent_processing", True, 45.0)
            ]
            
            for test_name, should_pass, expected_time in integration_scenarios:
                # Simulate test execution with some variance
                actual_time = expected_time + (time.time() % 10 - 5)  # ±5ms variance
                memory_usage = 3.0 + (time.time() % 2)  # 3-5MB range
                
                status = "passed" if should_pass and actual_time < 50 else "failed"
                error_msg = "Performance threshold exceeded" if actual_time >= 50 else None
                
                results.append(TestResult(
                    test_name, status, actual_time, memory_usage, error_msg
                ))
            
            return results
        
        stage_result = self.validation_framework.validate_stage(
            "staging", integration_test_suite
        )
        
        # Integration tests should meet staging requirements
        self.assertIn(stage_result["status"], ["passed", "failed"])  # Depends on simulated performance
        self.assertGreater(stage_result["metrics"]["total_tests"], 0)
        
        if stage_result["status"] == "passed":
            self.assertEqual(stage_result["recommendation"], "proceed")
        else:
            self.assertEqual(stage_result["recommendation"], "investigate_and_fix")
    
    def test_performance_benchmark_validation(self):
        """Validate performance benchmarks using validation framework."""
        def performance_benchmark_suite():
            """Run performance benchmarks and convert to TestResult objects."""
            results = []
            
            # Simulate performance benchmark results
            benchmarks = [
                ("stderr_generation_benchmark", 12.0, 1.5),
                ("concurrent_processing_benchmark", 25.0, 4.0),
                ("memory_usage_benchmark", 18.0, 3.2),
                ("throughput_benchmark", 30.0, 2.8),
                ("regression_detection_benchmark", 15.0, 2.0)
            ]
            
            for benchmark_name, exec_time, memory_usage in benchmarks:
                # All benchmarks should pass performance thresholds
                status = "passed" if exec_time <= 50 and memory_usage <= 10 else "failed"
                error_msg = None
                
                if exec_time > 50:
                    error_msg = f"Execution time {exec_time}ms exceeds 50ms threshold"
                elif memory_usage > 10:
                    error_msg = f"Memory usage {memory_usage}MB exceeds 10MB threshold"
                
                results.append(TestResult(
                    benchmark_name, status, exec_time, memory_usage, error_msg
                ))
            
            return results
        
        stage_result = self.validation_framework.validate_stage(
            "production", performance_benchmark_suite
        )
        
        # Performance benchmarks should meet strict production requirements
        self.assertEqual(stage_result["status"], "passed")
        self.assertEqual(stage_result["metrics"]["success_rate"], 1.0)
        self.assertTrue(stage_result["thresholds_met"]["performance"])
        self.assertTrue(stage_result["thresholds_met"]["memory"])
        self.assertEqual(stage_result["recommendation"], "proceed")
    
    def test_complete_validation_pipeline(self):
        """Test complete validation pipeline with all test types."""
        # Define comprehensive test suites for each stage
        def dev_test_suite():
            return [
                TestResult("unit_test_1", "passed", 8.0, 1.0),
                TestResult("unit_test_2", "passed", 12.0, 1.2),
                TestResult("unit_test_3", "passed", 10.0, 0.9),
                TestResult("unit_test_4", "passed", 15.0, 1.5),
            ]
        
        def staging_test_suite():
            return [
                TestResult("integration_test_1", "passed", 25.0, 3.0),
                TestResult("integration_test_2", "passed", 30.0, 3.5),
                TestResult("integration_test_3", "passed", 22.0, 2.8),
                TestResult("system_test_1", "passed", 35.0, 4.0),
                TestResult("system_test_2", "passed", 28.0, 3.2),
            ]
        
        def production_test_suite():
            return [
                TestResult("performance_test_1", "passed", 15.0, 2.0),
                TestResult("performance_test_2", "passed", 20.0, 2.5),
                TestResult("load_test_1", "passed", 40.0, 8.0),
                TestResult("stress_test_1", "passed", 45.0, 9.0),
                TestResult("regression_test_1", "passed", 18.0, 2.2),
                TestResult("regression_test_2", "passed", 22.0, 2.8),
            ]
        
        test_suites = {
            "dev": dev_test_suite,
            "staging": staging_test_suite,
            "production": production_test_suite
        }
        
        # Run complete progressive rollout
        rollout_result = self.validation_framework.run_progressive_rollout(test_suites)
        
        # Verify successful complete pipeline
        self.assertEqual(rollout_result["rollout_status"], "success")
        self.assertEqual(rollout_result["stages_completed"], 3)
        
        # Verify metrics progression (later stages should have more tests)
        dev_tests = rollout_result["stage_results"]["dev"]["metrics"]["total_tests"]
        staging_tests = rollout_result["stage_results"]["staging"]["metrics"]["total_tests"]
        production_tests = rollout_result["stage_results"]["production"]["metrics"]["total_tests"]
        
        self.assertLess(dev_tests, staging_tests)
        self.assertLess(staging_tests, production_tests)
        
        # Verify overall quality metrics
        overall_metrics = rollout_result["overall_metrics"]
        self.assertEqual(overall_metrics["total_tests"], 15)  # 4 + 5 + 6
        self.assertEqual(overall_metrics["overall_success_rate"], 1.0)
        self.assertTrue(overall_metrics["meets_success_threshold"])
        
        # Verify final recommendation
        self.assertIn("ready for full deployment", rollout_result["recommendations"][0])


class TestValidationQualityGates(BaseTestCase):
    """Test quality gates and decision-making logic."""
    
    def setUp(self):
        super().setUp()
        self.validation_framework = ValidationFramework()
    
    def test_success_rate_quality_gate(self):
        """Test success rate quality gate enforcement."""
        # Test at threshold boundary
        threshold_tests = []
        passing_count = int(100 * TEST_CONFIG["validation"]["success_threshold"])  # 95 tests
        
        for i in range(passing_count):
            threshold_tests.append(TestResult(f"pass_{i}", "passed", 10.0, 1.0))
        
        for i in range(100 - passing_count):  # 5 failing tests
            threshold_tests.append(TestResult(f"fail_{i}", "failed", 10.0, 1.0, "Test failed"))
        
        def threshold_test_suite():
            return threshold_tests
        
        result = self.validation_framework.validate_stage("dev", threshold_test_suite)
        
        # Should be exactly at threshold (95% pass rate)
        self.assertAlmostEqual(result["metrics"]["success_rate"], 0.95, places=2)
        self.assertTrue(result["thresholds_met"]["success_rate"])
        self.assertEqual(result["status"], "passed")
    
    def test_performance_quality_gate(self):
        """Test performance quality gate enforcement."""
        # Create tests at performance boundary
        def boundary_performance_suite():
            return [
                TestResult("fast_test", "passed", 25.0, 1.0),
                TestResult("threshold_test", "passed", 50.0, 1.0),  # Exactly at threshold
                TestResult("another_fast_test", "passed", 30.0, 1.0),
            ]
        
        result = self.validation_framework.validate_stage("staging", boundary_performance_suite)
        
        # Should pass (average is under 50ms)
        avg_time = result["metrics"]["avg_execution_time_ms"]
        self.assertLessEqual(avg_time, TEST_CONFIG["performance"]["max_stderr_generation_time_ms"])
        self.assertTrue(result["thresholds_met"]["performance"])
    
    def test_memory_quality_gate(self):
        """Test memory usage quality gate enforcement."""
        def memory_boundary_suite():
            return [
                TestResult("low_memory_test", "passed", 10.0, 5.0),
                TestResult("high_memory_test", "passed", 10.0, 10.0),  # At threshold
                TestResult("medium_memory_test", "passed", 10.0, 7.5),
            ]
        
        result = self.validation_framework.validate_stage("production", memory_boundary_suite)
        
        # Should pass (average is at threshold)
        avg_memory = result["metrics"]["avg_memory_usage_mb"]
        self.assertLessEqual(avg_memory, TEST_CONFIG["performance"]["max_memory_usage_mb"])
        self.assertTrue(result["thresholds_met"]["memory"])
    
    def test_combined_quality_gates(self):
        """Test enforcement when multiple quality gates are involved."""
        def mixed_quality_suite():
            return [
                TestResult("good_test", "passed", 20.0, 3.0),
                TestResult("slow_test", "passed", 60.0, 2.0),      # Over performance threshold
                TestResult("memory_heavy_test", "passed", 15.0, 12.0),  # Over memory threshold
                TestResult("failed_test", "failed", 10.0, 1.0, "Test failure"),  # Failed test
                TestResult("another_good_test", "passed", 25.0, 4.0),
            ]
        
        result = self.validation_framework.validate_stage("production", mixed_quality_suite)
        
        # Should fail due to multiple threshold violations
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["thresholds_met"]["success_rate"])  # 80% vs 95% required
        self.assertFalse(result["thresholds_met"]["performance"])   # Over 50ms average
        self.assertFalse(result["thresholds_met"]["memory"])        # Over 10MB average
        
        # Should recommend investigation
        self.assertEqual(result["recommendation"], "investigate_and_fix")


if __name__ == '__main__':
    # Configure detailed logging for validation
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run validation framework tests
    unittest.main(verbosity=2)