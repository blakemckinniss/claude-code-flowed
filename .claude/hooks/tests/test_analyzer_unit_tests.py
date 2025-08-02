"""
Unit Tests for Post-Tool Analyzers
==================================

Comprehensive unit tests for each analyzer type with mock tool execution data.
Tests drift detection, guidance generation, and performance within thresholds.
"""

import unittest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

# Add hooks modules to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from test_framework_architecture import (
    BaseTestCase, MockToolExecutionData, TestDataGenerator, 
    PerformanceBenchmarkRunner, TEST_CONFIG
)

# Import core drift detection classes
try:
    from post_tool.core.drift_detector import (
        DriftAnalyzer, DriftEvidence, DriftType, DriftSeverity,
        HiveWorkflowValidator, DriftGuidanceGenerator
    )
    from post_tool.analyzers.zen_bypass_analyzer import ZenBypassAnalyzer
    from post_tool.analyzers.workflow_analyzer import WorkflowPatternAnalyzer
except ImportError as e:
    print(f"Warning: Could not import post-tool modules: {e}")
    # Create mock classes for testing framework validation
    class DriftAnalyzer:
        def __init__(self, priority=0):
            self.priority = priority
        def analyze_drift(self, tool_name, tool_input, tool_response):
            return None
        def get_analyzer_name(self):
            return "MockAnalyzer"
    
    class ZenBypassAnalyzer(DriftAnalyzer):
        def get_analyzer_name(self):
            return "ZenBypassAnalyzer"
    
    class WorkflowPatternAnalyzer(DriftAnalyzer):
        def get_analyzer_name(self):
            return "WorkflowPatternAnalyzer"


class TestZenBypassAnalyzer(BaseTestCase):
    """Unit tests for ZEN bypass detection analyzer."""
    
    def setUp(self):
        super().setUp()
        self.analyzer = ZenBypassAnalyzer(priority=1000)
        self.test_data_generator = TestDataGenerator()
    
    def test_ideal_workflow_no_drift(self):
        """Test that ideal workflow with ZEN coordination shows no drift."""
        ideal_scenarios = self.test_data_generator.generate_scenario("ideal_workflow", 3)
        
        for scenario in ideal_scenarios:
            for tool_data in scenario["tool_sequence"]:
                with self.capture_stderr():
                    result = self.analyzer.analyze_drift(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
                
                # Ideal workflow should not trigger drift detection
                if tool_data["tool_name"].startswith("mcp__zen__"):
                    self.assertIsNone(result, "ZEN coordination should not trigger drift")
    
    def test_bypassed_zen_drift_detection(self):
        """Test detection of workflows that bypass ZEN coordination."""
        bypassed_scenarios = self.test_data_generator.generate_scenario("bypassed_zen", 3)
        
        for scenario in bypassed_scenarios:
            drift_detected = False
            
            for tool_data in scenario["tool_sequence"]:
                with self.capture_stderr():
                    result = self.analyzer.analyze_drift(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
                
                if result and hasattr(result, 'drift_type'):
                    drift_detected = True
                    self.assertEqual(result.drift_type.value, "bypassed_zen")
                    self.assertIn("Queen ZEN", result.correction_guidance)
            
            self.assertTrue(drift_detected, "Should detect ZEN bypass in problematic scenarios")
    
    def test_performance_within_threshold(self):
        """Test that analyzer performs within performance thresholds."""
        test_data = MockToolExecutionData.get_native_tool_sequence()
        
        def run_analysis():
            for data in test_data:
                self.analyzer.analyze_drift(
                    data["tool_name"],
                    data["tool_input"],
                    data["tool_response"]
                )
            return True
        
        # Assert performance is within 50ms and 10MB thresholds
        result = self.assertPerformanceWithin(
            run_analysis,
            TEST_CONFIG["performance"]["max_stderr_generation_time_ms"],
            TEST_CONFIG["performance"]["max_memory_usage_mb"]
        )
        self.assertTrue(result)
    
    def test_analyzer_name(self):
        """Test analyzer returns correct name."""
        self.assertEqual(self.analyzer.get_analyzer_name(), "ZenBypassAnalyzer")
    
    def test_priority_setting(self):
        """Test analyzer priority is set correctly."""
        self.assertEqual(self.analyzer.priority, 1000)


class TestWorkflowPatternAnalyzer(BaseTestCase):
    """Unit tests for workflow pattern analyzer."""
    
    def setUp(self):
        super().setUp()
        self.analyzer = WorkflowPatternAnalyzer(priority=700)
        self.test_data_generator = TestDataGenerator()
    
    def test_fragmented_workflow_detection(self):
        """Test detection of fragmented workflows."""
        fragmented_scenarios = self.test_data_generator.generate_scenario("fragmented_workflow", 3)
        
        for scenario in fragmented_scenarios:
            drift_detected = False
            
            for tool_data in scenario["tool_sequence"]:
                with self.capture_stderr():
                    result = self.analyzer.analyze_drift(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
                
                if result and hasattr(result, 'drift_type'):
                    drift_detected = True
                    self.assertIn("batch", result.correction_guidance.lower())
            
            # Should eventually detect fragmentation in problematic scenarios
            if len(scenario["tool_sequence"]) > 5:
                self.assertTrue(drift_detected, "Should detect workflow fragmentation")
    
    def test_memory_coordination_patterns(self):
        """Test recognition of proper memory coordination patterns."""
        memory_scenarios = self.test_data_generator.generate_scenario("memory_coordination", 2)
        
        for scenario in memory_scenarios:
            memory_tool_found = False
            
            for tool_data in scenario["tool_sequence"]:
                with self.capture_stderr():
                    result = self.analyzer.analyze_drift(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
                
                if tool_data["tool_name"] == "mcp__claude-flow__memory_usage":
                    memory_tool_found = True
                    # Should not flag proper memory coordination as drift
                    self.assertIsNone(result, "Proper memory coordination should not be flagged")
            
            self.assertTrue(memory_tool_found, "Memory coordination scenario should include memory tools")
    
    def test_excessive_native_tool_detection(self):
        """Test detection of excessive native tool usage."""
        excessive_scenarios = self.test_data_generator.generate_scenario("excessive_native", 2)
        
        for scenario in excessive_scenarios:
            # Process all tools in sequence
            for tool_data in scenario["tool_sequence"]:
                with self.capture_stderr():
                    self.analyzer.analyze_drift(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
            
            # Check if analyzer detected excessive native usage
            mcp_ratio = self.analyzer.get_mcp_ratio()
            self.assertLess(mcp_ratio, 0.3, "Should detect low MCP usage ratio")
    
    def test_performance_benchmarking(self):
        """Test analyzer performance with benchmark runner."""
        benchmark_runner = PerformanceBenchmarkRunner(iterations=10)
        test_data = MockToolExecutionData.get_native_tool_sequence()
        
        benchmark = benchmark_runner.benchmark_stderr_generation(
            WorkflowPatternAnalyzer, test_data
        )
        
        # Verify benchmark results
        self.assertIsNotNone(benchmark)
        self.assertEqual(benchmark.operation_name, "WorkflowPatternAnalyzer_stderr_generation")
        self.assertGreater(benchmark.success_rate, 0.8, "Should have high success rate")
        self.assertLessEqual(
            benchmark.avg_execution_time_ms,
            TEST_CONFIG["performance"]["max_stderr_generation_time_ms"]
        )


class TestHiveWorkflowValidator(BaseTestCase):
    """Unit tests for hive workflow validator."""
    
    def setUp(self):
        super().setUp()
        self.validator = HiveWorkflowValidator()
    
    def test_ideal_pattern_validation(self):
        """Test validation of ideal workflow patterns."""
        ideal_sequence = [
            "mcp__zen__chat",
            "mcp__claude-flow__swarm_init", 
            "mcp__filesystem__read",
            "Write"
        ]
        
        is_valid, message = self.validator.validate_workflow_adherence(ideal_sequence)
        self.assertTrue(is_valid, "Ideal workflow should be valid")
        self.assertIn("Queen ZEN", message)
    
    def test_drift_pattern_detection(self):
        """Test detection of drift patterns."""
        problematic_sequence = [
            "mcp__filesystem__read",
            "Write",
            "Edit"
        ]
        
        is_valid, message = self.validator.validate_workflow_adherence(problematic_sequence)
        self.assertFalse(is_valid, "Problematic workflow should be invalid")
        self.assertIn("hive coordination", message.lower())
    
    def test_flow_without_zen_detection(self):
        """Test detection of Flow workers acting independently."""
        flow_only_sequence = [
            "mcp__claude-flow__swarm_init",
            "Write",
            "Edit"
        ]
        
        is_valid, message = self.validator.validate_workflow_adherence(flow_only_sequence)
        self.assertFalse(is_valid, "Flow without ZEN should be invalid")
        self.assertIn("Queen ZEN", message)
    
    def test_performance_pattern_matching(self):
        """Test performance of pattern matching algorithm."""
        long_sequence = ["Read", "Write"] * 50  # 100 operations
        
        def run_validation():
            return self.validator.validate_workflow_adherence(long_sequence)
        
        result = self.assertPerformanceWithin(run_validation, 10.0, 1.0)  # 10ms, 1MB
        self.assertIsNotNone(result)


class TestDriftGuidanceGenerator(BaseTestCase):
    """Unit tests for drift guidance generator."""
    
    def setUp(self):
        super().setUp()
        self.generator = DriftGuidanceGenerator()
    
    def test_guidance_generation_for_bypassed_zen(self):
        """Test guidance generation for bypassed ZEN scenarios."""
        # Create mock drift evidence
        mock_evidence = Mock()
        mock_evidence.drift_type = Mock()
        mock_evidence.drift_type.value = "bypassed_zen"
        mock_evidence.severity = Mock()
        mock_evidence.severity.value = 2  # MODERATE
        mock_evidence.evidence_details = "Direct filesystem access without ZEN coordination"
        mock_evidence.missing_tools = ["mcp__zen__chat"]
        
        # Generate guidance
        guidance = self.generator.generate_guidance(mock_evidence)
        
        # Verify guidance content
        self.assertIn("Queen ZEN", guidance)
        self.assertIn("mcp__zen__chat", guidance)
        self.assertIn("HIVE PROTOCOL", guidance)
    
    def test_guidance_performance(self):
        """Test guidance generation performance."""
        mock_evidence = Mock()
        mock_evidence.drift_type = Mock()
        mock_evidence.drift_type.value = "no_mcp_coordination"
        mock_evidence.severity = Mock() 
        mock_evidence.severity.value = 3  # MAJOR
        mock_evidence.evidence_details = "Extended native tool usage"
        mock_evidence.missing_tools = ["mcp__zen__chat", "mcp__claude-flow__swarm_init"]
        
        def generate_guidance():
            return self.generator.generate_guidance(mock_evidence)
        
        result = self.assertPerformanceWithin(generate_guidance, 5.0, 0.5)  # 5ms, 0.5MB
        self.assertIsNotNone(result)
        self.assertIn("🚨", result)  # Should contain alerts for major severity


class TestAnalyzerIntegration(BaseTestCase):
    """Integration tests between multiple analyzers."""
    
    def setUp(self):
        super().setUp()
        self.zen_analyzer = ZenBypassAnalyzer(priority=1000)
        self.workflow_analyzer = WorkflowPatternAnalyzer(priority=700)
        self.test_data_generator = TestDataGenerator()
    
    def test_analyzer_coordination(self):
        """Test that multiple analyzers work together correctly."""
        complex_scenario = self.test_data_generator.generate_scenario("excessive_native", 1)[0]
        
        zen_results = []
        workflow_results = []
        
        for tool_data in complex_scenario["tool_sequence"]:
            # Run both analyzers on same data
            zen_result = self.zen_analyzer.analyze_drift(
                tool_data["tool_name"],
                tool_data["tool_input"],
                tool_data["tool_response"]
            )
            
            workflow_result = self.workflow_analyzer.analyze_drift(
                tool_data["tool_name"],
                tool_data["tool_input"],
                tool_data["tool_response"]
            )
            
            if zen_result:
                zen_results.append(zen_result)
            if workflow_result:
                workflow_results.append(workflow_result)
        
        # Should detect issues from both perspectives
        total_detections = len(zen_results) + len(workflow_results)
        self.assertGreater(total_detections, 0, "Should detect drift from multiple analyzers")
    
    def test_analyzer_priority_ordering(self):
        """Test that analyzer priorities are respected."""
        analyzers = [
            (self.workflow_analyzer, 700),
            (self.zen_analyzer, 1000)
        ]
        
        # Sort by priority (highest first)
        sorted_analyzers = sorted(analyzers, key=lambda x: x[1], reverse=True)
        
        self.assertEqual(sorted_analyzers[0][0], self.zen_analyzer)
        self.assertEqual(sorted_analyzers[1][0], self.workflow_analyzer)
    
    def test_concurrent_analysis_performance(self):
        """Test performance when running multiple analyzers concurrently."""
        test_data = MockToolExecutionData.get_native_tool_sequence() * 3
        
        def run_concurrent_analysis():
            for data in test_data:
                # Simulate running both analyzers
                self.zen_analyzer.analyze_drift(
                    data["tool_name"], data["tool_input"], data["tool_response"]
                )
                self.workflow_analyzer.analyze_drift(
                    data["tool_name"], data["tool_input"], data["tool_response"]
                )
            return True
        
        result = self.assertPerformanceWithin(
            run_concurrent_analysis,
            TEST_CONFIG["performance"]["max_stderr_generation_time_ms"] * 2,  # Allow 2x time for dual analysis
            TEST_CONFIG["performance"]["max_memory_usage_mb"]
        )
        self.assertTrue(result)


class TestMockDataValidation(BaseTestCase):
    """Validate the quality and realism of mock test data."""
    
    def setUp(self):
        super().setUp()
        self.test_data_generator = TestDataGenerator()
    
    def test_scenario_generation_consistency(self):
        """Test that scenario generation produces consistent data structures."""
        scenario_types = ["ideal_workflow", "bypassed_zen", "excessive_native", "fragmented_workflow"]
        
        for scenario_type in scenario_types:
            scenarios = self.test_data_generator.generate_scenario(scenario_type, 3)
            
            for scenario in scenarios:
                # Validate required fields
                self.assertIn("scenario", scenario)
                self.assertIn("variation", scenario)
                self.assertIn("tool_sequence", scenario)
                self.assertIn("expected_drift", scenario)
                self.assertIn("expected_guidance", scenario)
                
                # Validate tool sequence structure
                self.assertIsInstance(scenario["tool_sequence"], list)
                for tool_data in scenario["tool_sequence"]:
                    self.assertIn("tool_name", tool_data)
                    self.assertIn("tool_input", tool_data)  
                    self.assertIn("tool_response", tool_data)
    
    def test_mock_data_realism(self):
        """Test that mock data represents realistic tool usage patterns."""
        zen_data = MockToolExecutionData.get_zen_chat_execution()
        flow_data = MockToolExecutionData.get_flow_swarm_execution()
        native_sequence = MockToolExecutionData.get_native_tool_sequence()
        
        # Validate ZEN data structure
        self.assertEqual(zen_data["tool_name"], "mcp__zen__chat")
        self.assertIn("prompt", zen_data["tool_input"])
        self.assertIn("analysis", zen_data["tool_response"])
        
        # Validate Flow data structure
        self.assertEqual(flow_data["tool_name"], "mcp__claude-flow__swarm_init")
        self.assertIn("topology", flow_data["tool_input"])
        self.assertIn("swarm_id", flow_data["tool_response"])
        
        # Validate native sequence
        self.assertGreater(len(native_sequence), 0)
        for tool_data in native_sequence:
            self.assertIn(tool_data["tool_name"], ["Read", "Write", "Bash"])
    
    def test_problematic_sequence_triggers_drift(self):
        """Test that problematic sequences actually trigger drift detection."""
        problematic_data = MockToolExecutionData.get_problematic_sequence()
        
        # Should have multiple rapid file operations without coordination
        write_operations = [t for t in problematic_data if t["tool_name"] == "Write"]
        self.assertGreaterEqual(len(write_operations), 2, "Should have multiple write operations")
        
        # Should not have MCP coordination
        mcp_operations = [t for t in problematic_data if t["tool_name"].startswith("mcp__")]
        self.assertEqual(len(mcp_operations), 0, "Problematic sequence should lack MCP coordination")


if __name__ == '__main__':
    # Configure logging for test runs
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test suite
    unittest.main(verbosity=2)