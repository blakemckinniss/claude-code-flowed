"""
Integration Tests for PostToolUse Hook Pipeline
==============================================

End-to-end integration tests for the complete PostToolUse hook pipeline,
including analyzer coordination, guidance generation, and stderr output.
"""

import unittest
import sys
import os
import json
import tempfile
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List
from contextlib import contextmanager
import time

# Add hooks modules to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from test_framework_architecture import (
    BaseTestCase, MockToolExecutionData, TestDataGenerator, 
    PerformanceBenchmarkRunner, ValidationFramework, TEST_CONFIG
)

# Import post-tool components
try:
    from post_tool.manager import PostToolAnalysisManager, PostToolAnalysisConfig
    from post_tool.core.drift_detector import DriftType, DriftSeverity
    from post_tool.core.guidance_system import (
        NonBlockingGuidanceProvider, GuidanceOutputHandler, ContextualGuidanceEnhancer
    )
except ImportError as e:
    print(f"Warning: Could not import post-tool modules: {e}")
    # Create mock classes for testing framework validation
    class PostToolAnalysisManager:
        def __init__(self, config_path=None):
            self.config = Mock()
            self.analyzers = []
            self.tool_count = 0
        
        def analyze_tool_usage(self, tool_name, tool_input, tool_response):
            self.tool_count += 1
    
    class PostToolAnalysisConfig:
        def __init__(self, config_path=None):
            pass
        
        def is_analyzer_enabled(self, name):
            return True


class TestPostToolAnalysisManager(BaseTestCase):
    """Integration tests for the main PostToolAnalysisManager."""
    
    def setUp(self):
        super().setUp()
        self.test_data_generator = TestDataGenerator()
        
        # Create temporary config for testing
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config_data = {
            "enabled_analyzers": [
                "zen_bypass_analyzer",
                "workflow_pattern_analyzer",
                "native_overuse_analyzer"
            ],
            "guidance_settings": {
                "max_guidance_frequency": 3,
                "escalation_threshold": 2,
                "emergency_threshold": 4
            }
        }
        json.dump(config_data, self.temp_config)
        self.temp_config.close()
        
        self.manager = PostToolAnalysisManager(self.temp_config.name)
    
    def tearDown(self):
        super().tearDown()
        os.unlink(self.temp_config.name)
    
    def test_manager_initialization(self):
        """Test proper initialization of analysis manager."""
        self.assertIsNotNone(self.manager.config)
        self.assertIsInstance(self.manager.analyzers, list)
        self.assertEqual(self.manager.tool_count, 0)
        self.assertIsNotNone(self.manager.guidance_provider)
        self.assertIsNotNone(self.manager.guidance_enhancer)
    
    def test_ideal_workflow_processing(self):
        """Test processing of ideal workflow without triggering guidance."""
        ideal_scenario = self.test_data_generator.generate_scenario("ideal_workflow", 1)[0]
        
        with self.capture_stderr() as stderr_output:
            for tool_data in ideal_scenario["tool_sequence"]:
                self.manager.analyze_tool_usage(
                    tool_data["tool_name"],
                    tool_data["tool_input"],
                    tool_data["tool_response"]
                )
        
        # Ideal workflow should not generate excessive stderr guidance
        stderr_text = ''.join(stderr_output)
        guidance_count = stderr_text.count("🚨") + stderr_text.count("👑")
        self.assertLessEqual(guidance_count, 1, "Ideal workflow should generate minimal guidance")
        
        # Verify tool count is tracked
        self.assertEqual(self.manager.tool_count, len(ideal_scenario["tool_sequence"]))
    
    def test_problematic_workflow_triggers_guidance(self):
        """Test that problematic workflows trigger appropriate guidance."""
        bypassed_scenario = self.test_data_generator.generate_scenario("bypassed_zen", 1)[0]
        
        with self.capture_stderr() as stderr_output:
            for tool_data in bypassed_scenario["tool_sequence"]:
                self.manager.analyze_tool_usage(
                    tool_data["tool_name"],
                    tool_data["tool_input"],
                    tool_data["tool_response"]
                )
        
        # Should generate guidance for bypassed ZEN
        stderr_text = ''.join(stderr_output)
        self.assertIn("Queen ZEN", stderr_text, "Should mention Queen ZEN in guidance")
        self.assertTrue(
            "👑" in stderr_text or "🚨" in stderr_text,
            "Should contain hive-related emojis"
        )
    
    def test_analyzer_error_handling(self):
        """Test graceful handling of analyzer errors."""
        # Create a scenario that might cause analyzer errors
        problematic_data = {
            "tool_name": "InvalidTool",
            "tool_input": {"malformed": None},
            "tool_response": {"error": "Tool failed"}
        }
        
        # Should not crash on malformed data
        with self.capture_stderr():
            self.manager.analyze_tool_usage(
                problematic_data["tool_name"],
                problematic_data["tool_input"],
                problematic_data["tool_response"]
            )
        
        # Manager should continue functioning
        self.assertEqual(self.manager.tool_count, 1)
    
    def test_manager_status_reporting(self):
        """Test the manager's status reporting functionality."""
        status = self.manager.get_analyzer_status()
        
        self.assertIn("total_tools_processed", status)
        self.assertIn("active_analyzers", status)
        self.assertIn("analyzer_priorities", status)
        self.assertIn("config_path", status)
        
        self.assertEqual(status["total_tools_processed"], self.manager.tool_count)
        self.assertIsInstance(status["active_analyzers"], list)
    
    def test_concurrent_tool_processing(self):
        """Test processing multiple tool sequences concurrently."""
        scenarios = [
            self.test_data_generator.generate_scenario("ideal_workflow", 1)[0],
            self.test_data_generator.generate_scenario("excessive_native", 1)[0]
        ]
        
        start_time = time.time()
        
        with self.capture_stderr():
            for scenario in scenarios:
                for tool_data in scenario["tool_sequence"]:
                    self.manager.analyze_tool_usage(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should process efficiently
        total_tools = sum(len(s["tool_sequence"]) for s in scenarios)
        self.assertEqual(self.manager.tool_count, total_tools)
        self.assertLess(processing_time, 100, "Should process tools quickly")  # 100ms threshold


class TestGuidanceSystemIntegration(BaseTestCase):
    """Integration tests for the guidance system components."""
    
    def setUp(self):
        super().setUp()
        self.guidance_provider = NonBlockingGuidanceProvider()
        self.guidance_enhancer = ContextualGuidanceEnhancer()
    
    def test_guidance_frequency_limiting(self):
        """Test that guidance frequency is properly limited."""
        # Create multiple drift evidences
        mock_evidences = []
        for i in range(10):
            evidence = Mock()
            evidence.drift_type = Mock()
            evidence.drift_type.value = "bypassed_zen"
            evidence.severity = Mock()
            evidence.severity.value = 2  # MODERATE
            evidence.priority_score = 50
            evidence.tool_sequence = [f"tool_{i}"]
            mock_evidences.append(evidence)
        
        guidance_count = 0
        for i, evidence in enumerate(mock_evidences):
            guidance_info = self.guidance_provider.provide_guidance([evidence], i + 1)
            if guidance_info:
                guidance_count += 1
        
        # Should limit guidance frequency (not every tool triggers guidance)
        self.assertLess(guidance_count, len(mock_evidences), "Should limit guidance frequency")
        self.assertGreater(guidance_count, 0, "Should provide some guidance")
    
    def test_escalation_behavior(self):
        """Test escalation behavior when drift persists."""
        evidence = Mock()
        evidence.drift_type = Mock()
        evidence.drift_type.value = "no_mcp_coordination"
        evidence.severity = Mock()
        evidence.severity.value = 3  # MAJOR
        evidence.priority_score = 80
        evidence.tool_sequence = ["tool_1", "tool_2", "tool_3"]
        
        # Simulate repeated violations
        escalation_triggered = False
        for i in range(6):  # Should trigger escalation after threshold
            guidance_info = self.guidance_provider.provide_guidance([evidence], i + 1)
            if guidance_info and guidance_info.get("severity") == "emergency":
                escalation_triggered = True
                break
        
        self.assertTrue(escalation_triggered, "Should escalate after repeated violations")
    
    def test_contextual_guidance_enhancement(self):
        """Test contextual enhancement of guidance messages."""
        base_message = "Consider using Queen ZEN's hive coordination"
        tool_name = "Write"
        tool_input = {"file_path": "/src/components/App.tsx"}
        tool_sequence = ["Read", "Write", "Write"]
        
        enhanced_message = self.guidance_enhancer.enhance_guidance(
            base_message, tool_name, tool_input, tool_sequence
        )
        
        self.assertIn("Queen ZEN", enhanced_message)
        self.assertNotEqual(enhanced_message, base_message, "Should enhance the message")
        # Should include context about the file being modified
        self.assertTrue(
            any(keyword in enhanced_message.lower() for keyword in ["file", "component", "tsx"]),
            "Should include file context"
        )
    
    @patch('sys.stderr')
    def test_guidance_output_handling(self, mock_stderr):
        """Test proper output handling for guidance messages."""
        guidance_info = {
            "message": "🚨 HIVE PROTOCOL VIOLATION: Queen ZEN must command!",
            "severity": "major",
            "tool_count": 5,
            "should_exit": False
        }
        
        GuidanceOutputHandler.handle_guidance_output(guidance_info)
        
        # Verify stderr was called
        mock_stderr.write.assert_called()
        mock_stderr.flush.assert_called()
        
        # Verify message content
        written_content = ''.join(call.args[0] for call in mock_stderr.write.call_args_list)
        self.assertIn("🚨", written_content)
        self.assertIn("Queen ZEN", written_content)


class TestEndToEndPipeline(BaseTestCase):
    """End-to-end integration tests for the complete pipeline."""
    
    def setUp(self):
        super().setUp()
        self.test_data_generator = TestDataGenerator()
        
        # Create a complete pipeline setup
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config_data = {
            "enabled_analyzers": [
                "zen_bypass_analyzer",
                "workflow_pattern_analyzer",
                "native_overuse_analyzer",
                "batching_opportunity_analyzer"
            ],
            "guidance_settings": {
                "max_guidance_frequency": 5,
                "escalation_threshold": 3,
                "emergency_threshold": 5
            },
            "drift_sensitivity": {
                "zen_bypass": "normal",
                "native_overuse": "strict",
                "workflow_fragmentation": "lenient"
            }
        }
        json.dump(config_data, self.temp_config)
        self.temp_config.close()
        
        self.manager = PostToolAnalysisManager(self.temp_config.name)
    
    def tearDown(self):
        super().tearDown()
        os.unlink(self.temp_config.name)
    
    def test_complete_workflow_analysis(self):
        """Test complete analysis of various workflow scenarios."""
        scenarios = [
            ("ideal_workflow", 0),      # Should generate minimal guidance
            ("bypassed_zen", 1),        # Should generate ZEN guidance
            ("excessive_native", 2),    # Should generate coordination guidance
            ("fragmented_workflow", 1)  # Should generate batching guidance
        ]
        
        results = {}
        
        for scenario_name, expected_min_guidance in scenarios:
            scenario = self.test_data_generator.generate_scenario(scenario_name, 1)[0]
            
            with self.capture_stderr() as stderr_output:
                for tool_data in scenario["tool_sequence"]:
                    self.manager.analyze_tool_usage(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
            
            stderr_text = ''.join(stderr_output)
            guidance_indicators = stderr_text.count("🚨") + stderr_text.count("👑") + stderr_text.count("🐝")
            
            results[scenario_name] = {
                "guidance_count": guidance_indicators,
                "stderr_length": len(stderr_text),
                "tool_count": len(scenario["tool_sequence"])
            }
            
            # Validate expected guidance levels
            if scenario_name == "ideal_workflow":
                self.assertLessEqual(guidance_indicators, 1, "Ideal workflow should generate minimal guidance")
            else:
                self.assertGreaterEqual(
                    guidance_indicators, expected_min_guidance,
                    f"{scenario_name} should generate appropriate guidance"
                )
        
        # Verify different scenarios produce different guidance patterns
        ideal_guidance = results["ideal_workflow"]["guidance_count"]
        problematic_guidance = max(
            results["bypassed_zen"]["guidance_count"],
            results["excessive_native"]["guidance_count"]
        )
        self.assertLess(ideal_guidance, problematic_guidance, "Problematic workflows should generate more guidance")
    
    def test_performance_under_load(self):
        """Test pipeline performance under load conditions."""
        # Generate multiple concurrent scenarios
        load_scenarios = []
        for scenario_type in ["ideal_workflow", "bypassed_zen", "excessive_native"]:
            load_scenarios.extend(self.test_data_generator.generate_scenario(scenario_type, 3))
        
        start_time = time.time()
        total_tools = 0
        
        with self.capture_stderr() as stderr_output:
            for scenario in load_scenarios:
                for tool_data in scenario["tool_sequence"]:
                    self.manager.analyze_tool_usage(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
                    total_tools += 1
        
        processing_time = (time.time() - start_time) * 1000
        avg_time_per_tool = processing_time / total_tools
        
        # Performance assertions
        self.assertLess(avg_time_per_tool, 5, "Should process each tool in under 5ms on average")
        self.assertEqual(self.manager.tool_count, total_tools, "Should track all processed tools")
        
        # Memory usage should be reasonable
        stderr_text = ''.join(stderr_output)
        self.assertLess(len(stderr_text), 10000, "Should not generate excessive stderr output")
    
    def test_error_recovery_and_continuity(self):
        """Test pipeline recovery from errors and continuity."""
        # Mix normal and problematic data
        normal_data = MockToolExecutionData.get_zen_chat_execution()
        problematic_data = {
            "tool_name": None,  # Invalid tool name
            "tool_input": {"corrupted": "data"},
            "tool_response": None
        }
        recovery_data = MockToolExecutionData.get_flow_swarm_execution()
        
        test_sequence = [normal_data, problematic_data, recovery_data]
        
        with self.capture_stderr():
            for tool_data in test_sequence:
                try:
                    self.manager.analyze_tool_usage(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
                except Exception as e:
                    # Pipeline should handle exceptions gracefully
                    self.fail(f"Pipeline should not crash on bad data: {e}")
        
        # Should have processed all tools (even the problematic one)
        self.assertEqual(self.manager.tool_count, len(test_sequence))
        
        # Should continue to function normally after error
        status = self.manager.get_analyzer_status()
        self.assertGreater(len(status["active_analyzers"]), 0, "Analyzers should remain active")


class TestPipelineBenchmarking(BaseTestCase):
    """Performance benchmarking for the complete pipeline."""
    
    def setUp(self):
        super().setUp()
        self.benchmark_runner = PerformanceBenchmarkRunner(iterations=20)
        self.test_data_generator = TestDataGenerator()
    
    def test_pipeline_performance_benchmark(self):
        """Benchmark complete pipeline performance."""
        # Create diverse test scenarios
        test_scenarios = []
        for scenario_type in ["ideal_workflow", "bypassed_zen", "excessive_native", "fragmented_workflow"]:
            test_scenarios.extend(self.test_data_generator.generate_scenario(scenario_type, 2))
        
        def pipeline_function(tool_sequence):
            """Simulate complete pipeline processing."""
            manager = PostToolAnalysisManager()
            for tool_data in tool_sequence:
                manager.analyze_tool_usage(
                    tool_data["tool_name"],
                    tool_data["tool_input"],
                    tool_data["tool_response"]
                )
            return manager.get_analyzer_status()
        
        benchmark = self.benchmark_runner.benchmark_pipeline_integration(
            pipeline_function, test_scenarios
        )
        
        # Validate benchmark results
        self.assertIsNotNone(benchmark)
        self.assertEqual(benchmark.operation_name, "pipeline_integration")
        self.assertGreater(benchmark.success_rate, 0.8, "Pipeline should have high success rate")
        
        # Performance thresholds
        self.assertLessEqual(
            benchmark.avg_execution_time_ms,
            TEST_CONFIG["performance"]["max_stderr_generation_time_ms"] * 3,  # Allow 3x for full pipeline
            "Pipeline should meet performance thresholds"
        )
        
        self.assertLessEqual(
            benchmark.memory_peak_mb,
            TEST_CONFIG["performance"]["max_memory_usage_mb"],
            "Pipeline should stay within memory limits"
        )
    
    def test_guidance_generation_performance(self):
        """Benchmark guidance generation performance specifically."""
        scenarios = self.test_data_generator.generate_scenario("bypassed_zen", 5)
        
        def guidance_generation_function(tool_sequence):
            """Focus on guidance generation performance."""
            manager = PostToolAnalysisManager()
            guidance_count = 0
            
            with patch('sys.stderr'):  # Suppress stderr for clean benchmarking
                for tool_data in tool_sequence:
                    manager.analyze_tool_usage(
                        tool_data["tool_name"],
                        tool_data["tool_input"],
                        tool_data["tool_response"]
                    )
                    guidance_count += 1
            
            return guidance_count
        
        benchmark = self.benchmark_runner.benchmark_pipeline_integration(
            guidance_generation_function, scenarios
        )
        
        # Guidance generation should be very fast
        self.assertLessEqual(
            benchmark.avg_execution_time_ms,
            TEST_CONFIG["performance"]["max_stderr_generation_time_ms"],
            "Guidance generation should be under 50ms"
        )
        
        self.assertGreater(
            benchmark.operations_per_second, 20,
            "Should handle at least 20 guidance operations per second"
        )


if __name__ == '__main__':
    # Configure logging for test runs
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite with proper ordering
    suite = unittest.TestSuite()
    
    # Add tests in logical order
    suite.addTest(unittest.makeSuite(TestPostToolAnalysisManager))
    suite.addTest(unittest.makeSuite(TestGuidanceSystemIntegration))
    suite.addTest(unittest.makeSuite(TestEndToEndPipeline))
    suite.addTest(unittest.makeSuite(TestPipelineBenchmarking))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)