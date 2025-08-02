#!/usr/bin/env python3
"""Comprehensive testing framework for tool pattern analyzers.

Tests the tool pattern analysis system including:
- Pattern detection accuracy
- Feedback generation quality  
- Performance benchmarks
- Integration with PostToolUse hook
- Progressive verbosity adaptation
"""

import json
import os
import sys
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import shutil

# Set up test environment
test_dir = os.path.dirname(os.path.abspath(__file__))
hooks_dir = os.path.dirname(test_dir)
sys.path.insert(0, hooks_dir)

from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

from modules.analyzers.tool_pattern_analyzer import (
    ToolPatternAnalyzer, ToolCategory, FeedbackType, UserExpertiseLevel,
    ToolUsage, UsagePattern, FeedbackMessage
)
from modules.analyzers.intelligent_feedback_generator import (
    IntelligentFeedbackGenerator, ContextualFeedback, FeedbackIntensity
)
from modules.analyzers.progressive_verbosity_adapter import (
    ProgressiveVerbosityAdapter, VerbosityLevel, ContextType
)


class TestToolPatternAnalyzer(unittest.TestCase):
    """Test suite for ToolPatternAnalyzer."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = ToolPatternAnalyzer(memory_window=10)
        self.test_session_dir = tempfile.mkdtemp()
        
        # Mock session directory
        os.environ['TEST_SESSION_DIR'] = self.test_session_dir
        
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_session_dir):
            shutil.rmtree(self.test_session_dir)
    
    def test_tool_categorization(self):
        """Test tool categorization accuracy."""
        test_cases = [
            ("mcp__zen__analyze", ToolCategory.MCP_ZEN),
            ("mcp__claude-flow__swarm_init", ToolCategory.MCP_CLAUDE_FLOW),
            ("mcp__filesystem__read_file", ToolCategory.MCP_FILESYSTEM),
            ("mcp__github__create_issue", ToolCategory.MCP_GITHUB),
            ("Write", ToolCategory.FILE_OPERATIONS),
            ("Grep", ToolCategory.SEARCH_OPERATIONS),
            ("WebSearch", ToolCategory.WEB_OPERATIONS),
            ("Bash", ToolCategory.SYSTEM_OPERATIONS),
            ("Task", ToolCategory.WORKFLOW_ORCHESTRATION),
        ]
        
        for tool_name, expected_category in test_cases:
            with self.subTest(tool=tool_name):
                result = self.analyzer.categorize_tool(tool_name)
                self.assertEqual(result, expected_category)
    
    def test_usage_pattern_detection(self):
        """Test detection of usage patterns."""
        # Simulate sequential file operations
        file_operations = [
            ("Read", {"file_path": "/test/file1.py"}, {"success": True}),
            ("Write", {"file_path": "/test/file2.py", "content": "test"}, {"success": True}),
            ("Edit", {"file_path": "/test/file3.py", "old_string": "old", "new_string": "new"}, {"success": True}),
            ("Read", {"file_path": "/test/file4.py"}, {"success": True}),
        ]
        
        for tool_name, tool_input, tool_response in file_operations:
            self.analyzer.record_tool_usage(tool_name, tool_input, tool_response)
        
        patterns = self.analyzer.analyze_current_patterns()
        
        # Should detect sequential file operations pattern
        self.assertTrue(len(patterns) > 0)
        
        # Check for fragmented workflow pattern
        fragmented_patterns = [p for p in patterns if p.pattern_type in ["sequential_file_ops", "fragmented_workflow"]]
        self.assertTrue(len(fragmented_patterns) > 0)
    
    def test_mcp_coordination_detection(self):
        """Test detection of lack of MCP coordination."""
        # Simulate workflow without MCP tools
        non_mcp_operations = [
            ("Write", {"file_path": "/test/app.py", "content": "code"}, {"success": True}),
            ("Bash", {"command": "npm install"}, {"success": True}),
            ("Task", {"description": "Deploy app"}, {"success": True}),
            ("Read", {"file_path": "/test/config.json"}, {"success": True}),
            ("Edit", {"file_path": "/test/app.py", "old_string": "old", "new_string": "new"}, {"success": True}),
        ]
        
        for tool_name, tool_input, tool_response in non_mcp_operations:
            self.analyzer.record_tool_usage(tool_name, tool_input, tool_response)
        
        patterns = self.analyzer.analyze_current_patterns()
        
        # Should detect low MCP coordination
        mcp_patterns = [p for p in patterns if p.pattern_type == "low_mcp_coordination"]
        self.assertTrue(len(mcp_patterns) > 0)
        
        pattern = mcp_patterns[0]
        self.assertLess(pattern.efficiency_score, 0.8)  # Should indicate inefficiency
        self.assertIn("MCP", " ".join(pattern.optimization_opportunities))
    
    def test_performance_pattern_detection(self):
        """Test detection of performance issues."""
        # Simulate slow operations
        slow_operations = [
            ("WebSearch", {"query": "test"}, {"success": True}),
            ("Bash", {"command": "complex_build_script.sh"}, {"success": True}),
            ("WebFetch", {"url": "http://example.com"}, {"success": True}),
        ]
        
        for tool_name, tool_input, tool_response in slow_operations:
            self.analyzer.record_tool_usage(tool_name, tool_input, tool_response, execution_time=6.0)
        
        patterns = self.analyzer.analyze_current_patterns()
        
        # Should detect performance degradation
        perf_patterns = [p for p in patterns if p.pattern_type == "performance_degradation"]
        self.assertTrue(len(perf_patterns) > 0)
    
    def test_feedback_message_generation(self):
        """Test generation of appropriate feedback messages."""
        # Record some inefficient patterns
        operations = [
            ("Read", {"file_path": "/test/file1.py"}, {"success": True}),
            ("Read", {"file_path": "/test/file2.py"}, {"success": True}),
            ("Read", {"file_path": "/test/file3.py"}, {"success": True}),
        ]
        
        current_tool = "Read"
        current_input = {"file_path": "/test/file4.py"}
        current_response = {"success": True}
        
        for tool_name, tool_input, tool_response in operations:
            self.analyzer.record_tool_usage(tool_name, tool_input, tool_response)
        
        feedback = self.analyzer.generate_intelligent_feedback(
            current_tool, current_input, current_response
        )
        
        self.assertTrue(len(feedback) > 0)
        
        # Check feedback quality
        for msg in feedback:
            self.assertIsInstance(msg, FeedbackMessage)
            self.assertIn(msg.feedback_type, [FeedbackType.OPTIMIZATION, FeedbackType.GUIDANCE, 
                                            FeedbackType.WARNING, FeedbackType.SUCCESS, FeedbackType.EDUCATIONAL])
            self.assertGreater(len(msg.message), 10)  # Should have meaningful content
            self.assertGreater(len(msg.actionable_steps), 0)  # Should provide actions
    
    def test_user_expertise_inference(self):
        """Test user expertise level inference."""
        # Simulate beginner behavior
        self.analyzer.session_context = {
            "mcp_tool_usage": 2,
            "advanced_patterns_used": 0,
            "error_rate": 0.4
        }
        
        expertise = self.analyzer._infer_user_expertise()
        self.assertEqual(expertise, UserExpertiseLevel.BEGINNER)
        
        # Simulate expert behavior
        self.analyzer.session_context = {
            "mcp_tool_usage": 25,
            "advanced_patterns_used": 8,
            "error_rate": 0.05
        }
        
        expertise = self.analyzer._infer_user_expertise()
        self.assertEqual(expertise, UserExpertiseLevel.EXPERT)
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Record mixed success/failure operations
        operations = [
            ("Write", {}, {"success": True}),
            ("Write", {}, {"success": False}),
            ("Write", {}, {"success": True}),
            ("Read", {}, {"success": True}),
        ]
        
        for tool_name, tool_input, tool_response in operations:
            self.analyzer.record_tool_usage(tool_name, tool_input, tool_response)
        
        # Test success rate for Write operations
        write_success_rate = self.analyzer._calculate_success_rate(["Write"])
        self.assertAlmostEqual(write_success_rate, 2/3, places=2)  # 2 out of 3 Write operations succeeded
        
        # Test success rate for Read operations
        read_success_rate = self.analyzer._calculate_success_rate(["Read"])
        self.assertEqual(read_success_rate, 1.0)  # 1 out of 1 Read operation succeeded


class TestIntelligentFeedbackGenerator(unittest.TestCase):
    """Test suite for IntelligentFeedbackGenerator."""
    
    def setUp(self):
        """Set up test environment."""
        self.generator = IntelligentFeedbackGenerator()
        self.test_session_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_session_dir):
            shutil.rmtree(self.test_session_dir)
    
    def test_contextual_feedback_generation(self):
        """Test generation of contextual feedback."""
        tool_name = "Write"
        tool_input = {"file_path": "/test/app.py", "content": "print('hello')"}
        tool_response = {"success": True}
        
        feedback = self.generator.generate_contextual_feedback(
            tool_name, tool_input, tool_response
        )
        
        if feedback:  # Feedback might be None if suppressed
            self.assertIn("CLAUDE CODE", feedback)
            self.assertTrue(len(feedback) > 100)  # Should be substantial
    
    def test_zen_guidance_integration(self):
        """Test integration with ZEN consultant."""
        context = "Write operation on Python file"
        zen_guidance = self.generator._get_zen_guidance(context)
        
        self.assertIn("complexity", zen_guidance)
        self.assertIn("coordination", zen_guidance)
        self.assertIn("recommended_tools", zen_guidance)
    
    def test_feedback_suppression(self):
        """Test feedback suppression logic."""
        # Test max messages per session
        self.generator.user_preferences["max_messages_per_session"] = 1
        
        # Generate first feedback
        feedback1 = self.generator.generate_contextual_feedback(
            "Write", {"file_path": "/test/file1.py", "content": "test"}, {"success": True}
        )
        
        # Should generate feedback (if patterns detected)
        # Generate second feedback - should be suppressed
        feedback2 = self.generator.generate_contextual_feedback(
            "Write", {"file_path": "/test/file2.py", "content": "test"}, {"success": True}
        )
        
        # One of them should be None due to suppression
        self.assertTrue(feedback1 is None or feedback2 is None)
    
    def test_feedback_intensity_adaptation(self):
        """Test feedback intensity adaptation."""
        feedback_msg = FeedbackMessage(
            feedback_type=FeedbackType.OPTIMIZATION,
            priority=9,  # High priority
            title="Test",
            message="Test message",
            actionable_steps=["Step 1"],
            related_tools=["tool1"],
            expertise_level=UserExpertiseLevel.ADVANCED,
            show_technical_details=True
        )
        
        zen_guidance = {
            "complexity": MockComplexity(),
            "coordination": MockCoordination(),
            "recommended_tools": ["mcp__zen__analyze"],
            "guidance_level": "verbose",
            "categories": ["development"]
        }
        
        contextual_feedback = self.generator._synthesize_feedback(
            [feedback_msg], zen_guidance, "Write"
        )
        
        self.assertIsNotNone(contextual_feedback)
        self.assertEqual(contextual_feedback.feedback_intensity, FeedbackIntensity.VERBOSE)


class TestProgressiveVerbosityAdapter(unittest.TestCase):
    """Test suite for ProgressiveVerbosityAdapter."""
    
    def setUp(self):
        """Set up test environment."""
        self.adapter = ProgressiveVerbosityAdapter()
        self.test_session_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_session_dir):
            shutil.rmtree(self.test_session_dir)
    
    def test_context_type_detection(self):
        """Test context type detection."""
        # Test onboarding context
        context = self.adapter.detect_context_type(
            "Write", {"file_path": "/test/hello.py"}, [], UserExpertiseLevel.BEGINNER
        )
        
        # Should detect onboarding for beginners with low interaction count
        if self.adapter.session_context.get("interaction_count", 0) < 10:
            self.assertEqual(context, ContextType.ONBOARDING)
    
    def test_verbosity_level_determination(self):
        """Test verbosity level determination."""
        # Test expert with low priority
        verbosity = self.adapter._determine_verbosity_level(
            UserExpertiseLevel.EXPERT, ContextType.ROUTINE, 0.2
        )
        
        # Should be minimal for experts in routine context with low priority
        self.assertEqual(verbosity, VerbosityLevel.MINIMAL)
        
        # Test beginner with high priority
        verbosity = self.adapter._determine_verbosity_level(
            UserExpertiseLevel.BEGINNER, ContextType.LEARNING, 0.9
        )
        
        # Should be educational for beginners in learning context with high priority
        self.assertEqual(verbosity, VerbosityLevel.EDUCATIONAL)
    
    def test_adapted_feedback_creation(self):
        """Test creation of adapted feedback."""
        base_message = "Consider using MCP tools for better coordination"
        templates = self.adapter.VERBOSITY_TEMPLATES[UserExpertiseLevel.INTERMEDIATE]
        
        adapted = self.adapter._create_adapted_feedback(
            base_message, VerbosityLevel.DETAILED, templates, 
            ContextType.EXPLORATION, "mcp__zen__analyze", UserExpertiseLevel.INTERMEDIATE
        )
        
        self.assertIsNotNone(adapted.primary_message)
        self.assertIsNotNone(adapted.explanation)
        self.assertGreater(len(adapted.examples), 0)
        self.assertGreater(adapted.estimated_reading_time, 0)
    
    def test_reading_time_estimation(self):
        """Test reading time estimation."""
        short_text = "Quick message"
        medium_text = "This is a longer message with more detailed explanation and examples"
        long_text = " ".join(["This is a very long message with extensive details"] * 20)
        
        short_time = self.adapter._estimate_reading_time(short_text, None, [], None)
        medium_time = self.adapter._estimate_reading_time(medium_text, "explanation", ["example"], None)
        long_time = self.adapter._estimate_reading_time(long_text, "long explanation", ["ex1", "ex2"], "technical notes")
        
        self.assertLessEqual(short_time, medium_time)
        self.assertLessEqual(medium_time, long_time)
        self.assertGreaterEqual(short_time, 5)  # Minimum 5 seconds


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for the analyzer system."""
    
    def setUp(self):
        """Set up benchmarking environment."""
        self.analyzer = ToolPatternAnalyzer()
        self.generator = IntelligentFeedbackGenerator()
        self.adapter = ProgressiveVerbosityAdapter()
    
    def test_pattern_analysis_performance(self):
        """Benchmark pattern analysis performance."""
        # Generate test data
        operations = []
        for i in range(100):
            operations.append(("Write", {"file_path": f"/test/file{i}.py"}, {"success": True}))
        
        # Record operations
        for tool_name, tool_input, tool_response in operations:
            self.analyzer.record_tool_usage(tool_name, tool_input, tool_response)
        
        # Benchmark pattern analysis
        start_time = time.time()
        self.analyzer.analyze_current_patterns()
        analysis_time = time.time() - start_time
        
        # Should complete within 50ms
        self.assertLess(analysis_time, 0.05, f"Pattern analysis took {analysis_time*1000:.2f}ms, should be <50ms")
        
        print(f"Pattern analysis performance: {analysis_time*1000:.2f}ms for {len(operations)} operations")
    
    def test_feedback_generation_performance(self):
        """Benchmark feedback generation performance."""
        tool_name = "Write"
        tool_input = {"file_path": "/test/app.py", "content": "print('hello world')"}
        tool_response = {"success": True}
        
        # Warm up the system
        self.generator.generate_contextual_feedback(tool_name, tool_input, tool_response)
        
        # Benchmark feedback generation
        start_time = time.time()
        self.generator.generate_contextual_feedback(tool_name, tool_input, tool_response)
        generation_time = time.time() - start_time
        
        # Should complete within 100ms
        self.assertLess(generation_time, 0.1, f"Feedback generation took {generation_time*1000:.2f}ms, should be <100ms")
        
        print(f"Feedback generation performance: {generation_time*1000:.2f}ms")
    
    def test_verbosity_adaptation_performance(self):
        """Benchmark verbosity adaptation performance."""
        base_message = "Consider optimizing your workflow with MCP coordination tools"
        recent_tools = ["Write", "Edit", "Read", "Bash"]
        
        # Benchmark adaptation
        start_time = time.time()
        self.adapter.adapt_feedback_verbosity(
            base_message, "Write", {"file_path": "/test/app.py"}, 
            UserExpertiseLevel.INTERMEDIATE, recent_tools, 0.7
        )
        adaptation_time = time.time() - start_time
        
        # Should complete within 10ms
        self.assertLess(adaptation_time, 0.01, f"Verbosity adaptation took {adaptation_time*1000:.2f}ms, should be <10ms")
        
        print(f"Verbosity adaptation performance: {adaptation_time*1000:.2f}ms")
    
    def test_end_to_end_performance(self):
        """Benchmark end-to-end feedback pipeline performance."""
        tool_name = "Write"
        tool_input = {"file_path": "/test/complex_app.py", "content": "# Complex application code"}
        tool_response = {"success": True}
        execution_time = 0.5
        
        # Benchmark complete pipeline
        start_time = time.time()
        
        # This simulates the complete pipeline as it would run in post_tool_use.py
        try:
            from modules.analyzers.intelligent_feedback_generator import generate_intelligent_stderr_feedback
            generate_intelligent_stderr_feedback(tool_name, tool_input, tool_response, execution_time)
        except ImportError:
            # Fallback to direct testing
            self.generator.generate_contextual_feedback(tool_name, tool_input, tool_response, execution_time)
        
        pipeline_time = time.time() - start_time
        
        # Should complete within 100ms for the complete pipeline
        self.assertLess(pipeline_time, 0.1, f"End-to-end pipeline took {pipeline_time*1000:.2f}ms, should be <100ms")
        
        print(f"End-to-end pipeline performance: {pipeline_time*1000:.2f}ms")


# Mock classes for testing
class MockComplexity:
    """Mock complexity object."""
    def __init__(self):
        self.value = "medium"

class MockCoordination:
    """Mock coordination object."""
    def __init__(self):
        self.value = "SWARM"


# Test data generators
def generate_mock_tool_usage(count: int = 10) -> List[Dict[str, Any]]:
    """Generate mock tool usage data for testing."""
    tools = [
        ("Write", {"file_path": "/test/app.py", "content": "code"}, {"success": True}),
        ("Read", {"file_path": "/test/config.json"}, {"success": True}),
        ("Bash", {"command": "npm install"}, {"success": True}),
        ("mcp__zen__analyze", {"query": "analyze code"}, {"success": True}),
        ("Task", {"description": "Deploy app"}, {"success": False, "error": "Connection failed"}),
    ]
    
    mock_data = []
    for i in range(count):
        tool_name, tool_input, tool_response = tools[i % len(tools)]
        mock_data.append({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": tool_response,
            "execution_time": 0.1 + (i % 5) * 0.1,
            "timestamp": time.time() - (count - i) * 60  # Spread over last hour
        })
    
    return mock_data


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestToolPatternAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestIntelligentFeedbackGenerator))
    test_suite.addTest(unittest.makeSuite(TestProgressiveVerbosityAdapter))
    test_suite.addTest(unittest.makeSuite(TestPerformanceBenchmarks))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TOOL PATTERN ANALYZER TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            newline = '\n'
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split(newline)[0]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            newline = '\n'
            print(f"  - {test}: {traceback.split(newline)[-2]}")
    
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)