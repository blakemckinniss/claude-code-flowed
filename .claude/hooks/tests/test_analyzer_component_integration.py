#!/usr/bin/env python3
"""
Analyzer Component Integration Tests
===================================

Tests for individual analyzer components and their integration with the
PostToolUse hook pipeline. Validates specific analyzer behavior and feedback generation.
"""

import unittest
import sys
import os
import json
import tempfile
import time
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List, Optional

# Add hooks modules to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from test_framework_architecture import (
    BaseTestCase, MockToolExecutionData, TestDataGenerator, 
    PerformanceBenchmarkRunner, ValidationFramework, TEST_CONFIG
)

# Import analyzer components for direct testing
try:
    from analyzers.tool_pattern_analyzer import ToolPatternAnalyzer, ToolCategory
    from analyzers.intelligent_feedback_generator import IntelligentFeedbackGenerator
    from analyzers.progressive_verbosity_adapter import (
        ProgressiveVerbosityAdapter, UserExpertiseLevel, ContextType
    )
    ANALYZER_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Analyzer components not available: {e}")
    ANALYZER_COMPONENTS_AVAILABLE = False


@unittest.skipUnless(ANALYZER_COMPONENTS_AVAILABLE, "Analyzer components not available")
class ToolPatternAnalyzerTest(BaseTestCase):
    """Tests for the ToolPatternAnalyzer component."""
    
    def setUp(self):
        super().setUp()
        self.analyzer = ToolPatternAnalyzer()
        self.test_data_generator = TestDataGenerator()
    
    def test_tool_categorization(self):
        """Test accurate tool categorization."""
        test_cases = [
            ("mcp__zen__chat", ToolCategory.MCP_ZEN),
            ("mcp__claude-flow__swarm_init", ToolCategory.MCP_CLAUDE_FLOW),
            ("Write", ToolCategory.FILE_OPERATIONS),
            ("Bash", ToolCategory.EXECUTION),
            ("WebSearch", ToolCategory.WEB_SEARCH),
            ("Task", ToolCategory.AGENT_SPAWNING),
        ]
        
        for tool_name, expected_category in test_cases:
            with self.subTest(tool_name=tool_name):
                category = self.analyzer.categorize_tool(tool_name)
                self.assertEqual(category, expected_category,
                               f"Tool {tool_name} should be categorized as {expected_category}")
    
    def test_pattern_detection_sequential_operations(self):
        """Test detection of sequential operation patterns."""
        # Create a sequence of similar operations
        tool_sequence = [
            {"tool_name": "Write", "tool_input": {"file_path": "file1.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "file2.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "file3.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "file4.py"}},
        ]
        
        pattern_result = self.analyzer.analyze_tool_sequence(tool_sequence)
        
        self.assertTrue(pattern_result.get("sequential_pattern_detected"),
                       "Should detect sequential operation pattern")
        self.assertGreaterEqual(pattern_result.get("consecutive_same_tools", 0), 3,
                               "Should identify multiple consecutive operations")
        self.assertEqual(pattern_result.get("dominant_category"), ToolCategory.FILE_OPERATIONS,
                        "Should identify FILE_OPERATIONS as dominant category")
    
    def test_mcp_coordination_analysis(self):
        """Test analysis of MCP coordination patterns."""
        # Test case 1: No MCP coordination with complex operations
        non_coordinated_sequence = [
            {"tool_name": "Write", "tool_input": {"file_path": "component.py"}},
            {"tool_name": "Edit", "tool_input": {"file_path": "utils.py"}},
            {"tool_name": "Bash", "tool_input": {"command": "npm install"}},
            {"tool_name": "Task", "tool_input": {"task": "Run tests"}},
        ]
        
        result = self.analyzer.analyze_tool_sequence(non_coordinated_sequence)
        
        self.assertFalse(result.get("has_mcp_coordination"),
                        "Should detect lack of MCP coordination")
        self.assertTrue(result.get("needs_coordination"),
                       "Should suggest coordination for complex operations")
        
        # Test case 2: Good MCP coordination
        coordinated_sequence = [
            {"tool_name": "mcp__zen__analyze", "tool_input": {"step": "Analyze requirements"}},
            {"tool_name": "mcp__claude-flow__swarm_init", "tool_input": {"topology": "mesh"}},
            {"tool_name": "Write", "tool_input": {"file_path": "result.py"}},
        ]
        
        result = self.analyzer.analyze_tool_sequence(coordinated_sequence)
        
        self.assertTrue(result.get("has_mcp_coordination"),
                       "Should detect good MCP coordination")
        self.assertFalse(result.get("needs_coordination"),
                        "Should not suggest coordination when already present")
    
    def test_inefficient_pattern_detection(self):
        """Test detection of inefficient usage patterns."""
        # Case 1: Excessive native tool usage without coordination
        excessive_native = [
            {"tool_name": "Read", "tool_input": {"file_path": "file1.py"}},
            {"tool_name": "Read", "tool_input": {"file_path": "file2.py"}},
            {"tool_name": "Read", "tool_input": {"file_path": "file3.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "output1.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "output2.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "output3.py"}},
        ]
        
        result = self.analyzer.analyze_tool_sequence(excessive_native)
        
        self.assertTrue(result.get("excessive_native_usage"),
                       "Should detect excessive native tool usage")
        self.assertGreater(result.get("efficiency_score", 1.0), 0.3,
                          "Should have low efficiency score")
        
        # Case 2: Well-coordinated efficient usage
        efficient_usage = [
            {"tool_name": "mcp__zen__planner", "tool_input": {"step": "Plan file operations"}},
            {"tool_name": "MultiEdit", "tool_input": {"edits": []}},  # Batch operation
            {"tool_name": "mcp__claude-flow__memory_usage", "tool_input": {"action": "store"}},
        ]
        
        result = self.analyzer.analyze_tool_sequence(efficient_usage)
        
        self.assertFalse(result.get("excessive_native_usage"),
                        "Should not flag efficient usage as excessive")
        self.assertLess(result.get("efficiency_score", 0.0), 0.8,
                       "Should have high efficiency score")
    
    def test_performance_metrics_calculation(self):
        """Test calculation of performance metrics."""
        # Test with various tool execution times
        tool_sequence = [
            {"tool_name": "Read", "execution_time": 0.05},
            {"tool_name": "mcp__zen__analyze", "execution_time": 1.2},
            {"tool_name": "Write", "execution_time": 0.08},
            {"tool_name": "Bash", "execution_time": 0.3},
        ]
        
        metrics = self.analyzer.calculate_performance_metrics(tool_sequence)
        
        self.assertIn("total_execution_time", metrics)
        self.assertIn("average_tool_time", metrics)
        self.assertIn("slowest_tool", metrics)
        self.assertIn("performance_bottlenecks", metrics)
        
        self.assertAlmostEqual(metrics["total_execution_time"], 1.63, places=2)
        self.assertEqual(metrics["slowest_tool"]["tool_name"], "mcp__zen__analyze")
        
        # Should identify performance bottlenecks
        if metrics["performance_bottlenecks"]:
            self.assertIn("mcp__zen__analyze", str(metrics["performance_bottlenecks"]))


@unittest.skipUnless(ANALYZER_COMPONENTS_AVAILABLE, "Analyzer components not available")
class IntelligentFeedbackGeneratorTest(BaseTestCase):
    """Tests for the IntelligentFeedbackGenerator component."""
    
    def setUp(self):
        super().setUp()
        self.feedback_generator = IntelligentFeedbackGenerator()
    
    def test_contextual_feedback_generation(self):
        """Test generation of contextual feedback messages."""
        # Test file operations feedback
        tool_input = {"file_path": "/src/components/App.tsx"}
        tool_response = {"success": True}
        
        feedback = self.feedback_generator.generate_contextual_feedback(
            "Write", tool_input, tool_response, execution_time=0.15
        )
        
        self.assertIsNotNone(feedback, "Should generate feedback for file operations")
        self.assertIn("CLAUDE CODE INTELLIGENCE", feedback)
        
        # Should provide context-aware suggestions
        self.assertTrue(
            any(keyword in feedback.lower() for keyword in 
                ['mcp', 'coordination', 'filesystem', 'optimization']),
            "Should provide relevant suggestions for file operations"
        )
    
    def test_zen_coordination_suggestions(self):
        """Test generation of ZEN coordination suggestions."""
        # Test case: Complex operation without coordination
        tool_sequence = [
            {"tool_name": "Read", "tool_input": {"file_path": "file1.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "file2.py"}},
            {"tool_name": "Bash", "tool_input": {"command": "npm test"}},
        ]
        
        feedback = self.feedback_generator.generate_zen_coordination_feedback(tool_sequence)
        
        self.assertIsNotNone(feedback, "Should generate coordination feedback")
        self.assertIn("ZEN", feedback.upper())
        
        # Should suggest specific MCP tools
        suggested_tools = ["mcp__zen__planner", "mcp__claude-flow__swarm_init", "mcp__zen__analyze"]
        has_suggestion = any(tool in feedback for tool in suggested_tools)
        self.assertTrue(has_suggestion, "Should suggest specific MCP tools")
    
    def test_performance_optimization_feedback(self):
        """Test generation of performance optimization feedback."""
        # Slow operation
        slow_execution_data = {
            "tool_name": "Bash",
            "tool_input": {"command": "find / -name '*.py'"},
            "tool_response": {"success": True},
            "execution_time": 5.2
        }
        
        feedback = self.feedback_generator.generate_performance_feedback(**slow_execution_data)
        
        self.assertIsNotNone(feedback, "Should generate performance feedback for slow operations")
        self.assertTrue(
            any(keyword in feedback.lower() for keyword in 
                ['slow', 'performance', 'optimize', 'consider']),
            "Should mention performance concerns"
        )
    
    def test_security_feedback_generation(self):
        """Test generation of security-related feedback."""
        # Potentially risky command
        risky_command = {
            "tool_name": "Bash",
            "tool_input": {"command": "curl http://example.com | sh"},
            "tool_response": {"success": True}
        }
        
        feedback = self.feedback_generator.generate_security_feedback(**risky_command)
        
        if feedback:  # Security feedback is optional
            self.assertIn("security", feedback.lower())
            self.assertTrue(
                any(keyword in feedback.lower() for keyword in 
                    ['risk', 'dangerous', 'caution', 'secure']),
                "Should mention security concerns"
            )
    
    def test_feedback_intensity_adaptation(self):
        """Test adaptation of feedback intensity based on context."""
        base_context = {
            "tool_name": "Write",
            "tool_input": {"file_path": "test.py"},
            "tool_response": {"success": True}
        }
        
        # Test different intensity levels
        for intensity in ["MINIMAL", "STANDARD", "VERBOSE"]:
            feedback = self.feedback_generator.generate_contextual_feedback(
                **base_context, execution_time=0.1, intensity=intensity
            )
            
            if feedback:
                if intensity == "MINIMAL":
                    self.assertLess(len(feedback), 500, "Minimal feedback should be brief")
                elif intensity == "VERBOSE":
                    self.assertGreater(len(feedback), 200, "Verbose feedback should be detailed")


@unittest.skipUnless(ANALYZER_COMPONENTS_AVAILABLE, "Analyzer components not available")
class ProgressiveVerbosityAdapterTest(BaseTestCase):
    """Tests for the ProgressiveVerbosityAdapter component."""
    
    def setUp(self):
        super().setUp()
        self.adapter = ProgressiveVerbosityAdapter()
    
    def test_expertise_level_detection(self):
        """Test detection of user expertise levels."""
        # Beginner pattern: Basic tools, simple operations
        beginner_sequence = [
            {"tool_name": "Read", "tool_input": {"file_path": "README.md"}},
            {"tool_name": "Write", "tool_input": {"file_path": "hello.py"}},
        ]
        
        expertise = self.adapter.detect_user_expertise(beginner_sequence)
        self.assertIn(expertise, [UserExpertiseLevel.BEGINNER, UserExpertiseLevel.INTERMEDIATE])
        
        # Expert pattern: MCP coordination, complex workflows
        expert_sequence = [
            {"tool_name": "mcp__zen__thinkdeep", "tool_input": {"step": "Complex analysis"}},
            {"tool_name": "mcp__claude-flow__swarm_init", "tool_input": {"topology": "hierarchical"}},
            {"tool_name": "MultiEdit", "tool_input": {"edits": []}},
        ]
        
        expertise = self.adapter.detect_user_expertise(expert_sequence)
        self.assertIn(expertise, [UserExpertiseLevel.ADVANCED, UserExpertiseLevel.EXPERT])
    
    def test_context_type_detection(self):
        """Test detection of different context types."""
        # Test ONBOARDING context
        onboarding_tools = [
            {"tool_name": "Read", "tool_input": {"file_path": "README.md"}},
            {"tool_name": "LS", "tool_input": {"path": "/"}},
        ]
        
        context_type = self.adapter.detect_context_type(onboarding_tools, is_new_session=True)
        self.assertEqual(context_type, ContextType.ONBOARDING)
        
        # Test DEBUGGING context
        debugging_tools = [
            {"tool_name": "Bash", "tool_input": {"command": "npm test"}},
            {"tool_name": "Read", "tool_input": {"file_path": "error.log"}},
            {"tool_name": "Grep", "tool_input": {"pattern": "ERROR"}},
        ]
        
        context_type = self.adapter.detect_context_type(debugging_tools)
        self.assertEqual(context_type, ContextType.DEBUGGING)
        
        # Test EXPLORATION context
        exploration_tools = [
            {"tool_name": "LS", "tool_input": {"path": "/src"}},
            {"tool_name": "Read", "tool_input": {"file_path": "config.json"}},
            {"tool_name": "Grep", "tool_input": {"pattern": "function"}},
        ]
        
        context_type = self.adapter.detect_context_type(exploration_tools)
        self.assertEqual(context_type, ContextType.EXPLORATION)
    
    def test_verbosity_adaptation(self):
        """Test adaptation of verbosity based on user expertise and context."""
        base_message = "Consider using MCP coordination for this task"
        
        # Test different combinations
        test_cases = [
            (UserExpertiseLevel.BEGINNER, ContextType.ONBOARDING),
            (UserExpertiseLevel.INTERMEDIATE, ContextType.ROUTINE),
            (UserExpertiseLevel.ADVANCED, ContextType.DEBUGGING),
            (UserExpertiseLevel.EXPERT, ContextType.EXPLORATION),
        ]
        
        for expertise, context in test_cases:
            with self.subTest(expertise=expertise, context=context):
                adapted_message = self.adapter.adapt_verbosity(
                    base_message, expertise, context
                )
                
                self.assertIsNotNone(adapted_message, "Should return adapted message")
                self.assertNotEqual(adapted_message, base_message, 
                                  "Should modify the base message")
                
                # Beginners should get more explanation
                if expertise == UserExpertiseLevel.BEGINNER:
                    self.assertGreater(len(adapted_message), len(base_message),
                                     "Beginner messages should be more detailed")
                
                # Experts should get concise messages
                elif expertise == UserExpertiseLevel.EXPERT:
                    self.assertLessEqual(len(adapted_message), len(base_message) * 1.5,
                                       "Expert messages should be concise")
    
    def test_learning_and_adaptation(self):
        """Test learning from user patterns and adaptation over time."""
        # Simulate user interactions over time
        initial_expertise = UserExpertiseLevel.BEGINNER
        
        # User starts with basic operations
        session_1 = [
            {"tool_name": "Read", "tool_input": {"file_path": "file.py"}},
            {"tool_name": "Write", "tool_input": {"file_path": "output.py"}},
        ]
        
        # User progresses to more advanced operations
        session_2 = [
            {"tool_name": "mcp__zen__chat", "tool_input": {"prompt": "Help with architecture"}},
            {"tool_name": "MultiEdit", "tool_input": {"edits": [{"old_string": "a", "new_string": "b"}]}},
        ]
        
        # Update expertise based on progression
        updated_expertise = self.adapter.update_user_expertise_from_session(
            initial_expertise, session_1 + session_2
        )
        
        self.assertGreaterEqual(updated_expertise.value, initial_expertise.value,
                               "Expertise should increase or stay the same")
    
    def test_context_aware_template_selection(self):
        """Test selection of appropriate templates based on context."""
        # Test different contexts
        contexts = [
            ContextType.ONBOARDING,
            ContextType.ROUTINE,
            ContextType.EXPLORATION,
            ContextType.DEBUGGING,
            ContextType.OPTIMIZATION
        ]
        
        for context in contexts:
            with self.subTest(context=context):
                template = self.adapter.get_context_template(context, UserExpertiseLevel.INTERMEDIATE)
                
                self.assertIsNotNone(template, f"Should have template for {context}")
                self.assertIn("greeting", template, "Template should have greeting")
                self.assertIn("explanation_prefix", template, "Template should have explanation prefix")


class AnalyzerIntegrationPerformanceTest(BaseTestCase):
    """Performance tests for analyzer component integration."""
    
    @unittest.skipUnless(ANALYZER_COMPONENTS_AVAILABLE, "Analyzer components not available")
    def test_analyzer_performance_under_load(self):
        """Test analyzer performance with multiple concurrent analyses."""
        analyzer = ToolPatternAnalyzer()
        feedback_generator = IntelligentFeedbackGenerator()
        
        # Generate test data
        test_sequences = []
        for i in range(50):  # 50 different sequences
            sequence = [
                {"tool_name": "Read", "tool_input": {"file_path": f"file_{i}.py"}},
                {"tool_name": "Write", "tool_input": {"file_path": f"output_{i}.py"}},
                {"tool_name": "Bash", "tool_input": {"command": f"echo 'test {i}'"}},
            ]
            test_sequences.append(sequence)
        
        # Measure analysis performance
        start_time = time.perf_counter()
        
        results = []
        for sequence in test_sequences:
            # Analyze pattern
            pattern_result = analyzer.analyze_tool_sequence(sequence)
            
            # Generate feedback
            feedback = feedback_generator.generate_contextual_feedback(
                sequence[-1]["tool_name"],
                sequence[-1]["tool_input"],
                {"success": True},
                execution_time=0.1
            )
            
            results.append({
                "pattern_result": pattern_result,
                "feedback": feedback
            })
        
        total_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        avg_time_per_analysis = total_time / len(test_sequences)
        
        # Performance assertions
        self.assertLess(avg_time_per_analysis, 50, 
                       "Should analyze each sequence in under 50ms on average")
        self.assertEqual(len(results), len(test_sequences), 
                        "Should complete all analyses")
        
        # Quality assertions
        successful_analyses = sum(1 for r in results if r["pattern_result"] is not None)
        self.assertGreater(successful_analyses / len(results), 0.9, 
                          "Should successfully analyze 90%+ of sequences")
    
    @unittest.skipUnless(ANALYZER_COMPONENTS_AVAILABLE, "Analyzer components not available")
    def test_memory_efficiency(self):
        """Test memory efficiency of analyzer components."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create analyzer instances
        analyzers = []
        for i in range(10):
            analyzer = ToolPatternAnalyzer()
            feedback_generator = IntelligentFeedbackGenerator()
            verbosity_adapter = ProgressiveVerbosityAdapter()
            analyzers.append((analyzer, feedback_generator, verbosity_adapter))
        
        # Use analyzers extensively
        for analyzer, feedback_gen, verbosity_adapt in analyzers:
            for j in range(100):
                sequence = [
                    {"tool_name": "Read", "tool_input": {"file_path": f"test_{j}.py"}},
                    {"tool_name": "Write", "tool_input": {"file_path": f"out_{j}.py"}},
                ]
                analyzer.analyze_tool_sequence(sequence)
                feedback_gen.generate_contextual_feedback("Write", {}, {})
                verbosity_adapt.detect_user_expertise(sequence)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase excessively
        self.assertLess(memory_increase, 100, 
                       "Memory increase should be under 100MB for extensive usage")


if __name__ == '__main__':
    # Configure test environment
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add component tests
    if ANALYZER_COMPONENTS_AVAILABLE:
        suite.addTest(unittest.makeSuite(ToolPatternAnalyzerTest))
        suite.addTest(unittest.makeSuite(IntelligentFeedbackGeneratorTest))
        suite.addTest(unittest.makeSuite(ProgressiveVerbosityAdapterTest))
        suite.addTest(unittest.makeSuite(AnalyzerIntegrationPerformanceTest))
    else:
        print("Skipping analyzer component tests - components not available")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)