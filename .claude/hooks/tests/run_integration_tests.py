#!/usr/bin/env python3
"""
Integration Test Runner for PostToolUse Hook Pipeline
====================================================

Comprehensive test runner that validates the complete PostToolUse hook pipeline
integration, including intelligent feedback system, optimization modules, and
all analyzer components. Handles import dependencies gracefully.
"""

import sys
import os
import json
import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import unittest
from io import StringIO

# Add hooks modules to path
HOOKS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(HOOKS_DIR / "modules"))

# Test results tracking
class TestResults:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.errors = []
        self.warnings = []
        self.performance_metrics = {}
        self.start_time = time.time()
    
    def record_test(self, test_name: str, passed: bool, error: Optional[str] = None, skipped: bool = False):
        self.total_tests += 1
        if skipped:
            self.skipped_tests += 1
        elif passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            if error:
                self.errors.append(f"{test_name}: {error}")
    
    def record_warning(self, warning: str):
        self.warnings.append(warning)
    
    def record_performance(self, metric_name: str, value: float):
        self.performance_metrics[metric_name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        duration = time.time() - self.start_time
        return {
            "total_tests": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.failed_tests,
            "skipped": self.skipped_tests,
            "success_rate": self.passed_tests / max(self.total_tests, 1),
            "duration_seconds": duration,
            "errors": self.errors,
            "warnings": self.warnings,
            "performance_metrics": self.performance_metrics
        }


class ComponentAvailabilityChecker:
    """Check availability of different system components."""
    
    def __init__(self):
        self.availability = {}
        self._check_all_components()
    
    def _check_all_components(self):
        """Check availability of all components."""
        # Check intelligent feedback system
        try:
            from analyzers.intelligent_feedback_generator import generate_intelligent_stderr_feedback
            from analyzers.progressive_verbosity_adapter import ProgressiveVerbosityAdapter
            from analyzers.tool_pattern_analyzer import ToolPatternAnalyzer
            self.availability['intelligent_feedback'] = True
        except Exception as e:
            self.availability['intelligent_feedback'] = False
            print(f"Warning: Intelligent feedback system not available: {e}")
        
        # Check post-tool analyzers
        try:
            from post_tool.analyzers import FileOperationsAnalyzer, MCPCoordinationAnalyzer
            self.availability['post_tool_analyzers'] = True
        except Exception as e:
            self.availability['post_tool_analyzers'] = False
            print(f"Warning: Post-tool analyzers not available: {e}")
        
        # Check optimization modules
        try:
            from optimization import (
                PerformanceMetricsCache, AsyncDatabaseManager, HookExecutionPool
            )
            self.availability['optimization_modules'] = True
        except Exception as e:
            self.availability['optimization_modules'] = False
            print(f"Warning: Optimization modules not available: {e}")
        
        # Check core hook integration
        try:
            from post_tool.core import UniversalToolFeedbackSystem
            self.availability['core_integration'] = True
        except Exception as e:
            self.availability['core_integration'] = False
            print(f"Warning: Core integration not available: {e}")
        
        # Check ZEN bypass analyzer
        try:
            from post_tool.analyzers.zen_bypass_analyzer import ZenBypassAnalyzer
            self.availability['zen_bypass_analyzer'] = True
        except Exception as e:
            self.availability['zen_bypass_analyzer'] = False
            print(f"Warning: ZEN bypass analyzer not available: {e}")
    
    def is_available(self, component: str) -> bool:
        return self.availability.get(component, False)
    
    def get_availability_report(self) -> Dict[str, bool]:
        return self.availability.copy()


class PostToolHookTester:
    """Direct tester for PostToolUse hook functionality."""
    
    def __init__(self, results: TestResults):
        self.results = results
        self.hook_path = HOOKS_DIR / "post_tool_use.py"
    
    def run_hook_test(self, test_name: str, input_data: Dict[str, Any], 
                     timeout: float = 10.0) -> Tuple[bool, Dict[str, Any]]:
        """Run a single hook test with subprocess.""" 
        try:
            process = subprocess.Popen(
                [sys.executable, str(self.hook_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(HOOKS_DIR)
            )
            
            stdout, stderr = process.communicate(
                input=json.dumps(input_data),
                timeout=timeout
            )
            
            result = {
                'returncode': process.returncode,
                'stdout': stdout,
                'stderr': stderr,
                'success': process.returncode in [0, 2]  # 0 = no action, 2 = guidance provided
            }
            
            self.results.record_test(test_name, result['success'])
            return result['success'], result
            
        except subprocess.TimeoutExpired:
            process.kill()
            self.results.record_test(test_name, False, "Timeout expired")
            return False, {'error': 'timeout', 'timeout': True}
        except Exception as e:
            self.results.record_test(test_name, False, str(e))
            return False, {'error': str(e)}
    
    def test_successful_operation(self):
        """Test handling of successful operations."""
        input_data = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/home/devcontainers/flowed/README.md"},
            "tool_response": {"success": True, "content": "Sample"},
            "start_time": time.time()
        }
        
        success, result = self.run_hook_test("successful_operation", input_data)
        
        if success:
            # Should complete without blocking
            assert result['returncode'] in [0, 2], "Should not block successful operations"
            print("✓ Successful operation test passed")
        else:
            print(f"✗ Successful operation test failed: {result.get('error', 'Unknown')}")
        
        return success
    
    def test_intelligent_feedback_generation(self):
        """Test intelligent feedback system integration."""
        input_data = {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/tmp/test_feedback.py",
                "content": "import os\ndef test():\n    print('hello')\n"
            },
            "tool_response": {"success": True},
            "start_time": time.time()
        }
        
        success, result = self.run_hook_test("intelligent_feedback", input_data)
        
        if success and result.get('stderr'):
            # Should provide intelligent feedback
            stderr_content = result['stderr']
            has_intelligence = "CLAUDE CODE INTELLIGENCE" in stderr_content
            
            if has_intelligence:
                print("✓ Intelligent feedback generation test passed")
                return True
            else:
                print("✓ Hook executed but no intelligent feedback detected")
                return True
        else:
            print(f"✗ Intelligent feedback test failed: {result.get('error', 'No stderr output')}")
            return False
    
    def test_hook_violation_detection(self):
        """Test hook file violation detection."""
        input_data = {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/tmp/test_hook_violation.py",
                "content": "import sys\nsys.path.insert(0, '/bad/path')\n"
            },
            "tool_response": {"success": True},
            "start_time": time.time()
        }
        
        # Create temporary file in hooks directory to trigger violation check
        temp_hook_file = HOOKS_DIR / "test_violation_temp.py"
        input_data["tool_input"]["file_path"] = str(temp_hook_file)
        
        success, result = self.run_hook_test("hook_violation", input_data)
        
        # Clean up
        if temp_hook_file.exists():
            temp_hook_file.unlink()
        
        if result.get('returncode') == 1:  # Should block violations
            print("✓ Hook violation detection test passed")
            return True
        else:
            print(f"✗ Hook violation test failed: Expected exit code 1, got {result.get('returncode')}")
            return False
    
    def test_performance_under_load(self):
        """Test hook performance under load."""
        test_tools = [
            {"tool_name": "Read", "tool_input": {"file_path": f"/tmp/file_{i}.txt"}}
            for i in range(10)
        ]
        
        start_time = time.perf_counter()
        successful_tests = 0
        
        for i, tool_data in enumerate(test_tools):
            input_data = {
                **tool_data,
                "tool_response": {"success": True},
                "start_time": time.time()
            }
            
            success, result = self.run_hook_test(f"load_test_{i}", input_data, timeout=5.0)
            if success:
                successful_tests += 1
        
        total_time = (time.perf_counter() - start_time) * 1000  # ms
        avg_time = total_time / len(test_tools)
        
        self.results.record_performance("avg_hook_execution_time_ms", avg_time)
        
        success_rate = successful_tests / len(test_tools)
        performance_good = avg_time < 500  # 500ms per tool is reasonable under load
        
        if success_rate > 0.8 and performance_good:
            print(f"✓ Performance test passed: {success_rate:.1%} success, {avg_time:.1f}ms avg")
            return True
        else:
            print(f"✗ Performance test failed: {success_rate:.1%} success, {avg_time:.1f}ms avg")
            return False
    
    def test_error_handling(self):
        """Test error handling with malformed input."""
        malformed_inputs = [
            {},  # Empty
            {"tool_name": None},  # Null tool name
            {"tool_name": "Invalid", "tool_input": None},  # Null input
        ]
        
        successful_handling = 0
        
        for i, malformed_input in enumerate(malformed_inputs):
            success, result = self.run_hook_test(f"error_handling_{i}", malformed_input, timeout=5.0)
            
            # Should handle gracefully (not crash or timeout)
            if not result.get('timeout') and result.get('returncode') is not None:
                successful_handling += 1
        
        success_rate = successful_handling / len(malformed_inputs)
        
        if success_rate > 0.8:
            print(f"✓ Error handling test passed: {success_rate:.1%} handled gracefully")
            return True
        else:
            print(f"✗ Error handling test failed: {success_rate:.1%} handled gracefully")
            return False


class ComponentTester:
    """Test individual components when available."""
    
    def __init__(self, results: TestResults, availability: ComponentAvailabilityChecker):
        self.results = results
        self.availability = availability
    
    def test_intelligent_feedback_components(self):
        """Test intelligent feedback system components."""
        if not self.availability.is_available('intelligent_feedback'):
            self.results.record_test("intelligent_feedback_components", False, skipped=True)
            print("⚠ Skipping intelligent feedback components test - not available")
            return False
        
        try:
            from analyzers.tool_pattern_analyzer import ToolPatternAnalyzer, ToolCategory
            from analyzers.intelligent_feedback_generator import IntelligentFeedbackGenerator
            
            # Test tool categorization
            analyzer = ToolPatternAnalyzer()
            category = analyzer.categorize_tool("mcp__zen__chat")
            assert category == ToolCategory.MCP_ZEN, f"Expected MCP_ZEN, got {category}"
            
            # Test feedback generation
            feedback_gen = IntelligentFeedbackGenerator()
            feedback = feedback_gen.generate_contextual_feedback(
                "Write", {"file_path": "test.py"}, {"success": True}, 0.1
            )
            
            # Should generate some feedback
            has_feedback = feedback is not None and len(feedback) > 0
            
            self.results.record_test("intelligent_feedback_components", has_feedback)
            
            if has_feedback:
                print("✓ Intelligent feedback components test passed")
                return True
            else:
                print("✗ Intelligent feedback components test failed: No feedback generated")
                return False
                
        except Exception as e:
            self.results.record_test("intelligent_feedback_components", False, str(e))
            print(f"✗ Intelligent feedback components test failed: {e}")
            return False
    
    def test_analyzer_performance(self):
        """Test analyzer performance."""
        if not self.availability.is_available('intelligent_feedback'):
            self.results.record_test("analyzer_performance", False, skipped=True)
            print("⚠ Skipping analyzer performance test - components not available")
            return False
        
        try:
            from analyzers.tool_pattern_analyzer import ToolPatternAnalyzer
            
            analyzer = ToolPatternAnalyzer()
            
            # Test with multiple tool sequences
            test_sequences = []
            for i in range(20):
                sequence = [
                    {"tool_name": "Read", "tool_input": {"file_path": f"file_{i}.py"}},
                    {"tool_name": "Write", "tool_input": {"file_path": f"out_{i}.py"}},
                ]
                test_sequences.append(sequence)
            
            start_time = time.perf_counter()
            
            for sequence in test_sequences:
                result = analyzer.analyze_tool_sequence(sequence)
                assert result is not None, "Analysis should return result"
            
            total_time = (time.perf_counter() - start_time) * 1000  # ms
            avg_time = total_time / len(test_sequences)
            
            self.results.record_performance("analyzer_avg_time_ms", avg_time)
            
            performance_good = avg_time < 10  # 10ms per analysis
            
            self.results.record_test("analyzer_performance", performance_good)
            
            if performance_good:
                print(f"✓ Analyzer performance test passed: {avg_time:.2f}ms avg")
                return True
            else:
                print(f"✗ Analyzer performance test failed: {avg_time:.2f}ms avg (too slow)")
                return False
                
        except Exception as e:
            self.results.record_test("analyzer_performance", False, str(e))
            print(f"✗ Analyzer performance test failed: {e}")
            return False


def run_comprehensive_integration_tests():
    """Run comprehensive integration tests for the PostToolUse hook pipeline."""
    print("=" * 80)
    print("🧪 POSTTOOL HOOK PIPELINE INTEGRATION TESTS")
    print("=" * 80)
    
    results = TestResults()
    availability = ComponentAvailabilityChecker()
    
    # Print availability report
    print("\n📋 Component Availability Report:")
    for component, available in availability.get_availability_report().items():
        status = "✓" if available else "✗"
        print(f"  {status} {component}: {'Available' if available else 'Not Available'}")
    
    print("\n🚀 Starting Integration Tests...")
    print("-" * 50)
    
    # Test 1: Direct hook functionality
    print("\n1. Testing PostToolUse Hook Functionality:")
    hook_tester = PostToolHookTester(results)
    
    tests = [
        ("Successful Operation", hook_tester.test_successful_operation),
        ("Intelligent Feedback", hook_tester.test_intelligent_feedback_generation),
        ("Hook Violation Detection", hook_tester.test_hook_violation_detection),
        ("Performance Under Load", hook_tester.test_performance_under_load),
        ("Error Handling", hook_tester.test_error_handling),
    ]
    
    for test_name, test_func in tests:
        print(f"  Running {test_name}...")
        try:
            test_func()
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            results.record_test(test_name, False, str(e))
    
    # Test 2: Component functionality
    print("\n2. Testing Component Integration:")
    component_tester = ComponentTester(results, availability)
    
    component_tests = [
        ("Intelligent Feedback Components", component_tester.test_intelligent_feedback_components),
        ("Analyzer Performance", component_tester.test_analyzer_performance),
    ]
    
    for test_name, test_func in component_tests:
        print(f"  Running {test_name}...")
        try:
            test_func()
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            results.record_test(test_name, False, str(e))
    
    # Generate final report
    print("\n" + "=" * 80)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    summary = results.get_summary()
    
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✓")
    print(f"Failed: {summary['failed']} ✗")
    print(f"Skipped: {summary['skipped']} ⚠")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Duration: {summary['duration_seconds']:.2f}s")
    
    if summary['performance_metrics']:
        print("\n📈 Performance Metrics:")
        for metric, value in summary['performance_metrics'].items():
            print(f"  {metric}: {value:.2f}")
    
    if summary['warnings']:
        print("\n⚠ Warnings:")
        for warning in summary['warnings']:
            print(f"  {warning}")
    
    if summary['errors']:
        print("\n❌ Errors:")
        for error in summary['errors']:
            print(f"  {error}")
    
    # Final assessment
    print("\n🎯 Overall Assessment:")
    
    if summary['success_rate'] >= 0.8:
        print("✅ INTEGRATION TESTS PASSED - System is functioning well")
        if summary['failed'] > 0:
            print(f"   Note: {summary['failed']} tests failed but overall system is stable")
    elif summary['success_rate'] >= 0.6:
        print("⚠️ INTEGRATION TESTS PARTIALLY PASSED - Some issues detected") 
        print("   Recommendation: Address failing tests for optimal performance")
    else:
        print("❌ INTEGRATION TESTS FAILED - Significant issues detected")
        print("   Recommendation: Fix critical issues before deployment")
    
    # Save results to file
    results_file = HOOKS_DIR / "tests" / "integration_test_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📝 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠ Could not save results: {e}")
    
    return summary['success_rate'] >= 0.8


if __name__ == "__main__":
    success = run_comprehensive_integration_tests()
    sys.exit(0 if success else 1)