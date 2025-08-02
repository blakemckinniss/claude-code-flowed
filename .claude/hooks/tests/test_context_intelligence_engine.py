#!/usr/bin/env python3
"""Comprehensive test suite for Context Intelligence Engine.

Tests all components: GitContextAnalyzer, TechStackDetector, SmartPromptEnhancer,
ProgressiveVerbositySystem, and the main ContextIntelligenceEngine.
"""

import unittest
import asyncio
import tempfile
import os
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open

# Add the modules path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

try:
    from modules.core.context_intelligence_engine import (
        ContextIntelligenceEngine,
        GitContextAnalyzer,
        TechStackDetector,
        SmartPromptEnhancer,
        ProgressiveVerbositySystem,
        TechStack,
        UserExpertiseLevel,
        GitContext,
        ProjectContext,
        EnhancedPrompt,
        create_context_aware_directive
    )
    from modules.memory.zen_memory_integration import ZenMemoryManager, get_zen_memory_manager
except ImportError as e:
    print(f"Import error: {e}")
    # Create mock classes for testing if imports fail
    class MockContextIntelligenceEngine:
        pass
    
    ContextIntelligenceEngine = MockContextIntelligenceEngine


class TestGitContextAnalyzer(unittest.TestCase):
    """Test cases for GitContextAnalyzer."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = GitContextAnalyzer()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('subprocess.run')
    def test_git_available_detection(self, mock_run):
        """Test git availability detection."""
        # Test git available
        mock_run.return_value.returncode = 0
        analyzer = GitContextAnalyzer()
        self.assertTrue(analyzer.git_available)
        
        # Test git not available
        mock_run.side_effect = FileNotFoundError()
        analyzer = GitContextAnalyzer()
        self.assertFalse(analyzer.git_available)
        
    @patch('subprocess.run')
    def test_analyze_repository_context_success(self, mock_run):
        """Test successful repository context analysis."""
        # Mock git commands
        mock_responses = [
            MagicMock(stdout="main\n", returncode=0),  # current branch
            MagicMock(stdout="M file1.py\nA file2.js\n", returncode=0),  # status
            MagicMock(stdout="abc123|Fix bug in auth|John Doe|2024-01-15 10:30:00 +0000\n"
                             "def456|Add new feature|Jane Smith|2024-01-14 15:20:00 +0000\n", 
                     returncode=0),  # log
            MagicMock(stdout="2024-01-01 00:00:00 +0000\n", returncode=0)  # first commit
        ]
        mock_run.side_effect = mock_responses
        
        self.analyzer.git_available = True
        context = self.analyzer.analyze_repository_context()
        
        self.assertIsInstance(context, GitContext)
        self.assertTrue(context.is_repo)
        self.assertEqual(context.current_branch, "main")
        self.assertEqual(context.uncommitted_changes, 2)
        self.assertEqual(len(context.recent_commits), 2)
        self.assertGreater(context.branch_health, 0.0)
        
    @patch('subprocess.run')
    def test_analyze_repository_context_no_git(self, mock_run):
        """Test repository context analysis when git is not available."""
        self.analyzer.git_available = False
        context = self.analyzer.analyze_repository_context()
        
        self.assertIsInstance(context, GitContext)
        self.assertFalse(context.is_repo)
        self.assertEqual(context.current_branch, "unknown")
        self.assertEqual(context.uncommitted_changes, 0)
        
    def test_calculate_branch_health(self):
        """Test branch health calculation."""
        # Good health scenario
        commits = [
            {'hash': 'abc', 'message': 'Fix', 'author': 'Dev', 'date': '2024-01-15'},
            {'hash': 'def', 'message': 'Add', 'author': 'Dev', 'date': '2024-01-14'},
            {'hash': 'ghi', 'message': 'Update', 'author': 'Dev', 'date': '2024-01-13'},
            {'hash': 'jkl', 'message': 'Refactor', 'author': 'Dev', 'date': '2024-01-12'},
            {'hash': 'mno', 'message': 'Test', 'author': 'Dev', 'date': '2024-01-11'}
        ]
        health = self.analyzer._calculate_branch_health(commits, 2)
        self.assertGreaterEqual(health, 0.8)  # Should be high with regular commits, few uncommitted
        
        # Poor health scenario
        health_poor = self.analyzer._calculate_branch_health([], 15)
        self.assertLessEqual(health_poor, 0.5)  # Should be low with no commits, many uncommitted


class TestTechStackDetector(unittest.TestCase):
    """Test cases for TechStackDetector."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.detector = TechStackDetector(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_detect_nodejs_project(self):
        """Test Node.js project detection."""
        # Create package.json
        package_json = {
            "name": "test-project",
            "dependencies": {"express": "^4.18.0"},
            "devDependencies": {"jest": "^28.0.0"}
        }
        
        with open(os.path.join(self.temp_dir, "package.json"), "w") as f:
            json.dump(package_json, f)
            
        # Create some JS files
        with open(os.path.join(self.temp_dir, "app.js"), "w") as f:
            f.write("const express = require('express');\nmodule.exports = app;")
            
        stacks = self.detector.detect_technology_stacks()
        self.assertIn(TechStack.NODEJS, stacks)
        
    def test_detect_python_project(self):
        """Test Python project detection."""
        # Create requirements.txt
        with open(os.path.join(self.temp_dir, "requirements.txt"), "w") as f:
            f.write("flask==2.0.0\nrequests==2.28.0\n")
            
        # Create Python files
        with open(os.path.join(self.temp_dir, "main.py"), "w") as f:
            f.write("import flask\nfrom flask import Flask\ndef main():\n    pass")
            
        stacks = self.detector.detect_technology_stacks()
        self.assertIn(TechStack.PYTHON, stacks)
        
    def test_detect_typescript_project(self):
        """Test TypeScript project detection."""
        # Create tsconfig.json
        with open(os.path.join(self.temp_dir, "tsconfig.json"), "w") as f:
            f.write('{"compilerOptions": {"target": "ES2020"}}')
            
        # Create TypeScript file
        with open(os.path.join(self.temp_dir, "app.ts"), "w") as f:
            f.write("interface User { name: string; }\ntype Status = 'active' | 'inactive';")
            
        stacks = self.detector.detect_technology_stacks()
        self.assertIn(TechStack.TYPESCRIPT, stacks)
        
    def test_detect_react_project(self):
        """Test React project detection."""
        # Create package.json with React
        package_json = {
            "name": "react-app",
            "dependencies": {"react": "^18.0.0", "react-dom": "^18.0.0"}
        }
        
        with open(os.path.join(self.temp_dir, "package.json"), "w") as f:
            json.dump(package_json, f)
            
        # Create JSX file
        with open(os.path.join(self.temp_dir, "App.jsx"), "w") as f:
            f.write("import React, { useState } from 'react';\nfunction App() { return jsx; }")
            
        stacks = self.detector.detect_technology_stacks()
        self.assertIn(TechStack.REACT, stacks)
        self.assertIn(TechStack.NODEJS, stacks)  # Should also detect Node.js
        
    def test_unknown_project(self):
        """Test unknown project detection."""
        # Create only text files
        with open(os.path.join(self.temp_dir, "readme.txt"), "w") as f:
            f.write("This is a text file.")
            
        stacks = self.detector.detect_technology_stacks()
        self.assertIn(TechStack.UNKNOWN, stacks)


class TestSmartPromptEnhancer(unittest.TestCase):
    """Test cases for SmartPromptEnhancer."""
    
    def setUp(self):
        """Set up test environment."""
        self.memory_manager = ZenMemoryManager()
        self.enhancer = SmartPromptEnhancer(self.memory_manager)
        
        # Create mock project context
        self.project_context = ProjectContext(
            git_context=GitContext(
                is_repo=True,
                current_branch="main",
                uncommitted_changes=2,
                recent_commits=[],
                branch_health=0.8,
                last_activity=datetime.now(),
                repository_age_days=30,
                commit_frequency=1.5
            ),
            tech_stacks=[TechStack.NODEJS, TechStack.REACT],
            complexity_indicators={"code_files_count": 50},
            file_structure={".js": 20, ".jsx": 15, ".json": 5},
            project_size="medium",
            dependencies_count=25,
            test_coverage_estimate=0.7,
            documentation_quality=0.6
        )
        
    def test_calculate_vagueness_score(self):
        """Test vagueness score calculation."""
        # Vague prompt
        vague_prompt = "Help me fix the thing"
        vague_score = self.enhancer._calculate_vagueness_score(vague_prompt)
        self.assertGreater(vague_score, 0.5)
        
        # Specific prompt
        specific_prompt = "Implement user authentication with JWT tokens in the Node.js Express API"
        specific_score = self.enhancer._calculate_vagueness_score(specific_prompt)
        self.assertLess(specific_score, 0.3)
        
    def test_identify_missing_context(self):
        """Test missing context identification."""
        # Prompt with missing context
        prompt = "Fix the system"
        missing = self.enhancer._identify_missing_context(prompt, self.project_context)
        self.assertGreater(len(missing), 0)
        self.assertTrue(any("component" in item.lower() for item in missing))
        
        # Comprehensive prompt
        detailed_prompt = "Fix the authentication bug in the login controller by improving JWT validation"
        missing_detailed = self.enhancer._identify_missing_context(detailed_prompt, self.project_context)
        self.assertLessEqual(len(missing_detailed), 1)  # Should have minimal missing context
        
    def test_enhance_prompt(self):
        """Test prompt enhancement."""
        async def run_test():
            vague_prompt = "Update the app"
            enhanced = await self.enhancer.enhance_prompt(vague_prompt, self.project_context)
            
            self.assertIsInstance(enhanced, EnhancedPrompt)
            self.assertEqual(enhanced.original_prompt, vague_prompt)
            self.assertNotEqual(enhanced.enhanced_prompt, vague_prompt)
            self.assertGreater(enhanced.improvement_score, 0.0)
            self.assertGreater(len(enhanced.suggestions), 0)
        
        asyncio.run(run_test())
        
    def test_generate_suggestions(self):
        """Test suggestion generation."""
        prompt = "Add tests for the authentication system"
        suggestions = self.enhancer._generate_suggestions(prompt, self.project_context)
        
        self.assertIsInstance(suggestions, list)
        self.assertTrue(any("jest" in suggestion.lower() or "mocha" in suggestion.lower() 
                          for suggestion in suggestions))


class TestProgressiveVerbositySystem(unittest.TestCase):
    """Test cases for ProgressiveVerbositySystem."""
    
    def setUp(self):
        """Set up test environment."""
        self.memory_manager = ZenMemoryManager()
        self.verbosity_system = ProgressiveVerbositySystem(self.memory_manager)
        
    def test_detect_user_expertise_beginner(self):
        """Test beginner expertise detection."""
        beginner_prompt = "Help me learn how to setup a basic web server"
        expertise = self.verbosity_system.detect_user_expertise(beginner_prompt)
        self.assertEqual(expertise, UserExpertiseLevel.BEGINNER)
        
    def test_detect_user_expertise_intermediate(self):
        """Test intermediate expertise detection."""
        intermediate_prompt = "Create a REST API with authentication and database integration"
        expertise = self.verbosity_system.detect_user_expertise(intermediate_prompt)
        self.assertEqual(expertise, UserExpertiseLevel.INTERMEDIATE)
        
    def test_detect_user_expertise_advanced(self):
        """Test advanced expertise detection."""
        advanced_prompt = "Implement microservices architecture with Docker and API gateway"
        expertise = self.verbosity_system.detect_user_expertise(advanced_prompt)
        self.assertEqual(expertise, UserExpertiseLevel.ADVANCED)
        
    def test_detect_user_expertise_expert(self):
        """Test expert expertise detection."""
        expert_prompt = "Optimize scalability with Kubernetes orchestration and performance tuning"
        expertise = self.verbosity_system.detect_user_expertise(expert_prompt)
        self.assertEqual(expertise, UserExpertiseLevel.EXPERT)
        
    def test_adapt_directive_verbosity_beginner(self):
        """Test directive adaptation for beginners."""
        directive = "🚨 CRITICAL: Deploy the application"
        context = ProjectContext(
            git_context=GitContext(True, "main", 0, [], 0.8, datetime.now(), 30, 1.0),
            tech_stacks=[TechStack.NODEJS],
            complexity_indicators={},
            file_structure={},
            project_size="small",
            dependencies_count=10,
            test_coverage_estimate=0.5,
            documentation_quality=0.3
        )
        
        adapted = self.verbosity_system.adapt_directive_verbosity(
            directive, UserExpertiseLevel.BEGINNER, context
        )
        
        self.assertIn("BEGINNER-FRIENDLY", adapted)
        self.assertIn("HELPFUL CONTEXT", adapted)
        self.assertIn("Node.js", adapted)
        
    def test_adapt_directive_verbosity_expert(self):
        """Test directive adaptation for experts."""
        directive = "🚨 CRITICAL: Deploy the application with monitoring"
        context = ProjectContext(
            git_context=GitContext(True, "main", 0, [], 0.8, datetime.now(), 30, 1.0),
            tech_stacks=[TechStack.NODEJS],
            complexity_indicators={},
            file_structure={},
            project_size="large",
            dependencies_count=100,
            test_coverage_estimate=0.8,
            documentation_quality=0.9
        )
        
        adapted = self.verbosity_system.adapt_directive_verbosity(
            directive, UserExpertiseLevel.EXPERT, context
        )
        
        self.assertIn("EXPERT EXECUTION", adapted)
        self.assertNotIn("🚨", adapted)  # Emojis should be stripped for experts
        self.assertNotIn("CRITICAL", adapted)  # Verbose language should be removed


class TestContextIntelligenceEngine(unittest.TestCase):
    """Test cases for the main ContextIntelligenceEngine."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = ContextIntelligenceEngine(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine.zen_consultant)
        self.assertIsNotNone(self.engine.memory_manager)
        self.assertIsNotNone(self.engine.git_analyzer)
        self.assertIsNotNone(self.engine.tech_detector)
        self.assertIsNotNone(self.engine.prompt_enhancer)
        self.assertIsNotNone(self.engine.verbosity_system)
        
    def test_analyze_full_context(self):
        """Test full context analysis."""
        async def run_test():
            # Create a simple project structure
            with open(os.path.join(self.temp_dir, "package.json"), "w") as f:
                json.dump({"name": "test", "dependencies": {"express": "^4.0.0"}}, f)
                
            with open(os.path.join(self.temp_dir, "app.js"), "w") as f:
                f.write("const express = require('express');")
                
            context = await self.engine.analyze_full_context()
            
            self.assertIsInstance(context, ProjectContext)
            self.assertIsInstance(context.git_context, GitContext)
            self.assertIsInstance(context.tech_stacks, list)
            self.assertGreaterEqual(len(context.tech_stacks), 1)
        
        asyncio.run(run_test())
        
    def test_generate_intelligent_directive(self):
        """Test intelligent directive generation."""
        async def run_test():
            # Setup project files
            with open(os.path.join(self.temp_dir, "package.json"), "w") as f:
                json.dump({"name": "test-app", "dependencies": {"react": "^18.0.0"}}, f)
                
            prompt = "Add user authentication to the application"
            result = await self.engine.generate_intelligent_directive(prompt)
            
            self.assertIsInstance(result, dict)
            self.assertIn("directive", result)
            self.assertIn("context_analysis", result)
            self.assertIn("prompt_enhancement", result)
            self.assertIn("user_adaptation", result)
            self.assertIn("confidence_metrics", result)
            
            # Check context analysis
            context_analysis = result["context_analysis"]
            self.assertIn("git_status", context_analysis)
            self.assertIn("technology_stacks", context_analysis)
            self.assertIn("project_size", context_analysis)
            
            # Check prompt enhancement
            prompt_enhancement = result["prompt_enhancement"]
            self.assertIn("original_prompt", prompt_enhancement)
            self.assertIn("enhanced_prompt", prompt_enhancement)
            self.assertIn("improvement_score", prompt_enhancement)
            
            # Check user adaptation
            user_adaptation = result["user_adaptation"]
            self.assertIn("detected_expertise", user_adaptation)
            self.assertIn("verbosity_level", user_adaptation)
            
            # Check confidence metrics
            confidence_metrics = result["confidence_metrics"]
            self.assertIn("overall_confidence", confidence_metrics)
            self.assertGreater(confidence_metrics["overall_confidence"], 0.0)
            self.assertLessEqual(confidence_metrics["overall_confidence"], 1.0)
        
        asyncio.run(run_test())
        
    def test_caching_mechanism(self):
        """Test context caching mechanism."""
        # First call should populate cache
        asyncio.run(self.engine.analyze_full_context())
        first_timestamp = self.engine._cache_timestamp
        
        # Second call should use cache
        asyncio.run(self.engine.analyze_full_context())
        second_timestamp = self.engine._cache_timestamp
        
        self.assertEqual(first_timestamp, second_timestamp)
        
        # Force refresh should update cache
        asyncio.run(self.engine.analyze_full_context(force_refresh=True))
        third_timestamp = self.engine._cache_timestamp
        
        self.assertGreater(third_timestamp, first_timestamp)


class TestIntegrationFunction(unittest.TestCase):
    """Test cases for integration function."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_create_context_aware_directive_success(self):
        """Test successful context-aware directive creation."""
        async def run_test():
            # Create basic project structure
            with open(os.path.join(self.temp_dir, "package.json"), "w") as f:
                json.dump({"name": "test"}, f)
                
            result = await create_context_aware_directive(
                "Build a simple web server", 
                self.temp_dir
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn("hookSpecificOutput", result)
            hook_output = result["hookSpecificOutput"]
            self.assertEqual(hook_output["hookEventName"], "ContextIntelligentDirective")
            self.assertIn("additionalContext", hook_output)
            self.assertNotIn("fallback", hook_output)
        
        asyncio.run(run_test())
        
    def test_create_context_aware_directive_fallback(self):
        """Test fallback mechanism when error occurs."""
        async def run_test():
            # Use invalid project directory to trigger error
            with patch('modules.core.context_intelligence_engine.ContextIntelligenceEngine') as mock_engine:
                mock_engine.return_value.generate_intelligent_directive.side_effect = Exception("Test error")
                
                result = await create_context_aware_directive(
                    "Test prompt", 
                    "/nonexistent/directory"
                )
                
                self.assertIsInstance(result, dict)
                self.assertIn("hookSpecificOutput", result)
                hook_output = result["hookSpecificOutput"]
                self.assertTrue(hook_output.get("fallback", False))
                self.assertIn("error", hook_output)
        
        asyncio.run(run_test())


class TestPerformanceAndMemory(unittest.TestCase):
    """Performance and memory usage tests."""
    
    def test_memory_usage_analysis(self):
        """Test that context analysis doesn't consume excessive memory."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create multiple engines and analyze contexts
        engines = []
        for _i in range(10):
            temp_dir = tempfile.mkdtemp()
            engine = ContextIntelligenceEngine(temp_dir)
            engines.append((engine, temp_dir))
            
            # Run analysis
            asyncio.run(engine.analyze_full_context())
            
        # Clean up
        for engine, temp_dir in engines:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        del engines
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024, 
                       f"Memory usage increased by {memory_increase / (1024*1024):.1f}MB")
        
    def test_performance_context_analysis(self):
        """Test context analysis performance."""
        async def run_test():
            import time
            
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Create a moderately complex project
                os.makedirs(os.path.join(temp_dir, "src"))
                os.makedirs(os.path.join(temp_dir, "tests"))
                
                with open(os.path.join(temp_dir, "package.json"), "w") as f:
                    json.dump({
                        "name": "performance-test",
                        "dependencies": {"express": "^4.0.0", "react": "^18.0.0"}
                    }, f)
                    
                # Create multiple files
                for i in range(20):
                    with open(os.path.join(temp_dir, f"src/file{i}.js"), "w") as f:
                        f.write(f"// File {i}\nconst express = require('express');\n")
                        
                engine = ContextIntelligenceEngine(temp_dir)
                
                # Measure analysis time
                start_time = time.time()
                context = await engine.analyze_full_context()
                end_time = time.time()
                
                analysis_time = end_time - start_time
                
                # Analysis should complete within reasonable time (5 seconds)
                self.assertLess(analysis_time, 5.0, 
                               f"Context analysis took {analysis_time:.2f} seconds")
                
                # Verify context was analyzed
                self.assertIsInstance(context, ProjectContext)
                self.assertGreater(len(context.tech_stacks), 0)
                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        asyncio.run(run_test())


def run_all_tests():
    """Run all test suites."""
    print("🧪 Running Context Intelligence Engine Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGitContextAnalyzer,
        TestTechStackDetector,
        TestSmartPromptEnhancer,
        TestProgressiveVerbositySystem,
        TestContextIntelligenceEngine,
        TestIntegrationFunction,
        TestPerformanceAndMemory
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print(f"  • Tests run: {result.testsRun}")
    print(f"  • Failures: {len(result.failures)}")
    print(f"  • Errors: {len(result.errors)}")
    print(f"  • Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            # Use raw string or variable to avoid f-string escape issues
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  • {test}: {error_msg}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            # Use raw string or variable to avoid f-string escape issues
            tb_lines = traceback.split('\n')
            error_msg = tb_lines[-2] if len(tb_lines) > 1 else 'Unknown error'
            print(f"  • {test}: {error_msg}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)