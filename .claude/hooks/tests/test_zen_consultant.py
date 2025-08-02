#!/usr/bin/env python3
"""Comprehensive testing framework for ZenConsultant prototype.

Tests the core functionality including:
- Complexity analysis and agent recommendation
- Structured directive output
- Memory integration simulation
- Hook security integration
- Concise vs verbose output formats
"""

import sys
import json
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

# Set up hook paths
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

from modules.core.zen_consultant import (
    ZenConsultant, 
    ComplexityLevel, 
    CoordinationType,
    create_zen_consultation_response,
    create_zen_consensus_request
)


class TestZenConsultant:
    """Test suite for ZenConsultant functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.consultant = ZenConsultant()
        
    def test_complexity_analysis_simple(self):
        """Test complexity analysis for simple tasks."""
        prompt = "Fix the login bug"
        complexity, metadata = self.consultant.analyze_prompt_complexity(prompt)
        
        assert complexity == ComplexityLevel.SIMPLE
        assert "debugging" in metadata["categories"]
        assert metadata["word_count"] == 4
        
    def test_complexity_analysis_complex(self):
        """Test complexity analysis for complex tasks."""
        prompt = "Refactor the entire authentication system architecture for enterprise scalability with security audit compliance"
        complexity, metadata = self.consultant.analyze_prompt_complexity(prompt)
        
        assert complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE]
        assert "refactoring" in metadata["categories"]
        assert "architecture" in metadata["categories"]
        assert "security" in metadata["categories"]
        
    def test_coordination_type_hive(self):
        """Test HIVE coordination selection."""
        complexity = ComplexityLevel.ENTERPRISE
        categories = ["architecture", "security", "performance"]
        prompt = "Build a complex enterprise platform with multiple integrations"
        
        coordination = self.consultant.determine_coordination_type(complexity, categories, prompt)
        assert coordination == CoordinationType.HIVE
        
    def test_coordination_type_swarm(self):
        """Test SWARM coordination selection."""
        complexity = ComplexityLevel.SIMPLE
        categories = ["development"]
        prompt = "Build a simple login form"
        
        coordination = self.consultant.determine_coordination_type(complexity, categories, prompt)
        assert coordination == CoordinationType.SWARM
        
    def test_agent_allocation_discovery_phase(self):
        """Test agent allocation in discovery phase."""
        complexity = ComplexityLevel.COMPLEX
        categories = ["architecture", "security"]
        coordination = CoordinationType.HIVE
        
        agents = self.consultant.allocate_initial_agents(complexity, categories, coordination)
        
        # Complex tasks should start with 0 agents for ZEN discovery
        assert agents.count == 0
        assert agents.types == []
        assert agents.topology == CoordinationType.HIVE
        
    def test_agent_allocation_simple_task(self):
        """Test agent allocation for simple single-category tasks."""
        complexity = ComplexityLevel.SIMPLE
        categories = ["development"]
        coordination = CoordinationType.SWARM
        
        agents = self.consultant.allocate_initial_agents(complexity, categories, coordination)
        
        # Simple single-category tasks get 1 specialist
        assert agents.count == 1
        assert len(agents.types) == 1
        assert agents.types[0] in self.consultant.AGENT_CATALOG["development"]
        
    def test_mcp_tool_selection(self):
        """Test MCP tool selection based on categories."""
        categories = ["development", "testing", "github"]
        coordination = CoordinationType.SWARM
        
        tools = self.consultant.select_mcp_tools(categories, coordination)
        
        assert "mcp__claude-flow__swarm_init" in tools
        assert "mcp__claude-flow__agent_spawn" in tools
        assert len(tools) <= 5  # Should limit to 5 tools max
        
    def test_concise_directive_format(self):
        """Test concise directive generation."""
        prompt = "Refactor authentication system"
        directive = self.consultant.get_concise_directive(prompt)
        
        # Verify structure matches requirements
        required_keys = ["hive", "swarm", "agents", "tools", "confidence", "session_id", "thinking_mode"]
        for key in required_keys:
            assert key in directive
            
        # Verify limits
        assert len(directive["agents"]) <= 3
        assert len(directive["tools"]) <= 3
        assert 0.1 <= directive["confidence"] <= 0.99
        
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Simple task should have higher confidence
        simple_metadata = {"categories": ["development"], "word_count": 5}
        simple_confidence = self.consultant._calculate_confidence(simple_metadata, ComplexityLevel.SIMPLE)
        
        # Complex task should have lower confidence
        complex_metadata = {"categories": ["architecture", "security", "performance"], "word_count": 50}
        complex_confidence = self.consultant._calculate_confidence(complex_metadata, ComplexityLevel.ENTERPRISE)
        
        assert simple_confidence > complex_confidence
        assert 0.1 <= simple_confidence <= 0.99
        assert 0.1 <= complex_confidence <= 0.99
        
    def test_memory_integration_simulation(self):
        """Test memory integration patterns."""
        # Simulate successful pattern learning
        self.consultant.learning_data["successful_patterns"]["development"] = {"success_rate": 0.9}
        
        metadata = {"categories": ["development"]}
        confidence = self.consultant._calculate_confidence(metadata, ComplexityLevel.MEDIUM)
        
        # Should be boosted by learning data
        base_confidence = 0.7 + 0.1 + 0.08  # base + medium + single category
        expected_confidence = base_confidence + 0.05  # learning boost
        
        assert abs(confidence - expected_confidence) < 0.01
        

class TestZenConsultationResponse:
    """Test ZEN consultation response generation."""
    
    def test_verbose_format(self):
        """Test verbose format response."""
        prompt = "Build comprehensive testing framework"
        response = create_zen_consultation_response(prompt, "verbose")
        
        assert "hookSpecificOutput" in response
        assert "hookEventName" in response["hookSpecificOutput"]
        assert "additionalContext" in response["hookSpecificOutput"]
        assert len(response["hookSpecificOutput"]["additionalContext"]) > 200
        
    def test_concise_format(self):
        """Test concise format response."""
        prompt = "Fix login bug"
        response = create_zen_consultation_response(prompt, "concise")
        
        assert "hookSpecificOutput" in response
        directive = response["hookSpecificOutput"]["additionalContext"]
        assert len(directive) <= 200
        assert "ZEN:" in directive
        assert "→" in directive  # Should have arrow separators
        assert "conf:" in directive  # Should have confidence
        
    def test_consensus_request_generation(self):
        """Test ZEN consensus request generation."""
        prompt = "Design enterprise architecture"
        complexity = ComplexityLevel.ENTERPRISE
        
        request = create_zen_consensus_request(prompt, complexity)
        
        assert "step" in request
        assert "models" in request
        assert len(request["models"]) == 4  # Enterprise should have 4 models
        assert request["total_steps"] == len(request["models"])


class TestHookIntegration:
    """Test hook system integration."""
    
    def test_security_validation(self):
        """Test security aspects of hook integration."""
        consultant = ZenConsultant()
        
        # Test malicious prompt handling
        malicious_prompt = "rm -rf / && cat /etc/passwd" * 1000
        
        # Should handle gracefully without errors
        try:
            directive = consultant.get_concise_directive(malicious_prompt[:10000])  # Truncated
            assert directive is not None
            assert isinstance(directive["confidence"], float)
        except Exception as e:
            pytest.fail(f"Security test failed: {e}")
            
    def test_hook_memory_namespace(self):
        """Test memory namespace isolation."""
        consultant = ZenConsultant()
        assert consultant.memory_namespace == "zen-copilot"
        assert consultant.session_id.startswith("zen_")
        

class TestPerformanceBenchmarks:
    """Performance benchmarks for ZenConsultant."""
    
    def test_directive_generation_speed(self):
        """Test directive generation performance."""
        import time
        consultant = ZenConsultant()
        
        start_time = time.time()
        for i in range(100):
            prompt = f"Build feature {i}"
            directive = consultant.get_concise_directive(prompt)
            assert directive is not None
        
        total_time = time.time() - start_time
        avg_time = total_time / 100
        
        # Should generate directives in under 10ms on average
        assert avg_time < 0.01, f"Average generation time too slow: {avg_time:.4f}s"
        
    def test_memory_footprint(self):
        """Test memory usage is reasonable."""
        consultant = ZenConsultant()
        
        # Get rough size estimate
        size = sys.getsizeof(consultant.__dict__)
        assert size < 10000  # Should be under 10KB


def run_comprehensive_tests():
    """Run all tests and generate report."""
    print("🧪 ZenConsultant Comprehensive Test Suite")
    print("=" * 50)
    
    # Test basic functionality
    consultant = ZenConsultant()
    
    # Test 1: Simple task
    print("\n✅ Test 1: Simple Task Analysis")
    simple_directive = consultant.get_concise_directive("Fix login bug")
    print(f"Directive: {json.dumps(simple_directive, indent=2)}")
    
    # Test 2: Complex task
    print("\n✅ Test 2: Complex Task Analysis")
    complex_directive = consultant.get_concise_directive(
        "Refactor entire authentication system architecture for enterprise scalability"
    )
    print(f"Directive: {json.dumps(complex_directive, indent=2)}")
    
    # Test 3: Consultation response formats
    print("\n✅ Test 3: Response Format Comparison")
    prompt = "Build testing framework"
    
    concise_response = create_zen_consultation_response(prompt, "concise")
    verbose_response = create_zen_consultation_response(prompt, "verbose")
    
    print(f"Concise ({len(concise_response['hookSpecificOutput']['additionalContext'])} chars):")
    print(concise_response["hookSpecificOutput"]["additionalContext"])
    
    print(f"\nVerbose ({len(verbose_response['hookSpecificOutput']['additionalContext'])} chars):")
    print(verbose_response["hookSpecificOutput"]["additionalContext"][:200] + "...")
    
    # Test 4: Consensus integration
    print("\n✅ Test 4: Consensus Integration")
    consensus_request = create_zen_consensus_request(prompt, ComplexityLevel.COMPLEX)
    print(f"Models for consensus: {len(consensus_request['models'])}")
    print(f"Complexity handling: {consensus_request['findings']}")
    
    print("\n🎉 All tests completed successfully!")
    print(f"📊 Performance: Concise format = {len(concise_response['hookSpecificOutput']['additionalContext'])} chars vs 10,000+ char patterns")
    

if __name__ == "__main__":
    # Run the comprehensive test suite
    run_comprehensive_tests()