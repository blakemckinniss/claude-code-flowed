#!/usr/bin/env python3
"""Test suite for MultiModelConsensusValidator - Phase 3 ZEN Integration.

This test suite validates the multi-model consensus validation functionality,
including weighted voting, parallel execution, token budget management,
and integration with the existing pre-tool validation system.
"""

import asyncio
import json
import time
import unittest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from multi_model_consensus_validator import (
    MultiModelConsensusValidator,
    ConsensusResult,
    ConsensusConfidence,
    ModelResponse,
    ModelConfig,
    ModelCapability
)
from ..core.workflow_validator import (
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class TestMultiModelConsensusValidator(unittest.TestCase):
    """Test cases for MultiModelConsensusValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = MultiModelConsensusValidator(priority=800)
        self.context = WorkflowContextTracker()
        
        # Mock tool calls for testing
        self.critical_tool_call = {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/critical/system/config.py",
                "content": "DATABASE_URL = 'production://sensitive'"
            }
        }
        
        self.safe_tool_call = {
            "tool_name": "Read",
            "tool_input": {
                "file_path": "/docs/README.md"
            }
        }
        
        self.high_risk_tool_call = {
            "tool_name": "Bash",
            "tool_input": {
                "command": "rm -rf /tmp/cache && sudo systemctl restart nginx"
            }
        }
    
    def test_validator_initialization(self):
        """Test validator initialization and configuration."""
        validator = MultiModelConsensusValidator()
        
        # Check basic properties
        self.assertEqual(validator.get_validator_name(), "multi_model_consensus_validator")
        self.assertEqual(validator.priority, 800)  # High priority for critical decisions
        self.assertTrue(validator.enabled)
        
        # Check default configuration
        config = validator.get_config()
        self.assertTrue(config["enabled"])
        self.assertEqual(config["priority"], 800)
        
        # Check model capabilities mapping
        self.assertEqual(
            validator.model_capabilities["openai/o3"], 
            ModelCapability.LOGIC_REASONING
        )
        self.assertEqual(
            validator.model_capabilities["google/gemini-2.5-flash"], 
            ModelCapability.FAST_ANALYSIS
        )
    
    def test_requires_consensus_validation(self):
        """Test logic for determining when consensus validation is needed."""
        
        # Critical tools should always require consensus
        self.assertTrue(
            self.validator._requires_consensus_validation(
                "Write", {"file_path": "test.py"}, self.context
            )
        )
        
        self.assertTrue(
            self.validator._requires_consensus_validation(
                "Bash", {"command": "ls -la"}, self.context
            )
        )
        
        # High-risk operations should require consensus
        high_risk_input = {"content": "DELETE FROM users WHERE 1=1;"}
        self.assertTrue(
            self.validator._requires_consensus_validation(
                "Edit", high_risk_input, self.context
            )
        )
        
        # Safe operations should not require consensus
        self.assertFalse(
            self.validator._requires_consensus_validation(
                "Read", {"file_path": "README.md"}, self.context
            )
        )
        
        # Operations when disconnected from ZEN should require consensus
        self.context.tools_since_zen = 10  # Simulate disconnection
        self.assertTrue(
            self.validator._requires_consensus_validation(
                "Edit", {"file_path": "test.py"}, self.context
            )
        )
    
    def test_auto_select_models(self):
        """Test automatic model selection based on task characteristics."""
        
        # Test GitHub operations
        github_call = {"tool_name": "mcp__github__create_pull_request", "tool_input": {}}
        models = self.validator._auto_select_models(github_call)
        
        self.assertIn("openai/o3", models)  # Always include O3 for logic
        self.assertIn("google/gemini-2.5-flash", models)  # Always include Flash for speed
        self.assertIn("anthropic/claude-opus-4", models)  # GitHub expertise
        
        # Test file operations
        file_call = {"tool_name": "Write", "tool_input": {}}
        models = self.validator._auto_select_models(file_call)
        
        self.assertIn("openai/o3", models)
        self.assertIn("google/gemini-2.5-flash", models)
        self.assertIn("google/gemini-2.5-pro", models)  # Content analysis
        
        # Test system commands
        bash_call = {"tool_name": "Bash", "tool_input": {}}
        models = self.validator._auto_select_models(bash_call)
        
        self.assertIn("deepseek/deepseek-r1-0528", models)  # Security analysis
        
        # Check model count limits
        self.validator.set_config_value("max_models", 2)
        models = self.validator._auto_select_models(github_call)
        self.assertLessEqual(len(models), 2)
    
    def test_create_model_configs(self):
        """Test creation of model configurations with proper weighting."""
        
        models = ["openai/o3", "google/gemini-2.5-flash", "anthropic/claude-opus-4"]
        configs = self.validator._create_model_configs(models, self.critical_tool_call)
        
        self.assertEqual(len(configs), 3)
        
        # Check config properties
        for config in configs:
            self.assertIsInstance(config, ModelConfig)
            self.assertEqual(config.stance, "neutral")
            self.assertGreater(config.weight, 0)
            self.assertGreater(config.max_tokens, 0)
            self.assertEqual(config.temperature, 0.2)
        
        # Check O3 has higher weight (ZEN preference)
        o3_config = next(c for c in configs if c.model_id == "openai/o3")
        flash_config = next(c for c in configs if c.model_id == "google/gemini-2.5-flash")
        self.assertGreater(o3_config.weight, flash_config.weight)
    
    def test_token_allocation(self):
        """Test token allocation based on model capabilities."""
        
        # Test logic reasoning gets more tokens
        logic_tokens = self.validator._allocate_tokens(ModelCapability.LOGIC_REASONING, 3)
        fast_tokens = self.validator._allocate_tokens(ModelCapability.FAST_ANALYSIS, 3)
        
        self.assertGreater(logic_tokens, fast_tokens)
        
        # Test total doesn't exceed budget
        total_allocated = logic_tokens + fast_tokens + self.validator._allocate_tokens(
            ModelCapability.TECHNICAL_REVIEW, 3
        )
        self.assertLess(total_allocated, self.validator.get_config_value("token_budget", 10000))
    
    def test_parse_model_response(self):
        """Test parsing of model responses."""
        
        # Test valid JSON response
        valid_response = json.dumps({
            "recommendation": "allow",
            "confidence": 0.85,
            "reasoning": "Operation appears safe",
            "risks": [],
            "alternatives": []
        })
        
        recommendation, confidence, reasoning, tokens = self.validator._parse_model_response(valid_response)
        
        self.assertEqual(recommendation, "allow")
        self.assertEqual(confidence, 0.85)
        self.assertEqual(reasoning, "Operation appears safe")
        self.assertGreater(tokens, 0)
        
        # Test invalid JSON response
        invalid_response = "This is not JSON"
        recommendation, confidence, reasoning, tokens = self.validator._parse_model_response(invalid_response)
        
        self.assertEqual(recommendation, "error")
        self.assertEqual(confidence, 0.0)
        self.assertIn("Failed to parse", reasoning)
    
    def test_analyze_consensus(self):
        """Test consensus analysis with weighted voting."""
        
        # Create mock responses
        responses = [
            ModelResponse(
                model_id="openai/o3",
                recommendation="allow",
                confidence=0.9,
                reasoning="Safe operation",
                execution_time=1.0,
                token_usage=100,
                success=True
            ),
            ModelResponse(
                model_id="google/gemini-2.5-flash",
                recommendation="allow",
                confidence=0.8,
                reasoning="Appears safe",
                execution_time=0.5,
                token_usage=80,
                success=True
            ),
            ModelResponse(
                model_id="anthropic/claude-opus-4",
                recommendation="warn",
                confidence=0.7,
                reasoning="Some risks identified",
                execution_time=1.2,
                token_usage=120,
                success=True
            )
        ]
        
        consensus_result = self.validator._analyze_consensus(responses, self.critical_tool_call)
        
        # Check consensus properties
        self.assertIsInstance(consensus_result, ConsensusResult)
        self.assertIn(consensus_result.recommendation, ["allow", "warn", "block", "suggest"])
        self.assertGreaterEqual(consensus_result.confidence_score, 0.0)
        self.assertLessEqual(consensus_result.confidence_score, 1.0)
        self.assertEqual(len(consensus_result.model_responses), 3)
        self.assertEqual(consensus_result.total_tokens, 300)
    
    def test_determine_consensus_level(self):
        """Test consensus confidence level determination."""
        
        # Test unanimous consensus
        unanimous_score = 1.0
        level = self.validator._determine_consensus_level(unanimous_score)
        self.assertEqual(level, ConsensusConfidence.UNANIMOUS)
        
        # Test strong consensus
        strong_score = 0.85
        level = self.validator._determine_consensus_level(strong_score)
        self.assertEqual(level, ConsensusConfidence.STRONG)
        
        # Test moderate consensus
        moderate_score = 0.65
        level = self.validator._determine_consensus_level(moderate_score)
        self.assertEqual(level, ConsensusConfidence.MODERATE)
        
        # Test divided consensus
        divided_score = 0.3
        level = self.validator._determine_consensus_level(divided_score)
        self.assertEqual(level, ConsensusConfidence.DIVIDED)
    
    def test_get_fallback_strategy(self):
        """Test fallback strategy selection."""
        
        # Test timeout fallback
        timeout_strategy = self.validator._get_fallback_strategy("Model timeout occurred")
        self.assertIn("timeout", timeout_strategy.lower())
        
        # Test unavailable fallback
        unavailable_strategy = self.validator._get_fallback_strategy("Model unavailable")
        self.assertIn("available", unavailable_strategy.lower())
        
        # Test general fallback
        general_strategy = self.validator._get_fallback_strategy("Unknown error")
        self.assertIn("manual", general_strategy.lower())
    
    def test_generate_validation_result(self):
        """Test generation of ValidationResult from consensus."""
        
        # Test blocking consensus
        blocking_consensus = ConsensusResult(
            consensus_reached=True,
            confidence_score=0.9,
            consensus_level=ConsensusConfidence.STRONG,
            recommendation="block",
            dissenting_models=[],
            tool_name="Write",
            validation_context="Test"
        )
        
        result = self.validator._generate_validation_result(
            blocking_consensus, "Write", {"file_path": "test.py"}
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.severity, ValidationSeverity.BLOCK)
        self.assertIn("BLOCK OPERATION", result.message)
        
        # Test warning consensus
        warning_consensus = ConsensusResult(
            consensus_reached=True,
            confidence_score=0.7,
            consensus_level=ConsensusConfidence.MODERATE,
            recommendation="warn",
            dissenting_models=["model_x"],
            tool_name="Write",
            validation_context="Test"
        )
        
        result = self.validator._generate_validation_result(
            warning_consensus, "Write", {"file_path": "test.py"}
        )
        
        self.assertEqual(result.severity, ValidationSeverity.WARN)
        self.assertIn("PROCEED WITH CAUTION", result.message)
        
        # Test allow consensus (should return None)
        allow_consensus = ConsensusResult(
            consensus_reached=True,
            confidence_score=0.8,
            consensus_level=ConsensusConfidence.STRONG,
            recommendation="allow",
            dissenting_models=[],
            tool_name="Read",
            validation_context="Test"
        )
        
        result = self.validator._generate_validation_result(
            allow_consensus, "Read", {"file_path": "test.py"}
        )
        
        self.assertIsNone(result)
    
    def test_handle_no_consensus(self):
        """Test handling of cases where no consensus is reached."""
        
        no_consensus = ConsensusResult(
            consensus_reached=False,
            confidence_score=0.4,
            consensus_level=ConsensusConfidence.DIVIDED,
            recommendation="mixed",
            dissenting_models=["model_a", "model_b"],
            model_responses=[Mock(), Mock(), Mock()],
            fallback_strategy="Manual review recommended",
            tool_name="Write",
            validation_context="Test"
        )
        
        result = self.validator._handle_no_consensus(no_consensus, "Write")
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.severity, ValidationSeverity.WARN)
        self.assertIn("NO AGREEMENT REACHED", result.message)
        self.assertIn("2/3 dissenting", result.message)
        self.assertIn("Manual review recommended", result.hive_guidance)
    
    @patch('asyncio.run')
    def test_validate_workflow_impl(self, mock_asyncio_run):
        """Test the main validation workflow implementation."""
        
        # Mock consensus result
        mock_consensus = ConsensusResult(
            consensus_reached=True,
            confidence_score=0.8,
            consensus_level=ConsensusConfidence.STRONG,
            recommendation="warn",
            dissenting_models=[],
            tool_name="Write",
            validation_context="Test"
        )
        
        mock_asyncio_run.return_value = mock_consensus
        
        # Test validation of critical tool
        result = self.validator._validate_workflow_impl(
            "Write", {"file_path": "test.py"}, self.context
        )
        
        # Should trigger consensus validation
        mock_asyncio_run.assert_called_once()
        self.assertIsInstance(result, ValidationResult)
        
        # Test validation of non-critical tool
        mock_asyncio_run.reset_mock()
        result = self.validator._validate_workflow_impl(
            "Read", {"file_path": "test.py"}, self.context
        )
        
        # Should not trigger consensus validation
        mock_asyncio_run.assert_not_called()
        self.assertIsNone(result)
    
    def test_configuration_management(self):
        """Test validator configuration management."""
        
        # Test getting configuration
        config = self.validator.get_config()
        self.assertIn("enabled", config)
        self.assertIn("priority", config)
        self.assertIn("config", config)
        
        # Test setting configuration
        new_config = {
            "enabled": False,
            "priority": 900,
            "config": {
                "max_models": 5,
                "timeout_seconds": 60
            }
        }
        
        self.validator.set_config(new_config)
        
        self.assertFalse(self.validator.enabled)
        self.assertEqual(self.validator.priority, 900)
        self.assertEqual(self.validator.get_config_value("max_models"), 5)
        self.assertEqual(self.validator.get_config_value("timeout_seconds"), 60)
    
    def test_security_focused_validation(self):
        """Test security-focused aspects of consensus validation."""
        
        # Test high-risk operation detection
        high_risk_input = {
            "command": "sudo rm -rf /var/log/secure && history -c"
        }
        
        requires_consensus = self.validator._requires_consensus_validation(
            "Bash", high_risk_input, self.context
        )
        
        self.assertTrue(requires_consensus)
        
        # Test sensitive file detection
        sensitive_file_input = {
            "file_path": "/etc/passwd",
            "content": "root:x:0:0:root:/root:/bin/bash"
        }
        
        requires_consensus = self.validator._requires_consensus_validation(
            "Write", sensitive_file_input, self.context
        )
        
        self.assertTrue(requires_consensus)
    
    def test_integration_with_zen_patterns(self):
        """Test integration with ZEN's established patterns."""
        
        # Test model selection follows ZEN preferences
        models = self.validator._auto_select_models(self.critical_tool_call)
        
        # Should always include O3 (ZEN's preferred logic model)
        self.assertIn("openai/o3", models)
        
        # Should include Gemini Flash for fast analysis
        self.assertIn("google/gemini-2.5-flash", models)
        
        # Test token budget management
        total_budget = self.validator.get_config_value("token_budget", 10000)
        self.assertGreater(total_budget, 5000)  # Reasonable budget
        self.assertLess(total_budget, 20000)    # Not excessive
        
        # Test consensus thresholds are reasonable
        moderate_threshold = self.validator.consensus_thresholds[ConsensusConfidence.MODERATE]
        self.assertGreaterEqual(moderate_threshold, 0.5)  # At least majority
        self.assertLessEqual(moderate_threshold, 0.8)     # Not too strict


class TestConsensusResultDataclass(unittest.TestCase):
    """Test the ConsensusResult dataclass."""
    
    def test_consensus_result_creation(self):
        """Test creation and properties of ConsensusResult."""
        
        result = ConsensusResult(
            consensus_reached=True,
            confidence_score=0.85,
            consensus_level=ConsensusConfidence.STRONG,
            recommendation="allow",
            dissenting_models=["model_x"],
            tool_name="Write",
            validation_context="Test validation"
        )
        
        # Check basic properties
        self.assertTrue(result.consensus_reached)
        self.assertEqual(result.confidence_score, 0.85)
        self.assertEqual(result.consensus_level, ConsensusConfidence.STRONG)
        self.assertEqual(result.recommendation, "allow")
        self.assertEqual(result.dissenting_models, ["model_x"])
        
        # Check auto-generated fields
        self.assertIsNotNone(result.timestamp)
        self.assertEqual(result.execution_time, 0.0)
        self.assertEqual(result.total_tokens, 0)
        self.assertEqual(result.model_responses, [])


class TestModelResponseDataclass(unittest.TestCase):
    """Test the ModelResponse dataclass."""
    
    def test_model_response_creation(self):
        """Test creation and properties of ModelResponse."""
        
        response = ModelResponse(
            model_id="openai/o3",
            recommendation="allow",
            confidence=0.9,
            reasoning="Operation is safe",
            execution_time=1.5,
            token_usage=150,
            success=True,
            error_message=None
        )
        
        # Check all properties
        self.assertEqual(response.model_id, "openai/o3")
        self.assertEqual(response.recommendation, "allow")
        self.assertEqual(response.confidence, 0.9)
        self.assertEqual(response.reasoning, "Operation is safe")
        self.assertEqual(response.execution_time, 1.5)
        self.assertEqual(response.token_usage, 150)
        self.assertTrue(response.success)
        self.assertIsNone(response.error_message)


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)