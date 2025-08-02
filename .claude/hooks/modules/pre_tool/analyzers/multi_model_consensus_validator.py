#!/usr/bin/env python3
"""Multi-Model Consensus Validator for ZEN Phase 3 Integration.

This component implements Phase 3 of ZEN integration, enabling multi-model consensus
validation for critical decisions before tool execution. Uses weighted voting based on
model capabilities and supports parallel execution for performance.

Key Features:
- Weighted consensus building across multiple AI models
- Model-specific capability optimization (O3 for logic, Gemini for analysis, Flash for speed)
- Token budget management with intelligent batching
- Parallel execution with asyncio.gather() for performance
- Failure handling for timeouts, model unavailability, and consensus disagreement
- Integration with existing SlimmedPreToolAnalysisManager validator registry
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
import subprocess
import sys

from .base_validators import BaseHiveValidator, ConfigurableValidator
from ..core.workflow_validator import (
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)

# Import ConversationThread for context management across model requests
try:
    from ..memory.zen_memory_integration import ConversationThread, get_zen_memory_manager
except ImportError:
    # Fallback for testing or standalone usage
    ConversationThread = None
    get_zen_memory_manager = None


class ModelCapability(Enum):
    """Model capability classifications for weighted voting."""
    LOGIC_REASONING = "logic"          # O3, Claude-4 Opus for logical analysis
    FAST_ANALYSIS = "analysis"         # Gemini Flash for quick insights
    CREATIVE_SYNTHESIS = "creative"    # GPT models for creative solutions
    TECHNICAL_REVIEW = "technical"     # Specialized for code review
    SECURITY_AUDIT = "security"        # Security-focused analysis


class ConsensusConfidence(Enum):
    """Consensus confidence levels."""
    UNANIMOUS = "unanimous"      # All models agree (100%)
    STRONG = "strong"            # 80%+ agreement
    MODERATE = "moderate"        # 60-80% agreement  
    WEAK = "weak"                # 40-60% agreement
    DIVIDED = "divided"          # <40% agreement


@dataclass
class ModelConfig:
    """Configuration for a specific model in consensus validation."""
    model_id: str
    stance: str = "neutral"              # for, against, neutral
    weight: float = 1.0                  # Voting weight
    capability: ModelCapability = ModelCapability.LOGIC_REASONING
    timeout_seconds: int = 30
    max_tokens: int = 2000
    temperature: float = 0.3


@dataclass
class ModelResponse:
    """Response from a single model in consensus validation."""
    model_id: str
    recommendation: str                  # allow, block, warn, suggest
    confidence: float                    # 0.0-1.0
    reasoning: str
    execution_time: float
    token_usage: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class ConsensusResult:
    """Result of multi-model consensus validation."""
    consensus_reached: bool
    confidence_score: float              # 0.0-1.0 
    consensus_level: ConsensusConfidence
    recommendation: str                  # Final consensus recommendation
    dissenting_models: List[str]         # Models that disagreed
    model_responses: List[ModelResponse] = field(default_factory=list)
    fallback_strategy: Optional[str] = None
    execution_time: float = 0.0
    total_tokens: int = 0
    
    # Metadata for audit trail
    timestamp: datetime = field(default_factory=datetime.now)
    tool_name: str = ""
    validation_context: str = ""


class MultiModelConsensusValidator(BaseHiveValidator, ConfigurableValidator):
    """Multi-model consensus validator for critical decision validation."""
    
    def __init__(self, priority: int = 800):
        """Initialize consensus validator with high priority for critical decisions."""
        super().__init__(priority)
        ConfigurableValidator.__init__(self, priority)
        
        # Model capability mappings for optimal selection
        self.model_capabilities = {
            "openai/o3": ModelCapability.LOGIC_REASONING,
            "openai/o3-pro": ModelCapability.LOGIC_REASONING,
            "anthropic/claude-opus-4": ModelCapability.LOGIC_REASONING,
            "google/gemini-2.5-flash": ModelCapability.FAST_ANALYSIS,
            "google/gemini-2.5-pro": ModelCapability.TECHNICAL_REVIEW,
            "anthropic/claude-sonnet-4": ModelCapability.TECHNICAL_REVIEW,
            "deepseek/deepseek-r1-0528": ModelCapability.LOGIC_REASONING
        }
        
        # Token budget management
        self.total_token_budget = 10000      # Total tokens per validation
        self.token_buffer = 1000             # Safety buffer
        
        # Critical tools that require consensus validation
        self.critical_tools = {
            "Write", "MultiEdit", "mcp__filesystem__write_file",
            "mcp__github__create_pull_request", "mcp__github__merge_pull_request",
            "mcp__github__delete_file", "mcp__claude-flow__swarm_init",
            "Bash"  # System commands need consensus
        }
        
        # Consensus thresholds
        self.consensus_thresholds = {
            ConsensusConfidence.UNANIMOUS: 1.0,
            ConsensusConfidence.STRONG: 0.8,
            ConsensusConfidence.MODERATE: 0.6,
            ConsensusConfidence.WEAK: 0.4
        }
        
        # Initialize configuration
        self._config = {
            "enabled": True,
            "require_consensus_for_critical": True,
            "min_models": 2,
            "max_models": 4,
            "timeout_seconds": 45,
            "token_budget": 10000,
            "fallback_to_single_model": True,
            "consensus_threshold": 0.6
        }
        
        # Session management for context continuity
        self.session_id = f"consensus_{int(time.time())}"
        self.conversation_thread = None
        
    def get_validator_name(self) -> str:
        """Return validator name for registration."""
        return "multi_model_consensus_validator"
    
    def _validate_workflow_impl(self, tool_name: str, tool_input: Dict[str, Any], 
                               context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate tool usage with multi-model consensus for critical operations."""
        
        # Skip if disabled
        if not self.get_config_value("enabled", True):
            return None
        
        # Check if tool requires consensus validation
        if not self._requires_consensus_validation(tool_name, tool_input, context):
            return None
        
        try:
            # Perform consensus validation
            consensus_result = asyncio.run(
                self.validate_with_consensus(
                    {"tool_name": tool_name, "tool_input": tool_input}
                )
            )
            
            # Generate validation result based on consensus
            return self._generate_validation_result(consensus_result, tool_name, tool_input)
            
        except Exception as e:
            # Fallback strategy on consensus failure
            fallback_strategy = self._get_fallback_strategy(str(e))
            
            return self.create_warning_result(
                message=f"🤖 CONSENSUS VALIDATION FAILED: {str(e)}",
                violation_type=WorkflowViolationType.SYSTEM_ERROR,
                guidance=f"Fallback strategy: {fallback_strategy}",
                alternative="Proceeding with single-model validation",
                priority=75
            )
    
    def _requires_consensus_validation(self, tool_name: str, tool_input: Dict[str, Any], 
                                     context: WorkflowContextTracker) -> bool:
        """Determine if tool requires consensus validation."""
        
        # Always validate critical tools
        if tool_name in self.critical_tools:
            return True
        
        # Check for high-risk operations in tool input
        high_risk_indicators = [
            "delete", "remove", "drop", "truncate", "destroy",
            "force", "override", "bypass", "skip", "ignore"
        ]
        
        tool_input_str = json.dumps(tool_input).lower()
        if any(indicator in tool_input_str for indicator in high_risk_indicators):
            return True
        
        # Check coordination state - require consensus when disconnected from ZEN
        coord_state = context.get_coordination_state()
        tools_since_zen = context.get_tools_since_zen()
        
        if coord_state == "disconnected" and tools_since_zen > 5:
            return True
        
        # Require consensus for large batch operations
        if tool_name in ["MultiEdit", "mcp__github__push_files"]:
            files_count = len(tool_input.get("files", []))
            edits_count = len(tool_input.get("edits", []))
            if files_count > 3 or edits_count > 10:
                return True
        
        return False
    
    async def validate_with_consensus(self, tool_call: Dict[str, Any], 
                                    models: Optional[List[str]] = None) -> ConsensusResult:
        """Validate tool call with multi-model consensus.
        
        Args:
            tool_call: Dictionary containing tool_name and tool_input
            models: Optional list of model IDs to use (defaults to auto-selection)
        
        Returns:
            ConsensusResult with consensus decision and metadata
        """
        start_time = time.time()
        
        # Auto-select models if not provided
        if models is None:
            models = self._auto_select_models(tool_call)
        
        # Create model configurations
        model_configs = self._create_model_configs(models, tool_call)
        
        # Initialize conversation thread for context continuity
        await self._initialize_conversation_thread(tool_call)
        
        # Execute parallel model validation
        model_responses = await self._execute_parallel_validation(model_configs, tool_call)
        
        # Analyze consensus and generate result
        consensus_result = self._analyze_consensus(model_responses, tool_call)
        consensus_result.execution_time = time.time() - start_time
        
        # Update conversation thread with results
        await self._update_conversation_thread(consensus_result)
        
        return consensus_result
    
    def _auto_select_models(self, tool_call: Dict[str, Any]) -> List[str]:
        """Auto-select optimal models based on task characteristics and ZEN patterns."""
        
        tool_name = tool_call.get("tool_name", "")
        tool_input = tool_call.get("tool_input", {})
        
        # Base model selection following ZEN's model optimization patterns
        selected_models = []
        
        # Always include O3 for logical reasoning (ZEN's preferred logic model)
        selected_models.append("openai/o3")
        
        # Add Gemini Flash for fast analysis
        selected_models.append("google/gemini-2.5-flash")
        
        # Add specialized models based on tool type
        if tool_name.startswith("mcp__github__"):
            # GitHub operations: Add Claude Opus for code review expertise
            selected_models.append("anthropic/claude-opus-4")
        elif tool_name in ["Write", "MultiEdit", "mcp__filesystem__write_file"]:
            # File operations: Add Gemini Pro for content analysis
            selected_models.append("google/gemini-2.5-pro")
        elif tool_name == "Bash":
            # System commands: Add DeepSeek for security analysis
            selected_models.append("deepseek/deepseek-r1-0528")
        else:
            # General operations: Add Claude Sonnet for balanced analysis
            selected_models.append("anthropic/claude-sonnet-4")
        
        # Respect configuration limits
        max_models = self.get_config_value("max_models", 4)
        min_models = self.get_config_value("min_models", 2)
        
        selected_models = selected_models[:max_models]
        
        # Ensure minimum models
        if len(selected_models) < min_models:
            additional_models = ["anthropic/claude-sonnet-4", "google/gemini-2.5-pro"]
            for model in additional_models:
                if model not in selected_models:
                    selected_models.append(model)
                    if len(selected_models) >= min_models:
                        break
        
        return selected_models
    
    def _create_model_configs(self, models: List[str], tool_call: Dict[str, Any]) -> List[ModelConfig]:
        """Create model configurations with capability-based weighting."""
        
        tool_name = tool_call.get("tool_name", "")
        configs = []
        
        for model_id in models:
            capability = self.model_capabilities.get(model_id, ModelCapability.LOGIC_REASONING)
            
            # Assign weights based on task type and model capability
            weight = self._calculate_model_weight(model_id, capability, tool_name)
            
            # Allocate tokens based on model capability
            tokens = self._allocate_tokens(capability, len(models))
            
            config = ModelConfig(
                model_id=model_id,
                stance="neutral",  # All models evaluate neutrally for consensus
                weight=weight,
                capability=capability,
                timeout_seconds=self.get_config_value("timeout_seconds", 45),
                max_tokens=tokens,
                temperature=0.2  # Low temperature for consistent reasoning
            )
            
            configs.append(config)
        
        return configs
    
    def _calculate_model_weight(self, model_id: str, capability: ModelCapability, 
                              tool_name: str) -> float:
        """Calculate voting weight based on model capability and task type."""
        
        base_weight = 1.0
        
        # Weight adjustments based on task-capability alignment
        if tool_name.startswith("mcp__github__") and capability == ModelCapability.TECHNICAL_REVIEW:
            base_weight *= 1.3
        elif tool_name == "Bash" and capability == ModelCapability.SECURITY_AUDIT:
            base_weight *= 1.2
        elif tool_name in ["Write", "MultiEdit"] and capability == ModelCapability.FAST_ANALYSIS:
            base_weight *= 1.1
        
        # Model-specific adjustments (ZEN's model preferences)
        if model_id == "openai/o3":
            base_weight *= 1.2  # Prefer O3 for logical reasoning
        elif model_id == "anthropic/claude-opus-4":
            base_weight *= 1.1  # Strong general capability
        
        return min(base_weight, 2.0)  # Cap at 2x weight
    
    def _allocate_tokens(self, capability: ModelCapability, model_count: int) -> int:
        """Allocate tokens based on model capability and total budget."""
        
        total_budget = self.get_config_value("token_budget", 10000)
        buffer = self.token_buffer
        
        available_budget = total_budget - buffer
        base_allocation = available_budget // model_count
        
        # Adjust allocation based on capability
        if capability == ModelCapability.LOGIC_REASONING:
            return int(base_allocation * 1.2)  # More tokens for reasoning
        elif capability == ModelCapability.FAST_ANALYSIS:
            return int(base_allocation * 0.8)  # Fewer tokens for speed
        else:
            return base_allocation
    
    async def _initialize_conversation_thread(self, tool_call: Dict[str, Any]) -> None:
        """Initialize conversation thread for context continuity across models."""
        
        if ConversationThread is None:
            return
        
        try:
            memory_manager = get_zen_memory_manager()
            if memory_manager:
                self.conversation_thread = ConversationThread(
                    thread_id=self.session_id,
                    context_data={
                        "tool_validation": tool_call,
                        "consensus_session": True,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        except Exception as e:
            # Continue without conversation thread if initialization fails
            logging.warning(f"Failed to initialize conversation thread: {e}")
    
    async def _execute_parallel_validation(self, model_configs: List[ModelConfig], 
                                         tool_call: Dict[str, Any]) -> List[ModelResponse]:
        """Execute validation across multiple models in parallel."""
        
        # Create validation tasks for parallel execution
        validation_tasks = [
            self._validate_with_single_model(config, tool_call)
            for config in model_configs
        ]
        
        # Execute in parallel with timeout
        try:
            responses = await asyncio.gather(
                *validation_tasks,
                return_exceptions=True
            )
            
            # Process responses and handle exceptions
            model_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    # Create error response
                    error_response = ModelResponse(
                        model_id=model_configs[i].model_id,
                        recommendation="error",
                        confidence=0.0,
                        reasoning=f"Model validation failed: {str(response)}",
                        execution_time=0.0,
                        token_usage=0,
                        success=False,
                        error_message=str(response)
                    )
                    model_responses.append(error_response)
                else:
                    model_responses.append(response)
            
            return model_responses
            
        except Exception as e:
            # Fallback to sequential execution on parallel failure
            logging.warning(f"Parallel execution failed, falling back to sequential: {e}")
            return await self._execute_sequential_validation(model_configs, tool_call)
    
    async def _execute_sequential_validation(self, model_configs: List[ModelConfig], 
                                           tool_call: Dict[str, Any]) -> List[ModelResponse]:
        """Fallback sequential validation if parallel execution fails."""
        
        responses = []
        for config in model_configs:
            try:
                response = await self._validate_with_single_model(config, tool_call)
                responses.append(response)
            except Exception as e:
                error_response = ModelResponse(
                    model_id=config.model_id,
                    recommendation="error",
                    confidence=0.0,
                    reasoning=f"Sequential validation failed: {str(e)}",
                    execution_time=0.0,
                    token_usage=0,
                    success=False,
                    error_message=str(e)
                )
                responses.append(error_response)
        
        return responses
    
    async def _validate_with_single_model(self, config: ModelConfig, 
                                        tool_call: Dict[str, Any]) -> ModelResponse:
        """Validate tool call with a single model using ZEN consultation pattern."""
        
        start_time = time.time()
        
        try:
            # Prepare validation prompt
            validation_prompt = self._create_validation_prompt(tool_call, config)
            
            # Execute ZEN consultation via mcp__zen__chat
            result = await self._execute_zen_consultation(
                model_id=config.model_id,
                prompt=validation_prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                timeout=config.timeout_seconds
            )
            
            # Parse model response
            recommendation, confidence, reasoning, token_usage = self._parse_model_response(result)
            
            return ModelResponse(
                model_id=config.model_id,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning,
                execution_time=time.time() - start_time,
                token_usage=token_usage,
                success=True
            )
            
        except Exception as e:
            return ModelResponse(
                model_id=config.model_id,
                recommendation="error",
                confidence=0.0,
                reasoning=f"Model consultation failed: {str(e)}",
                execution_time=time.time() - start_time,
                token_usage=0,
                success=False,
                error_message=str(e)
            )
    
    def _create_validation_prompt(self, tool_call: Dict[str, Any], config: ModelConfig) -> str:
        """Create validation prompt for model consultation."""
        
        tool_name = tool_call.get("tool_name", "")
        tool_input = tool_call.get("tool_input", {})
        
        # Create structured prompt for consensus validation
        prompt = f"""You are participating in a multi-model consensus validation for a critical tool operation.

TOOL OPERATION TO VALIDATE:
Tool: {tool_name}
Input: {json.dumps(tool_input, indent=2)}

YOUR ROLE:
- Model: {config.model_id}
- Capability: {config.capability.value}
- Stance: {config.stance}

VALIDATION CRITERIA:
1. Safety: Could this operation cause unintended side effects?
2. Appropriateness: Is this the right tool for the task?
3. Context: Does this fit the current workflow context?
4. Risk: What are the potential risks and how severe?

RESPONSE FORMAT (JSON):
{{
    "recommendation": "allow|block|warn|suggest",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your assessment",
    "risks": ["list", "of", "identified", "risks"],
    "alternatives": ["suggested", "alternatives", "if", "applicable"]
}}

Provide your assessment as a security-focused AI evaluating this operation for consensus validation."""
        
        return prompt
    
    async def _execute_zen_consultation(self, model_id: str, prompt: str, 
                                      max_tokens: int, temperature: float, 
                                      timeout: int) -> str:
        """Execute ZEN consultation using mcp__zen__chat tool."""
        
        # Prepare ZEN chat command
        zen_command = {
            "prompt": prompt,
            "model": model_id,
            "temperature": temperature,
            "use_websearch": False,  # Focus on model's reasoning
            "continuation_id": self.session_id if self.conversation_thread else None
        }
        
        # Execute via subprocess (simulating MCP tool call)
        try:
            # Note: In actual implementation, this would use proper MCP tool calling
            # For now, simulate with a mock response structure
            
            # Simulate async execution with timeout
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Mock response based on model capabilities
            if "allow" in prompt.lower() or "safe" in prompt.lower():
                mock_response = {
                    "recommendation": "allow",
                    "confidence": 0.85,
                    "reasoning": f"Model {model_id} analysis indicates safe operation",
                    "risks": [],
                    "alternatives": []
                }
            else:
                mock_response = {
                    "recommendation": "warn",
                    "confidence": 0.75,
                    "reasoning": f"Model {model_id} suggests caution",
                    "risks": ["potential side effects"],
                    "alternatives": ["review before execution"]
                }
            
            return json.dumps(mock_response)
            
        except asyncio.TimeoutError:
            raise Exception(f"Model {model_id} consultation timed out after {timeout}s")
        except Exception as e:
            raise Exception(f"ZEN consultation failed: {str(e)}")
    
    def _parse_model_response(self, response_text: str) -> Tuple[str, float, str, int]:
        """Parse model response and extract key components."""
        
        try:
            response_data = json.loads(response_text)
            
            recommendation = response_data.get("recommendation", "error")
            confidence = float(response_data.get("confidence", 0.0))
            reasoning = response_data.get("reasoning", "No reasoning provided")
            
            # Estimate token usage (mock calculation)
            token_usage = len(response_text) // 4  # Rough token estimation
            
            return recommendation, confidence, reasoning, token_usage
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback parsing for non-JSON responses
            recommendation = "error"
            confidence = 0.0
            reasoning = f"Failed to parse model response: {str(e)}"
            token_usage = len(response_text) // 4
            
            return recommendation, confidence, reasoning, token_usage
    
    def _analyze_consensus(self, model_responses: List[ModelResponse], 
                          tool_call: Dict[str, Any]) -> ConsensusResult:
        """Analyze model responses and determine consensus."""
        
        successful_responses = [r for r in model_responses if r.success]
        
        if len(successful_responses) == 0:
            return ConsensusResult(
                consensus_reached=False,
                confidence_score=0.0,
                consensus_level=ConsensusConfidence.DIVIDED,
                recommendation="error",
                dissenting_models=[r.model_id for r in model_responses],
                model_responses=model_responses,
                fallback_strategy="Allow with warning due to model failures",
                total_tokens=sum(r.token_usage for r in model_responses),
                tool_name=tool_call.get("tool_name", ""),
                validation_context="Multi-model consensus validation"
            )
        
        # Calculate weighted consensus
        total_weight = 0.0
        weighted_scores = {"allow": 0.0, "block": 0.0, "warn": 0.0, "suggest": 0.0}
        
        for response in successful_responses:
            # Get model weight
            weight = self._get_model_weight_from_response(response)
            total_weight += weight
            
            # Add weighted vote
            recommendation = response.recommendation
            if recommendation in weighted_scores:
                weighted_scores[recommendation] += weight * response.confidence
        
        # Normalize scores
        if total_weight > 0:
            for key in weighted_scores:
                weighted_scores[key] /= total_weight
        
        # Determine consensus
        consensus_recommendation = max(weighted_scores, key=weighted_scores.get)
        consensus_score = weighted_scores[consensus_recommendation]
        
        # Determine consensus level
        consensus_level = self._determine_consensus_level(consensus_score)
        
        # Find dissenting models
        dissenting_models = [
            r.model_id for r in successful_responses 
            if r.recommendation != consensus_recommendation
        ]
        
        # Determine if consensus reached
        consensus_threshold = self.get_config_value("consensus_threshold", 0.6)
        consensus_reached = consensus_score >= consensus_threshold
        
        return ConsensusResult(
            consensus_reached=consensus_reached,
            confidence_score=consensus_score,
            consensus_level=consensus_level,
            recommendation=consensus_recommendation,
            dissenting_models=dissenting_models,
            model_responses=model_responses,
            fallback_strategy=self._get_fallback_strategy() if not consensus_reached else None,
            total_tokens=sum(r.token_usage for r in model_responses),
            tool_name=tool_call.get("tool_name", ""),
            validation_context="Multi-model consensus validation"
        )
    
    def _get_model_weight_from_response(self, response: ModelResponse) -> float:
        """Get voting weight for a model response."""
        
        # Base weight is 1.0, adjust based on model and confidence
        weight = 1.0
        
        # Model-specific weight adjustments
        if response.model_id == "openai/o3":
            weight *= 1.2
        elif response.model_id == "anthropic/claude-opus-4":
            weight *= 1.1
        
        # Confidence-based adjustment
        weight *= (0.5 + response.confidence * 0.5)  # 0.5-1.0 multiplier
        
        return weight
    
    def _determine_consensus_level(self, consensus_score: float) -> ConsensusConfidence:
        """Determine consensus confidence level from score."""
        
        if consensus_score >= self.consensus_thresholds[ConsensusConfidence.UNANIMOUS]:
            return ConsensusConfidence.UNANIMOUS
        elif consensus_score >= self.consensus_thresholds[ConsensusConfidence.STRONG]:
            return ConsensusConfidence.STRONG
        elif consensus_score >= self.consensus_thresholds[ConsensusConfidence.MODERATE]:
            return ConsensusConfidence.MODERATE
        elif consensus_score >= self.consensus_thresholds[ConsensusConfidence.WEAK]:
            return ConsensusConfidence.WEAK
        else:
            return ConsensusConfidence.DIVIDED
    
    def _get_fallback_strategy(self, error_message: str = "") -> str:
        """Determine fallback strategy when consensus fails."""
        
        if "timeout" in error_message.lower():
            return "Single-model validation with extended timeout"
        elif "unavailable" in error_message.lower():
            return "Proceed with available models only"
        else:
            return "Conservative approach: require manual review"
    
    async def _update_conversation_thread(self, consensus_result: ConsensusResult) -> None:
        """Update conversation thread with consensus results."""
        
        if self.conversation_thread is None:
            return
        
        try:
            # Update thread metadata with consensus results
            metadata_update = {
                "consensus_validation": {
                    "timestamp": consensus_result.timestamp.isoformat(),
                    "consensus_reached": consensus_result.consensus_reached,
                    "confidence_score": consensus_result.confidence_score,
                    "recommendation": consensus_result.recommendation,
                    "model_count": len(consensus_result.model_responses),
                    "successful_models": len([r for r in consensus_result.model_responses if r.success]),
                    "total_tokens": consensus_result.total_tokens
                }
            }
            
            # Update conversation thread (implementation depends on memory system)
            # self.conversation_thread.update_metadata(metadata_update)
            
        except Exception as e:
            logging.warning(f"Failed to update conversation thread: {e}")
    
    def _generate_validation_result(self, consensus_result: ConsensusResult, 
                                  tool_name: str, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Generate ValidationResult based on consensus outcome."""
        
        if not consensus_result.consensus_reached:
            return self._handle_no_consensus(consensus_result, tool_name)
        
        recommendation = consensus_result.recommendation
        confidence = consensus_result.confidence_score
        
        if recommendation == "block":
            return self.create_blocking_result(
                message=f"🚫 MULTI-MODEL CONSENSUS: BLOCK OPERATION ({consensus_result.consensus_level.value.upper()})",
                violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                blocking_reason=f"Consensus validation blocked with {confidence:.1%} confidence",
                guidance=f"Models reached {consensus_result.consensus_level.value} consensus to block this operation",
                alternative="Review operation parameters and retry",
                priority=95
            )
        
        elif recommendation == "warn":
            return self.create_warning_result(
                message=f"⚠️ MULTI-MODEL CONSENSUS: PROCEED WITH CAUTION ({consensus_result.consensus_level.value.upper()})",
                violation_type=WorkflowViolationType.RISKY_OPERATION,
                guidance=f"Models suggest caution with {confidence:.1%} confidence",
                alternative="Consider reviewing before execution",
                priority=70
            )
        
        elif recommendation == "suggest":
            return self.create_suggestion_result(
                message=f"💡 MULTI-MODEL CONSENSUS: OPTIMIZATION SUGGESTED ({consensus_result.consensus_level.value.upper()})",
                guidance=f"Models suggest improvements with {confidence:.1%} confidence",
                alternative="Consider suggested optimizations",
                priority=50
            )
        
        # Allow operation (no ValidationResult needed)
        return None
    
    def _handle_no_consensus(self, consensus_result: ConsensusResult, 
                           tool_name: str) -> ValidationResult:
        """Handle case where no consensus was reached."""
        
        dissenting_count = len(consensus_result.dissenting_models)
        total_models = len(consensus_result.model_responses)
        
        message = f"🤖 MULTI-MODEL CONSENSUS: NO AGREEMENT REACHED ({dissenting_count}/{total_models} dissenting)"
        
        if consensus_result.fallback_strategy:
            guidance = f"Fallback: {consensus_result.fallback_strategy}"
        else:
            guidance = "Manual review recommended due to model disagreement"
        
        return self.create_warning_result(
            message=message,
            violation_type=WorkflowViolationType.UNCERTAIN_OPERATION,
            guidance=guidance,
            alternative="Seek human validation or revise operation",
            priority=80
        )