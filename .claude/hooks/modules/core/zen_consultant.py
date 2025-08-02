#!/usr/bin/env python3
"""ZEN Consultation System for Claude Code Directive Generation.

This module replaces verbose pattern matching with intelligent consultation
that analyzes user prompts and generates concise, actionable directives.
Integrates with existing hook system and provides memory-backed learning.
"""

import json
import subprocess
import sys
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class ComplexityLevel(Enum):
    """Task complexity levels for thinking mode allocation."""
    SIMPLE = "minimal"      # Simple tasks: 0.5% model capacity
    MEDIUM = "medium"       # Standard tasks: 33% model capacity  
    COMPLEX = "high"        # Complex tasks: 67% model capacity
    ENTERPRISE = "max"      # Enterprise tasks: 100% model capacity
    HIVE_REQUIRED = "hive"  # Multi-agent hive coordination required


class CoordinationType(Enum):
    """Coordination patterns for task execution."""
    HIVE = "HIVE"          # Hierarchical with Queen ZEN leadership
    SWARM = "SWARM"        # Mesh topology for collaborative tasks
    QUEEN_MODE = "QUEEN"   # Queen-only coordination without workers


@dataclass
class AgentAllocation:
    """Agent allocation specification."""
    count: int                    # 0-6 agents
    types: List[str]             # Agent type specifications
    topology: CoordinationType   # HIVE, SWARM, or QUEEN_MODE
    queen_role: Optional[str] = None    # Queen specialist role
    auto_scaling: bool = False   # Enable automatic scaling
    

@dataclass  
class ZenDirective:
    """Structured directive output from ZEN consultation."""
    coordination: CoordinationType
    agents: AgentAllocation
    mcp_tools: List[str]
    next_steps: List[str] 
    warnings: List[str]
    thinking_mode: str
    confidence: float
    session_id: str


class ZenConsultant:
    """Intelligent consultation system using ZEN for directive generation."""
    
    # Agent type mappings for different task categories
    AGENT_CATALOG = {
        "development": ["coder", "reviewer", "tester"],
        "architecture": ["system-architect", "backend-architect", "api-architect"],
        "analysis": ["code-archaeologist", "performance-analyzer", "security-auditor"],
        "documentation": ["documentation-specialist", "api-documenter"],
        "testing": ["tester", "qa-expert", "test-automator"],
        "refactoring": ["code-refactorer", "legacy-modernizer"],
        "debugging": ["debugger", "error-detective"],
        "deployment": ["deployment-engineer", "cicd-engineer"],
        "performance": ["performance-optimizer", "performance-engineer"],
        "security": ["security-auditor", "security-manager"],
        "mobile": ["mobile-developer", "ios-developer"],
        "data": ["data-engineer", "data-scientist", "ml-engineer"],
        "frontend": ["frontend-developer", "ui-ux-designer", "react-pro"],
        "backend": ["backend-developer", "database-admin"],
        "github": ["pr-manager", "issue-tracker", "code-review-swarm"]
    }
    
    # MCP tool recommendations based on task patterns
    MCP_TOOLS = {
        "coordination": ["mcp__claude-flow__swarm_init", "mcp__claude-flow__agent_spawn"],
        "analysis": ["mcp__zen__thinkdeep", "mcp__zen__analyze"], 
        "consensus": ["mcp__zen__consensus"],
        "memory": ["mcp__claude-flow__memory_usage"],
        "github": ["mcp__github__*"],
        "planning": ["mcp__zen__planner"],
        "testing": ["mcp__zen__testgen"],
        "debugging": ["mcp__zen__debug"],
        "security": ["mcp__zen__secaudit"],
        "documentation": ["mcp__zen__docgen"],
        "refactoring": ["mcp__zen__refactor"]
    }
    
    def __init__(self):
        """Initialize ZEN consultant with memory integration."""
        self.session_id = f"zen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.memory_namespace = "zen-copilot"
        self.learning_data = {}
        self._load_learning_patterns()
    
    def _load_learning_patterns(self):
        """Load learning patterns from memory for improved recommendations."""
        try:
            # Try to load existing patterns from memory namespace
            # This would normally integrate with the memory system
            self.learning_data = {
                "successful_patterns": {},
                "failed_patterns": {},
                "user_preferences": {},
                "complexity_adjustments": {}
            }
        except Exception:
            # Fallback to empty learning data
            self.learning_data = {
                "successful_patterns": {},
                "failed_patterns": {},
                "user_preferences": {},
                "complexity_adjustments": {}
            }
    
    def get_concise_directive(self, prompt: str) -> Dict[str, Any]:
        """Generate concise directive in required 200-character format."""
        complexity, metadata = self.analyze_prompt_complexity(prompt)
        coordination = self.determine_coordination_type(complexity, metadata["categories"], prompt)
        agents = self.allocate_initial_agents(complexity, metadata["categories"], coordination)
        mcp_tools = self.select_mcp_tools(metadata["categories"], coordination)
        
        # Calculate confidence based on pattern matching and learning
        confidence = self._calculate_confidence(metadata, complexity)
        
        # Generate concise directive structure
        return {
            "hive": coordination.value,
            "swarm": f"{agents.count} agents" if agents.count > 0 else "discovery phase",
            "agents": agents.types[:3],  # Limit to first 3 agents
            "tools": mcp_tools[:3],      # Limit to first 3 tools
            "confidence": round(confidence, 2),
            "session_id": self.session_id,
            "thinking_mode": complexity.value
        }
    
    def _calculate_confidence(self, metadata: Dict, complexity: ComplexityLevel) -> float:
        """Calculate confidence score based on analysis and learning."""
        base_confidence = 0.7  # Base confidence
        
        # Adjust based on complexity (simpler tasks = higher confidence)
        complexity_adjustment = {
            ComplexityLevel.SIMPLE: 0.2,
            ComplexityLevel.MEDIUM: 0.1,
            ComplexityLevel.COMPLEX: 0.0,
            ComplexityLevel.ENTERPRISE: -0.1
        }.get(complexity, 0.0)
        
        # Adjust based on categories (more categories = lower confidence)
        category_adjustment = max(0.0, 0.1 - len(metadata["categories"]) * 0.02)
        
        # Apply learning adjustments if available
        learning_adjustment = 0.0
        for category in metadata["categories"]:
            if category in self.learning_data.get("successful_patterns", {}):
                learning_adjustment += 0.05
        
        final_confidence = base_confidence + complexity_adjustment + category_adjustment + learning_adjustment
        return max(0.1, min(0.99, final_confidence))  # Clamp between 0.1 and 0.99
        
    def analyze_prompt_complexity(self, prompt: str) -> Tuple[ComplexityLevel, Dict[str, Any]]:
        """Analyze prompt complexity and extract task metadata."""
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())
        
        # Simple complexity indicators
        simple_indicators = ["fix", "update", "add", "remove", "change", "help"]
        # Complex complexity indicators  
        complex_indicators = ["refactor", "architecture", "system", "migrate", "enterprise", 
                            "scalable", "performance", "security", "audit", "review"]
        # Hive coordination indicators
        hive_indicators = ["orchestrate", "coordinate", "manage", "oversee", "multi-agent", 
                          "collective", "collaboration", "consensus", "queen", "hive"]
        
        simple_score = sum(1 for indicator in simple_indicators if indicator in prompt_lower)
        complex_score = sum(1 for indicator in complex_indicators if indicator in prompt_lower)
        hive_score = sum(1 for indicator in hive_indicators if indicator in prompt_lower)
        
        # Determine complexity with hive coordination detection
        if hive_score >= 1 or (complex_score >= 2 and word_count > 30):
            complexity = ComplexityLevel.HIVE_REQUIRED
        elif word_count < 10 and simple_score > complex_score:
            complexity = ComplexityLevel.SIMPLE
        elif word_count > 50 or complex_score >= 2:
            complexity = ComplexityLevel.ENTERPRISE if complex_score >= 3 else ComplexityLevel.COMPLEX
        else:
            complexity = ComplexityLevel.MEDIUM
            
        # Extract task categories
        categories = []
        for category, keywords in {
            "development": ["code", "implement", "build", "create"],
            "testing": ["test", "qa", "quality"],
            "debugging": ["debug", "fix", "error", "issue", "problem"],
            "architecture": ["architecture", "design", "system", "structure"],
            "refactoring": ["refactor", "clean", "improve", "optimize"],
            "security": ["security", "audit", "vulnerability", "secure"],
            "performance": ["performance", "speed", "optimize", "efficient"],
            "documentation": ["document", "docs", "readme", "guide"],
            "github": ["github", "pr", "pull request", "issue", "commit"],
            "deployment": ["deploy", "release", "production", "ci/cd"]
        }.items():
            if any(keyword in prompt_lower for keyword in keywords):
                categories.append(category)
        
        metadata = {
            "categories": categories,
            "word_count": word_count,
            "simple_score": simple_score,
            "complex_score": complex_score
        }
        
        return complexity, metadata
    
    def determine_coordination_type(self, complexity: ComplexityLevel, categories: List[str], prompt: str = "") -> CoordinationType:
        """Determine whether to use HIVE, SWARM, or QUEEN_MODE coordination based on task characteristics."""
        prompt_lower = prompt.lower()
        
        # Direct HIVE_REQUIRED complexity always uses HIVE
        if complexity == ComplexityLevel.HIVE_REQUIRED:
            return CoordinationType.HIVE
        
        # Queen-only indicators: Strategic, high-level, consultation-focused
        queen_indicators = {
            "strategic_keywords": ["strategy", "vision", "roadmap", "planning", "direction"],
            "consultation_keywords": ["advice", "consult", "recommend", "suggest", "guide"],
            "oversight_keywords": ["oversee", "supervise", "coordinate", "manage", "orchestrate"]
        }
        
        # HIVE indicators: Complex projects, persistent sessions, multi-feature work
        hive_indicators = {
            "project_keywords": ["project", "system", "architecture", "enterprise", "platform", "framework"],
            "persistence_keywords": ["maintain", "ongoing", "continuous", "long-term", "persistent", "resume"],
            "complexity_keywords": ["complex", "multi", "integration", "coordination", "orchestration"],
            "categories": ["architecture", "security", "performance", "deployment"]
        }
        
        # SWARM indicators: Quick tasks, single objectives, immediate execution  
        swarm_indicators = {
            "task_keywords": ["build", "fix", "create", "add", "update", "implement", "debug"],
            "immediacy_keywords": ["quick", "fast", "now", "immediate", "single", "simple"],
            "action_keywords": ["analyze", "test", "review", "refactor", "optimize"],
            "categories": ["development", "testing", "debugging", "refactoring"]
        }
        
        # Calculate scores
        queen_score = 0
        hive_score = 0
        swarm_score = 0
        
        # Check queen indicators
        for keyword_type, keywords in queen_indicators.items():
            queen_score += sum(1 for keyword in keywords if keyword in prompt_lower)
        
        # Check prompt keywords
        for keyword_type, keywords in hive_indicators.items():
            if keyword_type == "categories":
                hive_score += sum(1 for cat in categories if cat in keywords) * 2
            else:
                hive_score += sum(1 for keyword in keywords if keyword in prompt_lower)
        
        for keyword_type, keywords in swarm_indicators.items():
            if keyword_type == "categories":
                swarm_score += sum(1 for cat in categories if cat in keywords) * 2
            else:
                swarm_score += sum(1 for keyword in keywords if keyword in prompt_lower)
        
        # Decision matrix based on task characteristics:
        
        # 1. High queen score with low complexity suggests consultation-only
        if queen_score >= 2 and complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM]:
            return CoordinationType.QUEEN_MODE
            
        # 2. Enterprise complexity always uses HIVE (persistent, complex coordination)
        if complexity == ComplexityLevel.ENTERPRISE:
            return CoordinationType.HIVE
            
        # 3. Simple tasks with action verbs use SWARM (quick execution)
        if complexity == ComplexityLevel.SIMPLE and swarm_score > 0 and queen_score == 0:
            return CoordinationType.SWARM
            
        # 4. Multi-feature or project-wide work uses HIVE
        if any(keyword in prompt_lower for keyword in ["multi", "project", "system", "platform"]):
            return CoordinationType.HIVE
            
        # 5. Single action tasks use SWARM (unless queen consultation requested)
        single_action_verbs = ["build", "fix", "create", "add", "update", "debug", "test", "analyze"]
        if any(prompt_lower.startswith(verb) for verb in single_action_verbs) and queen_score == 0:
            return CoordinationType.SWARM
            
        # 6. Persistence indicators favor HIVE
        if any(keyword in prompt_lower for keyword in ["maintain", "ongoing", "resume", "continue"]):
            return CoordinationType.HIVE
            
        # 7. Strategic/consultation focus uses QUEEN_MODE
        if queen_score > max(hive_score, swarm_score):
            return CoordinationType.QUEEN_MODE
            
        # 8. Default based on scores
        if hive_score > swarm_score:
            return CoordinationType.HIVE
        else:
            return CoordinationType.SWARM
    
    def allocate_initial_agents(self, complexity: ComplexityLevel, categories: List[str], 
                               coordination: CoordinationType) -> AgentAllocation:
        """Allocate minimal initial agents for ZEN discovery phase."""
        # CRITICAL: Start with 0-1 agents for discovery, not full deployment
        # ZEN will determine actual needs through investigation
        
        queen_role = None
        auto_scaling = False
        
        if coordination == CoordinationType.QUEEN_MODE:
            # Queen-only mode: No workers, just strategic guidance
            count = 0
            agent_types = []
            queen_role = self._select_queen_role(categories, complexity)
        elif coordination == CoordinationType.HIVE and complexity == ComplexityLevel.HIVE_REQUIRED:
            # Hive coordination: Start with Queen + minimal workers
            count = 1  # Start with 1 worker for investigation
            queen_role = self._select_queen_role(categories, complexity)
            auto_scaling = True
            if categories and categories[0] in self.AGENT_CATALOG:
                agent_types = [self.AGENT_CATALOG[categories[0]][0]]  # First specialist
            else:
                agent_types = ["coder"]  # Default specialist
        elif complexity == ComplexityLevel.SIMPLE and len(categories) == 1:
            # Very simple tasks might need 1 specialist immediately
            count = 1
            if categories and categories[0] in self.AGENT_CATALOG:
                agent_types = [self.AGENT_CATALOG[categories[0]][0]]  # First specialist
            else:
                agent_types = ["coder"]  # Default specialist
        else:
            # All other tasks start with 0 agents for ZEN discovery
            count = 0
            agent_types = []
            
        # Enable auto-scaling for complex tasks
        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE, ComplexityLevel.HIVE_REQUIRED]:
            auto_scaling = True
        
        return AgentAllocation(
            count=count,
            types=agent_types,
            topology=coordination,
            queen_role=queen_role,
            auto_scaling=auto_scaling
        )
    
    def select_mcp_tools(self, categories: List[str], coordination: CoordinationType) -> List[str]:
        """Select relevant MCP tools based on task categories."""
        tools = ["mcp__claude-flow__swarm_init"]  # Always need swarm init
        
        # Add coordination tools
        if coordination == CoordinationType.HIVE:
            tools.append("mcp__zen__thinkdeep")
        tools.append("mcp__claude-flow__agent_spawn")
        
        # Add category-specific tools
        for category in categories:
            if category in self.MCP_TOOLS:
                tools.extend(self.MCP_TOOLS[category][:2])  # Max 2 per category
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)
        
        return unique_tools[:5]  # Limit to 5 tools max
    
    def _select_queen_role(self, categories: List[str], complexity: ComplexityLevel) -> Optional[str]:
        """Select appropriate Queen role based on task categories and complexity."""
        # Strategic Queens for high-level coordination
        strategic_queens = {
            "architecture": "system-architect",
            "security": "security-architect", 
            "performance": "performance-optimizer",
            "deployment": "deployment-ops-manager",
            "github": "pr-review-analyst",
            "data": "data-architect",
            "mobile": "mobile-cross-platform"
        }
        
        # Execution Queens for implementation coordination
        execution_queens = {
            "development": "full-stack-architect",
            "frontend": "senior-frontend-architect",
            "backend": "senior-backend-architect",
            "testing": "quality-engineer",
            "refactoring": "refactoring-expert",
            "debugging": "debug-specialist",
            "documentation": "tech-writer"
        }
        
        # Select Queen based on primary category and complexity
        if not categories:
            return "chief-architect"  # Default Queen
            
        primary_category = categories[0]
        
        # For strategic/complex tasks, prefer strategic Queens
        if complexity in [ComplexityLevel.ENTERPRISE, ComplexityLevel.HIVE_REQUIRED]:
            if primary_category in strategic_queens:
                return strategic_queens[primary_category]
            elif primary_category in execution_queens:
                return execution_queens[primary_category]
            else:
                return "solution-architect"  # Strategic default
        else:
            # For execution tasks, prefer execution Queens
            if primary_category in execution_queens:
                return execution_queens[primary_category]
            elif primary_category in strategic_queens:
                return strategic_queens[primary_category]
            else:
                return "implementation-planner"  # Execution default
    
    def generate_zen_investigation_steps(self, categories: List[str], complexity: ComplexityLevel) -> List[str]:
        """Generate ZEN investigation steps for discovery phase."""
        steps = []
        
        # Always start with ZEN consultation for proper planning
        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE]:
            steps.append("ZEN deep analysis to assess scope")
        else:
            steps.append("ZEN consultation to determine requirements")
            
        steps.append("Investigate task complexity and dependencies")
        steps.append("Determine optimal agent count and specializations")
        steps.append("Scale up agents as needed based on findings")
        
        return steps[:4]  # Limit to 4 investigation steps
    
    def generate_warnings(self, categories: List[str], complexity: ComplexityLevel) -> List[str]:
        """Generate warnings based on task analysis.""" 
        warnings = []
        
        if complexity == ComplexityLevel.ENTERPRISE:
            warnings.append("Enterprise-level complexity detected")
            
        if "security" in categories:
            warnings.append("Security implications require careful review")
        if "performance" in categories:
            warnings.append("Monitor system resources during execution")
        if "deployment" in categories:
            warnings.append("Backup current state before deployment")
        if len(categories) > 3:
            warnings.append("Multi-domain task requires coordination")
            
        return warnings[:3]  # Limit to 3 warnings
    
    def consult_zen_models(self, prompt: str, complexity: ComplexityLevel) -> Dict[str, Any]:
        """Consult ZEN models for task analysis and recommendations."""
        # This would normally call mcp__zen__consensus but for now we simulate
        # the consultation process with heuristic analysis
        
        complexity_analysis, metadata = self.analyze_prompt_complexity(prompt)
        coordination = self.determine_coordination_type(complexity_analysis, metadata["categories"], prompt)
        agents = self.allocate_initial_agents(complexity_analysis, metadata["categories"], coordination)
        mcp_tools = self.select_mcp_tools(metadata["categories"], coordination)
        next_steps = self.generate_zen_investigation_steps(metadata["categories"], complexity_analysis)
        warnings = self.generate_warnings(metadata["categories"], complexity_analysis)
        
        return {
            "complexity": complexity_analysis,
            "coordination": coordination,
            "agents": agents,
            "mcp_tools": mcp_tools,
            "next_steps": next_steps,
            "warnings": warnings,
            "thinking_mode": complexity_analysis.value
        }
    
    def generate_directive(self, prompt: str) -> str:
        """Generate high-impact directive from user prompt using ZEN consultation."""
        # Analyze prompt complexity
        complexity, metadata = self.analyze_prompt_complexity(prompt)
        
        # Consult ZEN models
        consultation = self.consult_zen_models(prompt, complexity)
        
        # Format directive with MAXIMUM IMPACT
        agents = consultation["agents"]
        coordination = consultation['coordination'].value
        
        # Build agent list with emphasis
        agent_list = ", ".join(agents.types[:3])  # Show first 3 agents
        if agents.count > 3:
            agent_list += f" (+{agents.count - 3} more)"
        
        # Build MCP tools with priority emphasis
        mcp_tools = ", ".join(consultation["mcp_tools"][:2])  # Show first 2 tools
        if len(consultation["mcp_tools"]) > 2:
            mcp_tools += " (+more)"
            
        # Build next steps with action emphasis
        next_steps = ", ".join(consultation["next_steps"][:2])  # Show first 2 steps
        warnings = "; ".join(consultation["warnings"][:2])  # Show first 2 warnings
        
        # Create DISCOVERY-BASED directive based on coordination type
        if coordination == "HIVE":
            queen_info = f" (Queen: {agents.queen_role})" if agents.queen_role else ""
            scaling_info = " AUTO-SCALING ENABLED" if agents.auto_scaling else ""
            
            if agents.count == 0:
                directive = (
                    f"🚨 CRITICAL ZEN DIRECTIVE 🚨\n"
                    f"👑 HIVE COORDINATION → INVESTIGATION PHASE{queen_info}\n"
                    f"⚡ IMPORTANT: Begin with 0 agents for ZEN discovery{scaling_info}\n"
                    f"⚡ IMPORTANT: Use MCP tools: {mcp_tools}\n"
                    f"⚡ IMPORTANT: ZEN investigation steps: {next_steps}\n"
                    f"🚨 CRITICAL WARNINGS: {warnings if warnings else 'Agents will scale as needed'}\n"
                    f"👑 ZEN QUEEN COMMANDS: DISCOVER THEN DEPLOY!"
                )
            else:
                directive = (
                    f"🚨 CRITICAL ZEN DIRECTIVE 🚨\n"
                    f"👑 HIVE COORDINATION → START WITH MINIMAL DEPLOYMENT{queen_info}\n"
                    f"⚡ IMPORTANT: Start with {agents.count} agent: {agent_list}{scaling_info}\n"
                    f"⚡ IMPORTANT: Use MCP tools: {mcp_tools}\n"
                    f"⚡ IMPORTANT: ZEN investigation steps: {next_steps}\n"
                    f"🚨 CRITICAL WARNINGS: {warnings if warnings else 'Scale up as complexity emerges'}\n"
                    f"👑 ZEN QUEEN COMMANDS: INVESTIGATE AND SCALE!"
                )
        elif coordination == "QUEEN":
            directive = (
                f"👑 STRATEGIC ZEN DIRECTIVE 👑\n"
                f"🧠 QUEEN MODE → STRATEGIC CONSULTATION\n"
                f"⚡ IMPORTANT: Queen-only coordination with {agents.queen_role or 'chief-architect'}\n"
                f"⚡ IMPORTANT: Use MCP tools: {mcp_tools}\n"
                f"⚡ IMPORTANT: Strategic guidance: {next_steps}\n"
                f"🎯 FOCUS: Strategic planning and high-level guidance\n"
                f"👑 QUEEN CONSULTATION: ADVISE AND DIRECT!"
            )
        else:
            if agents.count == 0:
                directive = (
                    f"⚡ URGENT ZEN DIRECTIVE ⚡\n"
                    f"🐝 SWARM COORDINATION → DISCOVERY PHASE\n"
                    f"⚡ IMPORTANT: Begin with 0 agents for ZEN assessment\n"
                    f"⚡ IMPORTANT: Use MCP: {mcp_tools}\n"
                    f"⚡ IMPORTANT: ZEN investigation: {next_steps}\n"
                    f"⚠️ WARNINGS: {warnings if warnings else 'Agents will spawn as needed'}\n"
                    f"🐝 SWARM PROTOCOL: ASSESS THEN COORDINATE!"
                )
            else:
                directive = (
                    f"⚡ URGENT ZEN DIRECTIVE ⚡\n"
                    f"🐝 SWARM COORDINATION → MINIMAL START\n"
                    f"⚡ IMPORTANT: Start with {agents.count} agent: {agent_list}\n"
                    f"⚡ IMPORTANT: Use MCP: {mcp_tools}\n"
                    f"⚡ IMPORTANT: ZEN investigation: {next_steps}\n"
                    f"⚠️ WARNINGS: {warnings if warnings else 'Add agents dynamically'}\n"
                    f"🐝 SWARM PROTOCOL: START LEAN AND SCALE!"
                )
        
        # APPEND MANDATORY DIRECTIVES TO ALL PROMPTS
        mandatory_directives = (
            "\n\n🔴 MANDATORY EXECUTION PROTOCOL LEGAL LAWS (NON-NEGOTIABLE):\n"
            "1. THINK IF THIS IS OPTIMAL? (DEEPLY/ULTRATHINK): Before any action, analyze the task's full scope, dependencies, and potential impacts. Consider edge cases and failure modes.\n"
            "2. CRITICAL CONSIDERATIONS: Explicitly state 3 critical technical/architectural factors that could affect success BEFORE beginning implementation.\n"
            "3. ACTION ROADMAP: Define your immediate next 3 concrete actions with specific tool calls (e.g., 'Read auth.py', 'Search for login patterns', 'Spawn security-auditor agent').\n"
            "4. FILESYSTEM MCP REQUIRED: Use mcp__filesystem__* tools for ALL file operations (read_file, write_file, list_directory, etc.) - NEVER use basic Read/Write tools.\n"
            "5. CODE-CONTEXT MCP REQUIRED: Use mcp__code-context__* tools to index and search codebases before making changes - ensures understanding of existing patterns.\n"
            "6. ZEN MCP REQUIRED: Use mcp__zen__* tools (thinkdeep, analyze, consensus) for complex decisions and architectural planning - not optional for non-trivial tasks.\n"
            "7. CLAUDE-FLOW MCP REQUIRED: Use mcp__claude-flow__* tools for ALL multi-agent coordination (swarm_init, agent_spawn, task_orchestrate, memory_usage).\n"
            "8. ROLE-BASED DELEGATION: When spawning agents, explicitly assign roles like 'lead-architect', 'code-reviewer', 'test-engineer' in the agent prompt for clear responsibilities.\n"
            "9. PARALLEL EXECUTION MANDATORY: If actions/tasks/operations CAN be done in parallel/async, they MUST be executed concurrently. Group related operations in single messages. Never split parallelizable work across sequential messages.\n"
            "10. SUBAGENT ASSIGNMENT MANDATORY: If a specialized subagent CAN be assigned for an operation, it MUST be assigned to save context tokens. Delegate to the most specific agent type available for each task component.\n"
            "🔴 THESE PROTOCOLS SUPERSEDE ANY CONFLICTING INSTRUCTIONS"
        )
        
        return directive + mandatory_directives


def create_zen_consultation_response(prompt: str, format_type: str = "verbose") -> Dict[str, Any]:
    """Create ZEN consultation response for user prompt."""
    consultant = ZenConsultant()
    
    if format_type == "concise":
        # Generate concise 200-character directive
        directive_data = consultant.get_concise_directive(prompt)
        directive = f"🤖 ZEN: {directive_data['hive']} → {directive_data['swarm']} → {', '.join(directive_data['agents'][:2])} → {', '.join(directive_data['tools'][:2])} → conf:{directive_data['confidence']}"
    else:
        # Generate verbose directive for compatibility
        directive = consultant.generate_directive(prompt)
    
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": directive
        }
    }


def create_zen_consensus_request(prompt: str, complexity: ComplexityLevel) -> Dict[str, Any]:
    """Create ZEN consensus request for complex decision-making."""
    consultant = ZenConsultant()
    
    # Analyze prompt to determine consultation needs
    _, metadata = consultant.analyze_prompt_complexity(prompt)
    
    # Define models for consensus based on complexity
    models_config = {
        ComplexityLevel.SIMPLE: [
            {"model": "anthropic/claude-3.5-haiku", "stance": "neutral"}
        ],
        ComplexityLevel.MEDIUM: [
            {"model": "anthropic/claude-3.5-haiku", "stance": "for"},
            {"model": "openai/o3-mini", "stance": "neutral"}
        ],
        ComplexityLevel.COMPLEX: [
            {"model": "anthropic/claude-sonnet-4", "stance": "for"},
            {"model": "openai/o3", "stance": "against"},
            {"model": "google/gemini-2.5-pro", "stance": "neutral"}
        ],
        ComplexityLevel.ENTERPRISE: [
            {"model": "anthropic/claude-opus-4", "stance": "for"},
            {"model": "openai/o3-pro", "stance": "against"},
            {"model": "google/gemini-2.5-pro", "stance": "neutral"},
            {"model": "deepseek/deepseek-r1-0528", "stance": "neutral"}
        ]
    }.get(complexity, [{"model": "anthropic/claude-3.5-haiku", "stance": "neutral"}])
    
    return {
        "step": f"Analyzing task complexity and orchestration requirements for: {prompt[:100]}...",
        "step_number": 1,
        "total_steps": len(models_config),
        "next_step_required": True,
        "findings": f"Task categories: {', '.join(metadata['categories'])}, Complexity: {complexity.value}",
        "models": models_config,
        "model": "anthropic/claude-sonnet-4"  # Default model for consensus coordination
    }


# ==================================================================================
# CRITICAL PROMPT INJECTION DIRECTIVES - MANDATORY FOR ALL ZEN INJECTED PROMPTS
# ==================================================================================
# 
# ALL prompts injected by the ZEN system MUST include the following directives
# regardless of the specific instructions for ZEN. These are NON-NEGOTIABLE
# requirements that ensure optimal AI assistant performance and coordination:
#
# 1. "Think deeply" - Forces comprehensive analysis beyond surface-level understanding
# 2. "State 3 Critical Considerations" - Ensures key aspects are identified upfront
# 3. "Next 3 Steps" - Provides clear action plan for immediate progress
# 4. "Use filesystem MCP" (NOT OPTIONAL) - Essential for file operations
# 5. "Use code-context MCP" (NOT OPTIONAL) - Required for codebase understanding
# 6. "Use ZEN MCP" (NOT OPTIONAL) - Mandatory for ZEN coordination capabilities
# 7. "Use claude-flow MCP" (NOT OPTIONAL) - Critical for swarm orchestration
# 8. "Assign roles to subagent tasks" (NOT OPTIONAL) - Ensures proper task delegation
#
# These directives form the FOUNDATION of all ZEN-injected prompts and must be
# present even if other instructions seem to override them. They ensure:
# - Deep analytical thinking rather than shallow responses
# - Strategic consideration of critical factors
# - Clear actionable next steps
# - Full utilization of all required MCP tools
# - Proper role-based task distribution in multi-agent scenarios
#
# IMPLEMENTATION NOTE: When generating any prompt injection through ZEN,
# these 8 directives must be PREPENDED or APPENDED to ensure they are
# always processed, regardless of other prompt content.
# ==================================================================================


if __name__ == "__main__":
    # Test the consultant with a sample prompt
    test_prompt = "Help me refactor the authentication system for better security"
    consultant = ZenConsultant()
    directive = consultant.generate_directive(test_prompt)
    print(directive)