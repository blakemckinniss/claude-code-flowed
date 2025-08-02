#!/usr/bin/env python3
"""Progressive Verbosity Adapter - Adapts feedback verbosity based on user expertise and context.

This module implements adaptive feedback that evolves with user expertise, providing:
- Beginner-friendly guidance with explanations
- Intermediate guidance with best practices  
- Advanced optimization suggestions
- Expert-level technical insights
"""

import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .tool_pattern_analyzer import UserExpertiseLevel, ToolCategory


class VerbosityLevel(Enum):
    """Verbosity levels for feedback adaptation."""
    SILENT = "silent"          # No feedback (expert preference)
    MINIMAL = "minimal"        # Only critical issues
    CONCISE = "concise"        # Brief, actionable feedback
    DETAILED = "detailed"      # Comprehensive explanations
    EDUCATIONAL = "educational"  # Learning-focused with examples


class ContextType(Enum):
    """Context types for feedback adaptation."""
    ONBOARDING = "onboarding"      # First-time user experience
    ROUTINE = "routine"            # Regular development work
    EXPLORATION = "exploration"    # Trying new tools/patterns
    OPTIMIZATION = "optimization"  # Performance/efficiency focus
    DEBUGGING = "debugging"        # Error resolution
    LEARNING = "learning"          # Educational context


@dataclass
class AdaptationMetrics:
    """Metrics for tracking user adaptation."""
    total_interactions: int
    successful_patterns: int
    mcp_tool_adoption: float
    error_recovery_rate: float
    advanced_feature_usage: float
    feedback_effectiveness: float
    learning_velocity: float
    expertise_progression: float


@dataclass
class AdaptedFeedback:
    """Feedback adapted to user expertise and context."""
    primary_message: str
    explanation: Optional[str]
    examples: List[str]
    technical_notes: Optional[str]
    learning_resources: List[str]
    complexity_level: str
    estimated_reading_time: int  # seconds


class ProgressiveVerbosityAdapter:
    """Adapts feedback verbosity and complexity based on user progression."""
    
    # Verbosity templates for different expertise levels
    VERBOSITY_TEMPLATES = {
        UserExpertiseLevel.BEGINNER: {
            "greeting": "🎓 Learning Mode Active",
            "explanation_prefix": "💡 What this means:",
            "example_prefix": "📝 Example:",
            "next_steps_prefix": "🚀 Try this next:",
            "technical_prefix": "🔍 Technical detail:",
            "encouragement": "You're building great development habits!"
        },
        UserExpertiseLevel.INTERMEDIATE: {
            "greeting": "⚡ Development Optimization",
            "explanation_prefix": "💡 Optimization opportunity:",
            "example_prefix": "🛠️ Quick example:",
            "next_steps_prefix": "📋 Recommended actions:",
            "technical_prefix": "⚙️ Technical insight:",
            "encouragement": "You're mastering Claude Code workflows!"
        },
        UserExpertiseLevel.ADVANCED: {
            "greeting": "🔬 Advanced Analysis",
            "explanation_prefix": "⚡ Performance insight:",
            "example_prefix": "🎯 Implementation:",
            "next_steps_prefix": "🚀 Optimization path:",
            "technical_prefix": "🧠 Technical analysis:",
            "encouragement": "Excellent workflow optimization!"
        },
        UserExpertiseLevel.EXPERT: {
            "greeting": "🧬 Expert Intelligence",
            "explanation_prefix": "⚡ System analysis:",
            "example_prefix": "🔧 Implementation pattern:",
            "next_steps_prefix": "🎯 Strategic actions:",
            "technical_prefix": "📊 Performance metrics:",
            "encouragement": "Peak efficiency achieved!"
        }
    }
    
    # Context-specific verbosity adjustments
    CONTEXT_ADJUSTMENTS = {
        ContextType.ONBOARDING: {"verbosity_boost": 2, "include_examples": True, "include_explanations": True},
        ContextType.ROUTINE: {"verbosity_boost": 0, "include_examples": False, "include_explanations": False},
        ContextType.EXPLORATION: {"verbosity_boost": 1, "include_examples": True, "include_explanations": True},
        ContextType.OPTIMIZATION: {"verbosity_boost": 0, "include_examples": False, "include_explanations": False},
        ContextType.DEBUGGING: {"verbosity_boost": 1, "include_examples": True, "include_explanations": True},
        ContextType.LEARNING: {"verbosity_boost": 2, "include_examples": True, "include_explanations": True}
    }
    
    def __init__(self):
        """Initialize progressive verbosity adapter."""
        self.adaptation_history: List[AdaptationMetrics] = []
        self.user_preferences = self._load_user_preferences()
        self.session_context = self._load_session_context()
        self.feedback_effectiveness_tracker = {}
        
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences for verbosity adaptation."""
        try:
            pref_file = "/home/devcontainers/flowed/.claude/hooks/.session/verbosity_preferences.json"
            if os.path.exists(pref_file):
                with open(pref_file) as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            "preferred_verbosity": VerbosityLevel.DETAILED.value,
            "auto_adapt": True,
            "include_examples": True,
            "include_technical_details": True,
            "learning_mode": False,
            "context_awareness": True,
            "feedback_frequency": "standard"  # minimal, standard, verbose
        }
    
    def _load_session_context(self) -> Dict[str, Any]:
        """Load session context for adaptation."""
        try:
            context_file = "/home/devcontainers/flowed/.claude/hooks/.session/adaptation_context.json"
            if os.path.exists(context_file):
                with open(context_file) as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            "session_start_time": time.time(),
            "interaction_count": 0,
            "context_type": ContextType.ROUTINE.value,
            "recent_tool_categories": [],
            "adaptation_triggers": [],
            "learning_objectives": []
        }
    
    def detect_context_type(self, tool_name: str, tool_input: Dict[str, Any], 
                          recent_tools: List[str], user_expertise: UserExpertiseLevel) -> ContextType:
        """Detect the current context type for appropriate verbosity adaptation."""
        
        # Check for onboarding patterns
        if (self.session_context.get("interaction_count", 0) < 10 and 
            user_expertise == UserExpertiseLevel.BEGINNER):
            return ContextType.ONBOARDING
        
        # Check for exploration patterns
        mcp_tools = [tool for tool in recent_tools if tool.startswith("mcp__")]
        if len(set(mcp_tools)) > 2 and len(mcp_tools) < 5:  # Trying different MCP tools
            return ContextType.EXPLORATION
        
        # Check for debugging patterns
        error_indicators = ["error", "fix", "debug", "issue", "problem"]
        if any(indicator in str(tool_input).lower() for indicator in error_indicators):
            return ContextType.DEBUGGING
        
        # Check for optimization patterns
        if "optimize" in str(tool_input).lower() or tool_name.startswith("mcp__zen__"):
            return ContextType.OPTIMIZATION
        
        # Check for learning patterns
        learning_indicators = ["learn", "understand", "explain", "how", "why", "example"]
        if any(indicator in str(tool_input).lower() for indicator in learning_indicators):
            return ContextType.LEARNING
        
        # Default to routine
        return ContextType.ROUTINE
    
    def adapt_feedback_verbosity(self, base_message: str, tool_name: str, tool_input: Dict[str, Any],
                                user_expertise: UserExpertiseLevel, recent_tools: List[str],
                                feedback_priority: float) -> AdaptedFeedback:
        """Adapt feedback verbosity based on user expertise and context."""
        
        # Detect current context
        context_type = self.detect_context_type(tool_name, tool_input, recent_tools, user_expertise)
        
        # Get verbosity level
        verbosity_level = self._determine_verbosity_level(user_expertise, context_type, feedback_priority)
        
        # Get templates for user expertise level
        templates = self.VERBOSITY_TEMPLATES[user_expertise]
        
        # Adapt the message
        adapted_feedback = self._create_adapted_feedback(
            base_message, verbosity_level, templates, context_type, tool_name, user_expertise
        )
        
        # Update session context
        self._update_session_context(context_type, tool_name)
        
        return adapted_feedback
    
    def _determine_verbosity_level(self, user_expertise: UserExpertiseLevel, 
                                 context_type: ContextType, feedback_priority: float) -> VerbosityLevel:
        """Determine appropriate verbosity level."""
        
        # Base verbosity from user preferences
        base_verbosity = self.user_preferences.get("preferred_verbosity", VerbosityLevel.DETAILED.value)
        
        # Auto-adaptation based on expertise
        if self.user_preferences.get("auto_adapt", True):
            expertise_mapping = {
                UserExpertiseLevel.BEGINNER: VerbosityLevel.EDUCATIONAL,
                UserExpertiseLevel.INTERMEDIATE: VerbosityLevel.DETAILED,
                UserExpertiseLevel.ADVANCED: VerbosityLevel.CONCISE,
                UserExpertiseLevel.EXPERT: VerbosityLevel.MINIMAL
            }
            base_verbosity = expertise_mapping[user_expertise].value
        
        # Context adjustments
        context_adj = self.CONTEXT_ADJUSTMENTS.get(context_type, {"verbosity_boost": 0})
        verbosity_boost = context_adj["verbosity_boost"]
        
        # Priority adjustments
        if feedback_priority >= 0.8:  # High priority - increase verbosity
            verbosity_boost += 1
        elif feedback_priority <= 0.3:  # Low priority - decrease verbosity
            verbosity_boost -= 1
        
        # Apply verbosity mapping with boost
        verbosity_levels = list(VerbosityLevel)
        try:
            current_index = verbosity_levels.index(VerbosityLevel(base_verbosity))
            adjusted_index = max(0, min(len(verbosity_levels) - 1, current_index + verbosity_boost))
            return verbosity_levels[adjusted_index]
        except ValueError:
            return VerbosityLevel.DETAILED
    
    def _create_adapted_feedback(self, base_message: str, verbosity_level: VerbosityLevel,
                               templates: Dict[str, str], context_type: ContextType, 
                               tool_name: str, user_expertise: UserExpertiseLevel) -> AdaptedFeedback:
        """Create adapted feedback based on verbosity level and context."""
        
        # Start with base message
        primary_message = f"{templates['greeting']}: {base_message}"
        
        explanation = None
        examples = []
        technical_notes = None
        learning_resources = []
        
        # Add content based on verbosity level
        if verbosity_level in [VerbosityLevel.DETAILED, VerbosityLevel.EDUCATIONAL]:
            explanation = self._generate_explanation(tool_name, context_type, templates)
            
        if verbosity_level == VerbosityLevel.EDUCATIONAL:
            examples = self._generate_examples(tool_name, context_type, user_expertise)
            learning_resources = self._generate_learning_resources(tool_name, context_type)
            
        if (verbosity_level in [VerbosityLevel.DETAILED, VerbosityLevel.EDUCATIONAL] and 
            self.user_preferences.get("include_technical_details", True)):
            technical_notes = self._generate_technical_notes(tool_name, user_expertise)
        
        # Calculate estimated reading time
        reading_time = self._estimate_reading_time(primary_message, explanation, examples, technical_notes)
        
        return AdaptedFeedback(
            primary_message=primary_message,
            explanation=explanation,
            examples=examples,
            technical_notes=technical_notes,
            learning_resources=learning_resources,
            complexity_level=user_expertise.value,
            estimated_reading_time=reading_time
        )
    
    def _generate_explanation(self, tool_name: str, context_type: ContextType, 
                            templates: Dict[str, str]) -> str:
        """Generate contextual explanation."""
        explanations = {
            "mcp__zen__": "ZEN MCP tools provide intelligent coordination and analysis capabilities that enhance your development workflow with AI-powered insights.",
            "mcp__claude-flow__": "Claude Flow tools enable multi-agent coordination and swarm intelligence for complex development tasks.",
            "mcp__filesystem__": "Filesystem MCP tools provide enhanced file operations with better error handling and batch processing capabilities.",
            "Write": "File writing operations can be optimized through batch processing and intelligent coordination.",
            "Read": "File reading can be enhanced with MCP tools for better performance and context awareness.",
            "Edit": "File editing benefits from pattern analysis and intelligent workflow coordination.",
            "Bash": "System commands can be coordinated through ZEN intelligence for better error handling and workflow optimization."
        }
        
        for prefix, explanation in explanations.items():
            if tool_name.startswith(prefix) or tool_name == prefix:
                return f"{templates['explanation_prefix']} {explanation}"
        
        return f"{templates['explanation_prefix']} This tool can be enhanced through intelligent coordination patterns."
    
    def _generate_examples(self, tool_name: str, context_type: ContextType, 
                         user_expertise: UserExpertiseLevel) -> List[str]:
        """Generate contextual examples."""
        examples = []
        
        if tool_name.startswith("mcp__zen__"):
            examples.append("Try: mcp__zen__analyze for intelligent code analysis")
            if user_expertise != UserExpertiseLevel.BEGINNER:
                examples.append("Combine: mcp__zen__thinkdeep → mcp__claude-flow__swarm_init")
        
        elif tool_name.startswith("mcp__claude-flow__"):
            examples.append("Initialize: mcp__claude-flow__swarm_init with mesh topology")
            examples.append("Spawn agents: mcp__claude-flow__agent_spawn for specialized tasks")
        
        elif tool_name in ["Write", "Edit", "Read"]:
            examples.append("Batch operations: Use mcp__filesystem__read_multiple_files")
            examples.append("Enhanced workflow: mcp__zen__planner → file operations")
        
        elif tool_name == "Bash":
            examples.append("Coordinate commands: mcp__zen__planner → Bash sequence")
            examples.append("Error handling: mcp__zen__debug for command troubleshooting")
        
        return examples[:2] if examples else []  # Limit to 2 examples
    
    def _generate_technical_notes(self, tool_name: str, user_expertise: UserExpertiseLevel) -> Optional[str]:
        """Generate technical notes for advanced users."""
        if user_expertise in [UserExpertiseLevel.BEGINNER, UserExpertiseLevel.INTERMEDIATE]:
            return None
        
        technical_notes = {
            "mcp__zen__": "ZEN tools use neural pattern matching and context intelligence for optimization recommendations.",
            "mcp__claude-flow__": "Claude Flow implements swarm intelligence algorithms with memory persistence and agent coordination.",
            "mcp__filesystem__": "Filesystem MCP provides async I/O with batching, error recovery, and context preservation.",
            "Write": "File operations can be optimized through write batching and async execution patterns.",
            "Read": "Reading performance improves with predictive caching and parallel execution strategies."
        }
        
        for prefix, note in technical_notes.items():
            if tool_name.startswith(prefix) or tool_name == prefix:
                return note
        
        return None
    
    def _generate_learning_resources(self, tool_name: str, context_type: ContextType) -> List[str]:
        """Generate learning resources for educational contexts."""
        resources = []
        
        if context_type in [ContextType.ONBOARDING, ContextType.LEARNING]:
            resources.append("📚 See: CLAUDE.md for comprehensive MCP tool guide")
            
            if tool_name.startswith("mcp__zen__"):
                resources.append("🎯 Learn: ZEN orchestration patterns in the documentation")
            elif tool_name.startswith("mcp__claude-flow__"):
                resources.append("🚀 Explore: Multi-agent coordination examples")
        
        return resources
    
    def _estimate_reading_time(self, primary_message: str, explanation: Optional[str],
                             examples: List[str], technical_notes: Optional[str]) -> int:
        """Estimate reading time in seconds (average 200 words per minute)."""
        total_text = primary_message
        
        if explanation:
            total_text += " " + explanation
        if examples:
            total_text += " " + " ".join(examples)
        if technical_notes:
            total_text += " " + technical_notes
        
        word_count = len(total_text.split())
        reading_time = (word_count / 200) * 60  # Convert to seconds
        
        return max(5, int(reading_time))  # Minimum 5 seconds
    
    def _update_session_context(self, context_type: ContextType, tool_name: str):
        """Update session context for learning adaptation."""
        self.session_context["interaction_count"] += 1
        self.session_context["context_type"] = context_type.value
        
        # Track recent tool categories
        recent_categories = self.session_context.get("recent_tool_categories", [])
        if tool_name.startswith("mcp__"):
            category = "mcp"
        elif tool_name in ["Write", "Read", "Edit"]:
            category = "file_ops"
        else:
            category = "system"
        
        recent_categories.append(category)
        self.session_context["recent_tool_categories"] = recent_categories[-10:]  # Keep last 10
        
        # Save context
        self._save_session_context()
    
    def format_adapted_feedback(self, adapted_feedback: AdaptedFeedback) -> str:
        """Format adapted feedback for stderr output."""
        lines = []
        
        # Header with reading time estimate
        if adapted_feedback.estimated_reading_time > 15:
            lines.append(f"\n⏱️ Estimated reading time: {adapted_feedback.estimated_reading_time}s")
        
        # Primary message
        lines.append(f"\n{adapted_feedback.primary_message}")
        
        # Explanation
        if adapted_feedback.explanation:
            lines.append(f"\n{adapted_feedback.explanation}")
        
        # Examples
        if adapted_feedback.examples:
            lines.append("\n📝 Examples:")
            for example in adapted_feedback.examples:
                lines.append(f"  • {example}")
        
        # Technical notes
        if adapted_feedback.technical_notes:
            lines.append(f"\n🔍 Technical: {adapted_feedback.technical_notes}")
        
        # Learning resources
        if adapted_feedback.learning_resources:
            lines.append("\n📚 Learn More:")
            for resource in adapted_feedback.learning_resources:
                lines.append(f"  • {resource}")
        
        return "\n".join(lines) + "\n"
    
    def _save_session_context(self):
        """Save session context for future adaptation."""
        try:
            context_dir = "/home/devcontainers/flowed/.claude/hooks/.session"
            os.makedirs(context_dir, exist_ok=True)
            
            context_file = os.path.join(context_dir, "adaptation_context.json")
            with open(context_file, 'w') as f:
                json.dump(self.session_context, f, indent=2)
        except Exception:
            pass  # Silent failure for context saving
    
    def get_adaptation_metrics(self) -> AdaptationMetrics:
        """Get current adaptation metrics for monitoring."""
        interaction_count = self.session_context.get("interaction_count", 0)
        recent_categories = self.session_context.get("recent_tool_categories", [])
        
        mcp_usage = recent_categories.count("mcp") / max(1, len(recent_categories))
        
        return AdaptationMetrics(
            total_interactions=interaction_count,
            successful_patterns=len([cat for cat in recent_categories if cat == "mcp"]),
            mcp_tool_adoption=mcp_usage,
            error_recovery_rate=0.8,  # Would be calculated from actual data
            advanced_feature_usage=mcp_usage,
            feedback_effectiveness=0.7,  # Would be calculated from user responses
            learning_velocity=mcp_usage * interaction_count / 10,  # Simple metric
            expertise_progression=min(1.0, interaction_count / 50)  # Progress toward expertise
        )