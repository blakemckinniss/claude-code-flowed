#!/usr/bin/env python3
"""Intelligent Feedback Generator - Connects tool pattern analysis with ZEN intelligence.

This module serves as the bridge between tool usage pattern analysis and the ZEN consultant system,
generating contextually intelligent feedback that enhances Claude Code development experience.
"""

import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .tool_pattern_analyzer import ToolPatternAnalyzer, FeedbackMessage, FeedbackType, UserExpertiseLevel
from ..core.zen_consultant import ZenConsultant, ComplexityLevel, CoordinationType


class FeedbackIntensity(Enum):
    """Feedback intensity levels based on context."""
    MINIMAL = "minimal"      # Only critical feedback
    STANDARD = "standard"    # Normal feedback level
    VERBOSE = "verbose"      # Detailed feedback with examples
    EXPERT = "expert"        # Technical details and advanced suggestions


@dataclass
class ContextualFeedback:
    """Contextual feedback with ZEN intelligence integration."""
    primary_message: str
    technical_details: str
    zen_guidance: str
    actionable_recommendations: List[str]
    tool_suggestions: List[str]
    priority_score: float
    feedback_intensity: FeedbackIntensity


class IntelligentFeedbackGenerator:
    """Generates intelligent feedback by combining pattern analysis with ZEN consultation."""
    
    def __init__(self):
        """Initialize intelligent feedback generator."""
        self.pattern_analyzer = ToolPatternAnalyzer()
        self.zen_consultant = ZenConsultant()
        self.feedback_history: List[ContextualFeedback] = []
        self.user_preferences = self._load_user_preferences()
        
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences for feedback customization."""
        try:
            pref_file = "/home/devcontainers/flowed/.claude/hooks/.session/feedback_preferences.json"
            if os.path.exists(pref_file):
                with open(pref_file) as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            "feedback_intensity": FeedbackIntensity.STANDARD.value,
            "show_technical_details": True,
            "preferred_feedback_types": ["optimization", "guidance"],
            "suppress_repetitive": True,
            "max_messages_per_session": 3
        }
    
    def generate_contextual_feedback(self, tool_name: str, tool_input: Dict[str, Any], 
                                   tool_response: Dict[str, Any], execution_time: float = 0.0) -> Optional[str]:
        """Generate contextual feedback by combining pattern analysis with ZEN intelligence."""
        
        # Record usage and get pattern-based feedback
        pattern_feedback = self.pattern_analyzer.generate_intelligent_feedback(
            tool_name, tool_input, tool_response
        )
        
        if not pattern_feedback or self._should_suppress_feedback():
            return None
        
        # Get ZEN consultation for the current context
        zen_context = self._create_zen_context(tool_name, tool_input, pattern_feedback)
        zen_guidance = self._get_zen_guidance(zen_context)
        
        # Combine pattern analysis with ZEN intelligence
        contextual_feedback = self._synthesize_feedback(pattern_feedback, zen_guidance, tool_name)
        
        if contextual_feedback:
            # Format for stderr output
            formatted_feedback = self._format_contextual_feedback(contextual_feedback)
            
            # Store feedback for learning
            self.feedback_history.append(contextual_feedback)
            self._update_user_preferences_from_usage()
            
            return formatted_feedback
        
        return None
    
    def _should_suppress_feedback(self) -> bool:
        """Determine if feedback should be suppressed based on user preferences and context."""
        if not self.user_preferences.get("suppress_repetitive", True):
            return False
        
        # Check session limits
        max_messages = self.user_preferences.get("max_messages_per_session", 3)
        if len(self.feedback_history) >= max_messages:
            return True
        
        # Check for repetitive patterns in recent feedback
        if len(self.feedback_history) >= 2:
            recent_messages = [fb.primary_message for fb in self.feedback_history[-2:]]
            if len(set(recent_messages)) == 1:  # Same message repeated
                return True
        
        return False
    
    def _create_zen_context(self, tool_name: str, tool_input: Dict[str, Any], 
                          pattern_feedback: List[FeedbackMessage]) -> str:
        """Create ZEN consultation context from current tool usage and patterns."""
        patterns_summary = []
        for feedback in pattern_feedback:
            if feedback.feedback_type in [FeedbackType.OPTIMIZATION, FeedbackType.WARNING]:
                patterns_summary.append(f"{feedback.feedback_type.value}: {feedback.title}")
        
        zen_context = f"""
        Current Tool: {tool_name}
        Operation Type: {self.pattern_analyzer._classify_operation(tool_name, tool_input)}
        Detected Patterns: {'; '.join(patterns_summary) if patterns_summary else 'None'}
        User Expertise: {self.pattern_analyzer.user_expertise.value}
        Recent Tool Sequence: {' → '.join([usage.tool_name for usage in self.pattern_analyzer.usage_history[-5:]])}
        """
        
        return zen_context.strip()
    
    def _get_zen_guidance(self, context: str) -> Dict[str, Any]:
        """Get ZEN consultant guidance for the current context."""
        try:
            # Analyze context complexity
            complexity, metadata = self.zen_consultant.analyze_prompt_complexity(context)
            
            # Get coordination recommendations
            coordination = self.zen_consultant.determine_coordination_type(
                complexity, metadata.get("categories", []), context
            )
            
            # Get MCP tool recommendations
            mcp_tools = self.zen_consultant.select_mcp_tools(
                metadata.get("categories", []), coordination
            )
            
            return {
                "complexity": complexity,
                "coordination": coordination,
                "recommended_tools": mcp_tools,
                "guidance_level": self._determine_guidance_level(complexity),
                "categories": metadata.get("categories", [])
            }
        except Exception:
            # Fallback guidance
            return {
                "complexity": ComplexityLevel.MEDIUM,
                "coordination": CoordinationType.SWARM,
                "recommended_tools": ["mcp__zen__analyze"],
                "guidance_level": "standard",
                "categories": ["development"]
            }
    
    def _determine_guidance_level(self, complexity: ComplexityLevel) -> str:
        """Determine appropriate guidance level based on complexity and user expertise."""
        user_level = self.pattern_analyzer.user_expertise
        
        if complexity == ComplexityLevel.ENTERPRISE:
            return "expert" if user_level == UserExpertiseLevel.EXPERT else "verbose"
        elif complexity == ComplexityLevel.COMPLEX:
            return "verbose" if user_level in [UserExpertiseLevel.ADVANCED, UserExpertiseLevel.EXPERT] else "standard"
        elif complexity == ComplexityLevel.SIMPLE:
            return "minimal" if user_level == UserExpertiseLevel.EXPERT else "standard"
        else:
            return "standard"
    
    def _synthesize_feedback(self, pattern_feedback: List[FeedbackMessage], 
                           zen_guidance: Dict[str, Any], tool_name: str) -> Optional[ContextualFeedback]:
        """Synthesize pattern analysis with ZEN guidance into contextual feedback."""
        if not pattern_feedback:
            return None
        
        # Select the highest priority feedback message
        primary_feedback = max(pattern_feedback, key=lambda x: x.priority)
        
        # Create ZEN-enhanced guidance
        coordination_type = zen_guidance["coordination"].value if hasattr(zen_guidance["coordination"], 'value') else str(zen_guidance["coordination"])
        
        zen_enhanced_guidance = self._create_zen_enhanced_message(
            primary_feedback, zen_guidance, coordination_type
        )
        
        # Determine feedback intensity
        intensity = self._determine_feedback_intensity(
            primary_feedback, zen_guidance["guidance_level"], zen_guidance["complexity"]
        )
        
        # Create actionable recommendations
        recommendations = self._create_actionable_recommendations(
            primary_feedback, zen_guidance, tool_name
        )
        
        return ContextualFeedback(
            primary_message=primary_feedback.message,
            technical_details=self._create_technical_details(primary_feedback, zen_guidance),
            zen_guidance=zen_enhanced_guidance,
            actionable_recommendations=recommendations,
            tool_suggestions=zen_guidance["recommended_tools"][:3],  # Top 3 tools
            priority_score=primary_feedback.priority / 10.0,
            feedback_intensity=intensity
        )
    
    def _create_zen_enhanced_message(self, feedback: FeedbackMessage, zen_guidance: Dict[str, Any], 
                                   coordination_type: str) -> str:
        """Create ZEN-enhanced guidance message."""
        base_templates = {
            FeedbackType.OPTIMIZATION: [
                "🧠 ZEN Analysis: Your workflow shows optimization potential through {coordination} coordination.",
                "⚡ ZEN Insight: {coordination} topology could improve this pattern by ~{improvement}%.",
                "🔬 ZEN Assessment: Detected efficiency opportunity - {coordination} approach recommended."
            ],
            FeedbackType.GUIDANCE: [
                "🎯 ZEN Guidance: Consider {coordination} coordination for enhanced workflow intelligence.",
                "💡 ZEN Suggestion: {coordination} pattern aligns with your current development approach.",
                "🌟 ZEN Recommendation: Scale to {coordination} coordination for better outcomes."
            ],
            FeedbackType.WARNING: [
                "⚠️ ZEN Alert: Performance pattern detected - {coordination} coordination can help.",
                "🚨 ZEN Notice: Current pattern needs {coordination} approach for stability.",
                "⚡ ZEN Warning: Consider {coordination} coordination to prevent workflow drift."
            ]
        }
        
        templates = base_templates.get(feedback.feedback_type, base_templates[FeedbackType.GUIDANCE])
        template = templates[0]  # Use first template for consistency
        
        # Calculate estimated improvement based on pattern type
        improvement = 300 if coordination_type == "HIVE" else 200
        
        return template.format(
            coordination=coordination_type,
            improvement=improvement
        )
    
    def _determine_feedback_intensity(self, feedback: FeedbackMessage, guidance_level: str, 
                                    complexity: ComplexityLevel) -> FeedbackIntensity:
        """Determine appropriate feedback intensity."""
        user_pref = self.user_preferences.get("feedback_intensity", "standard")
        
        # Override based on feedback priority
        if feedback.priority >= 9:
            return FeedbackIntensity.VERBOSE
        elif feedback.priority <= 3:
            return FeedbackIntensity.MINIMAL
        
        # Use user preference or guidance level
        intensity_map = {
            "minimal": FeedbackIntensity.MINIMAL,
            "standard": FeedbackIntensity.STANDARD,
            "verbose": FeedbackIntensity.VERBOSE,
            "expert": FeedbackIntensity.EXPERT
        }
        
        return intensity_map.get(user_pref, intensity_map.get(guidance_level, FeedbackIntensity.STANDARD))
    
    def _create_technical_details(self, feedback: FeedbackMessage, zen_guidance: Dict[str, Any]) -> str:
        """Create technical details based on feedback intensity."""
        if not self.user_preferences.get("show_technical_details", True):
            return ""
        
        details = []
        
        # Add complexity analysis
        complexity = zen_guidance["complexity"]
        if hasattr(complexity, 'value'):
            details.append(f"Complexity Level: {complexity.value}")
        
        # Add category information
        categories = zen_guidance.get("categories", [])
        if categories:
            details.append(f"Task Categories: {', '.join(categories)}")
        
        # Add coordination details
        coordination = zen_guidance["coordination"]
        coord_value = coordination.value if hasattr(coordination, 'value') else str(coordination)
        details.append(f"Recommended Coordination: {coord_value}")
        
        # Add tool efficiency details
        if feedback.feedback_type == FeedbackType.OPTIMIZATION:
            details.append("Efficiency Impact: High - Consider immediate optimization")
        
        return " | ".join(details) if details else ""
    
    def _create_actionable_recommendations(self, feedback: FeedbackMessage, zen_guidance: Dict[str, Any], 
                                         tool_name: str) -> List[str]:
        """Create actionable recommendations combining pattern analysis with ZEN guidance."""
        recommendations = []
        
        # Start with pattern-based recommendations
        if feedback.actionable_steps:
            recommendations.extend(feedback.actionable_steps[:2])  # Top 2 from pattern analysis
        
        # Add ZEN-specific recommendations
        recommended_tools = zen_guidance["recommended_tools"]
        if recommended_tools:
            if "mcp__zen__" in recommended_tools[0]:
                recommendations.append(f"Use {recommended_tools[0]} for intelligent analysis")
            elif "mcp__claude-flow__" in recommended_tools[0]:
                recommendations.append(f"Initialize coordination with {recommended_tools[0]}")
        
        # Add context-specific recommendations
        coordination = zen_guidance["coordination"]
        coord_value = coordination.value if hasattr(coordination, 'value') else str(coordination)
        
        if coord_value == "HIVE" and len(recommendations) < 3:
            recommendations.append("Consider persistent session coordination for complex workflows")
        elif coord_value == "SWARM" and len(recommendations) < 3:
            recommendations.append("Use parallel execution for improved performance")
        
        # Limit to 3 recommendations maximum
        return recommendations[:3]
    
    def _format_contextual_feedback(self, feedback: ContextualFeedback) -> str:
        """Format contextual feedback for stderr output."""
        lines = []
        
        # Header based on intensity
        if feedback.feedback_intensity == FeedbackIntensity.EXPERT:
            lines.append("\n" + "="*80)
            lines.append("🧠 CLAUDE CODE EXPERT INTELLIGENCE")
            lines.append("="*80)
        elif feedback.feedback_intensity == FeedbackIntensity.VERBOSE:
            lines.append("\n" + "="*70)
            lines.append("🎯 CLAUDE CODE ADVANCED FEEDBACK")
            lines.append("="*70)
        else:
            lines.append("\n" + "="*60)
            lines.append("💡 CLAUDE CODE INTELLIGENCE")
            lines.append("="*60)
        
        # Primary message
        lines.append(f"\n{feedback.primary_message}")
        
        # ZEN guidance
        if feedback.zen_guidance:
            lines.append(f"\n{feedback.zen_guidance}")
        
        # Technical details for verbose/expert modes
        if feedback.technical_details and feedback.feedback_intensity in [FeedbackIntensity.VERBOSE, FeedbackIntensity.EXPERT]:
            lines.append(f"\n🔍 Technical Analysis: {feedback.technical_details}")
        
        # Actionable recommendations
        if feedback.actionable_recommendations:
            lines.append("\n📋 Immediate Actions:")
            for i, rec in enumerate(feedback.actionable_recommendations, 1):
                lines.append(f"  {i}. {rec}")
        
        # Tool suggestions for expert mode
        if feedback.tool_suggestions and feedback.feedback_intensity == FeedbackIntensity.EXPERT:
            lines.append(f"\n🛠️ Recommended Tools: {', '.join(feedback.tool_suggestions)}")
        
        # Priority indicator
        priority_emoji = "🔥" if feedback.priority_score >= 0.8 else "⚡" if feedback.priority_score >= 0.6 else "💡"
        lines.append(f"\n{priority_emoji} Optimization Priority: {feedback.priority_score:.1f}/1.0")
        
        # Footer
        footer_length = 80 if feedback.feedback_intensity == FeedbackIntensity.EXPERT else 70 if feedback.feedback_intensity == FeedbackIntensity.VERBOSE else 60
        lines.append("="*footer_length + "\n")
        
        return "\n".join(lines)
    
    def _update_user_preferences_from_usage(self):
        """Update user preferences based on usage patterns."""
        if len(self.feedback_history) < 5:
            return
        
        # Analyze feedback response patterns
        recent_feedback = self.feedback_history[-5:]
        
        # If user is consistently getting high-priority feedback, they might want more detail
        avg_priority = sum(fb.priority_score for fb in recent_feedback) / len(recent_feedback)
        if avg_priority > 0.8 and self.user_preferences.get("feedback_intensity") == "standard":
            self.user_preferences["feedback_intensity"] = "verbose"
        
        # Save updated preferences
        self._save_user_preferences()
    
    def _save_user_preferences(self):
        """Save user preferences for future sessions."""
        try:
            pref_dir = "/home/devcontainers/flowed/.claude/hooks/.session"
            os.makedirs(pref_dir, exist_ok=True)
            
            pref_file = os.path.join(pref_dir, "feedback_preferences.json")
            with open(pref_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception:
            pass  # Silent failure for preferences
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics for monitoring and optimization."""
        if not self.feedback_history:
            return {"total_feedback": 0}
        
        feedback_types = {}
        total_priority = 0
        
        for feedback in self.feedback_history:
            feedback_type = feedback.feedback_intensity.value
            feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
            total_priority += feedback.priority_score
        
        return {
            "total_feedback": len(self.feedback_history),
            "avg_priority": total_priority / len(self.feedback_history),
            "feedback_distribution": feedback_types,
            "user_expertise": self.pattern_analyzer.user_expertise.value,
            "session_effectiveness": min(1.0, total_priority / len(self.feedback_history))
        }


# Global instance for use in post_tool_use.py
_feedback_generator: Optional[IntelligentFeedbackGenerator] = None


def get_intelligent_feedback_generator() -> IntelligentFeedbackGenerator:
    """Get or create the global intelligent feedback generator."""
    global _feedback_generator
    if _feedback_generator is None:
        _feedback_generator = IntelligentFeedbackGenerator()
    return _feedback_generator


def generate_intelligent_stderr_feedback(tool_name: str, tool_input: Dict[str, Any], 
                                        tool_response: Dict[str, Any], execution_time: float = 0.0) -> Optional[str]:
    """Generate intelligent stderr feedback for the given tool usage."""
    try:
        generator = get_intelligent_feedback_generator()
        return generator.generate_contextual_feedback(tool_name, tool_input, tool_response, execution_time)
    except Exception:
        # Fallback to no feedback on error
        return None