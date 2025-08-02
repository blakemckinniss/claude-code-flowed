#!/usr/bin/env python3
"""Tool Pattern Analyzer - Comprehensive analysis of tool usage patterns with intelligent feedback generation.

This module analyzes tool usage patterns across different categories (MCP, File ops, Search, Web) 
and generates contextually intelligent stderr feedback messages that enhance the Claude Code development experience.
"""

import json
import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta


class ToolCategory(Enum):
    """Tool categories for pattern analysis."""
    MCP_ZEN = "mcp_zen"
    MCP_CLAUDE_FLOW = "mcp_claude_flow"
    MCP_FILESYSTEM = "mcp_filesystem"
    MCP_GITHUB = "mcp_github"
    MCP_OTHER = "mcp_other"
    FILE_OPERATIONS = "file_operations"
    SEARCH_OPERATIONS = "search_operations"
    WEB_OPERATIONS = "web_operations"
    SYSTEM_OPERATIONS = "system_operations"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"


class FeedbackType(Enum):
    """Types of feedback messages."""
    OPTIMIZATION = "optimization"
    GUIDANCE = "guidance"
    WARNING = "warning"
    SUCCESS = "success"
    EDUCATIONAL = "educational"


class UserExpertiseLevel(Enum):
    """User expertise levels for adaptive feedback."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ToolUsage:
    """Individual tool usage record."""
    tool_name: str
    category: ToolCategory
    timestamp: float
    success: bool
    execution_time: float
    input_size: int
    output_size: int
    context: Dict[str, Any]


@dataclass
class UsagePattern:
    """Detected usage pattern."""
    pattern_type: str
    tool_sequence: List[str]
    frequency: int
    efficiency_score: float
    optimization_opportunities: List[str]
    success_rate: float


@dataclass
class FeedbackMessage:
    """Structured feedback message."""
    feedback_type: FeedbackType
    priority: int  # 1-10, 10 being highest
    title: str
    message: str
    actionable_steps: List[str]
    related_tools: List[str]
    expertise_level: UserExpertiseLevel
    show_technical_details: bool


class ToolPatternAnalyzer:
    """Comprehensive tool pattern analyzer with intelligent feedback generation."""
    
    # Tool categorization mappings
    TOOL_CATEGORIES = {
        # MCP Tools
        "mcp__zen__chat": ToolCategory.MCP_ZEN,
        "mcp__zen__thinkdeep": ToolCategory.MCP_ZEN,
        "mcp__zen__planner": ToolCategory.MCP_ZEN,
        "mcp__zen__consensus": ToolCategory.MCP_ZEN,
        "mcp__zen__analyze": ToolCategory.MCP_ZEN,
        "mcp__zen__debug": ToolCategory.MCP_ZEN,
        "mcp__zen__testgen": ToolCategory.MCP_ZEN,
        "mcp__zen__refactor": ToolCategory.MCP_ZEN,
        "mcp__zen__secaudit": ToolCategory.MCP_ZEN,
        "mcp__zen__docgen": ToolCategory.MCP_ZEN,
        
        "mcp__claude-flow__swarm_init": ToolCategory.MCP_CLAUDE_FLOW,
        "mcp__claude-flow__agent_spawn": ToolCategory.MCP_CLAUDE_FLOW,
        "mcp__claude-flow__task_orchestrate": ToolCategory.MCP_CLAUDE_FLOW,
        "mcp__claude-flow__memory_usage": ToolCategory.MCP_CLAUDE_FLOW,
        "mcp__claude-flow__swarm_status": ToolCategory.MCP_CLAUDE_FLOW,
        
        "mcp__filesystem__read_file": ToolCategory.MCP_FILESYSTEM,
        "mcp__filesystem__write_file": ToolCategory.MCP_FILESYSTEM,
        "mcp__filesystem__list_directory": ToolCategory.MCP_FILESYSTEM,
        "mcp__filesystem__read_multiple_files": ToolCategory.MCP_FILESYSTEM,
        "mcp__filesystem__edit_file": ToolCategory.MCP_FILESYSTEM,
        
        "mcp__github__create_pull_request": ToolCategory.MCP_GITHUB,
        "mcp__github__list_issues": ToolCategory.MCP_GITHUB,
        "mcp__github__get_file_contents": ToolCategory.MCP_GITHUB,
        "mcp__github__create_issue": ToolCategory.MCP_GITHUB,
        
        # File Operations
        "Read": ToolCategory.FILE_OPERATIONS,
        "Write": ToolCategory.FILE_OPERATIONS,
        "Edit": ToolCategory.FILE_OPERATIONS,
        "MultiEdit": ToolCategory.FILE_OPERATIONS,
        
        # Search Operations
        "Grep": ToolCategory.SEARCH_OPERATIONS,
        "Glob": ToolCategory.SEARCH_OPERATIONS,
        
        # Web Operations
        "WebSearch": ToolCategory.WEB_OPERATIONS,
        "WebFetch": ToolCategory.WEB_OPERATIONS,
        
        # System Operations
        "Bash": ToolCategory.SYSTEM_OPERATIONS,
        "Task": ToolCategory.WORKFLOW_ORCHESTRATION,
        "TodoWrite": ToolCategory.WORKFLOW_ORCHESTRATION,
    }
    
    # Pattern definitions for intelligent analysis
    EFFICIENCY_PATTERNS = {
        "sequential_file_ops": {
            "pattern": r"(Read|Write|Edit) → (Read|Write|Edit) → (Read|Write|Edit)",
            "efficiency_loss": 0.6,
            "optimization": "Use mcp__filesystem__read_multiple_files for batch operations"
        },
        "no_mcp_coordination": {
            "pattern": r"^(?!.*mcp__).*(Task|Write|Edit|Bash).*(Task|Write|Edit|Bash)",
            "efficiency_loss": 0.4,
            "optimization": "Use ZEN MCP tools for intelligent coordination"
        },
        "fragmented_workflow": {
            "pattern": r"(Bash) → (Read|Write) → (Bash) → (Read|Write)",
            "efficiency_loss": 0.5,
            "optimization": "Use mcp__zen__planner for workflow optimization"
        }
    }
    
    def __init__(self, memory_window: int = 20):
        """Initialize tool pattern analyzer."""
        self.memory_window = memory_window
        self.usage_history: List[ToolUsage] = []
        self.detected_patterns: List[UsagePattern] = []
        self.user_expertise: UserExpertiseLevel = UserExpertiseLevel.INTERMEDIATE
        self.session_context: Dict[str, Any] = {}
        self._load_session_context()
    
    def _load_session_context(self):
        """Load session context for user adaptation."""
        try:
            context_file = "/home/devcontainers/flowed/.claude/hooks/.session/tool_patterns.json"
            if os.path.exists(context_file):
                with open(context_file) as f:
                    self.session_context = json.load(f)
                    # Infer user expertise from historical patterns
                    self.user_expertise = self._infer_user_expertise()
        except Exception:
            self.session_context = {}
    
    def _infer_user_expertise(self) -> UserExpertiseLevel:
        """Infer user expertise level from usage patterns."""
        mcp_usage = self.session_context.get("mcp_tool_usage", 0)
        advanced_patterns = self.session_context.get("advanced_patterns_used", 0)
        error_rate = self.session_context.get("error_rate", 0.5)
        
        if mcp_usage > 20 and advanced_patterns > 5 and error_rate < 0.1:
            return UserExpertiseLevel.EXPERT
        elif mcp_usage > 10 and advanced_patterns > 2 and error_rate < 0.2:
            return UserExpertiseLevel.ADVANCED
        elif mcp_usage > 5 and error_rate < 0.3:
            return UserExpertiseLevel.INTERMEDIATE
        else:
            return UserExpertiseLevel.BEGINNER
    
    def categorize_tool(self, tool_name: str) -> ToolCategory:
        """Categorize a tool by its name and purpose."""
        # Direct mapping
        if tool_name in self.TOOL_CATEGORIES:
            return self.TOOL_CATEGORIES[tool_name]
        
        # Pattern-based categorization for dynamic tools
        if tool_name.startswith("mcp__zen__"):
            return ToolCategory.MCP_ZEN
        elif tool_name.startswith("mcp__claude-flow__"):
            return ToolCategory.MCP_CLAUDE_FLOW
        elif tool_name.startswith("mcp__filesystem__"):
            return ToolCategory.MCP_FILESYSTEM
        elif tool_name.startswith("mcp__github__"):
            return ToolCategory.MCP_GITHUB
        elif tool_name.startswith("mcp__"):
            return ToolCategory.MCP_OTHER
        
        # Default fallback
        return ToolCategory.SYSTEM_OPERATIONS
    
    def record_tool_usage(self, tool_name: str, tool_input: Dict[str, Any], 
                         tool_response: Dict[str, Any], execution_time: float = 0.0):
        """Record a tool usage for pattern analysis."""
        category = self.categorize_tool(tool_name)
        success = tool_response.get("success", True)
        
        usage = ToolUsage(
            tool_name=tool_name,
            category=category,
            timestamp=time.time(),
            success=success,
            execution_time=execution_time,
            input_size=len(json.dumps(tool_input)),
            output_size=len(json.dumps(tool_response)),
            context={
                "file_path": tool_input.get("file_path") or tool_input.get("path", ""),
                "operation_type": self._classify_operation(tool_name, tool_input),
                "error_details": tool_response.get("error") if not success else None
            }
        )
        
        self.usage_history.append(usage)
        
        # Maintain memory window
        if len(self.usage_history) > self.memory_window:
            self.usage_history = self.usage_history[-self.memory_window:]
    
    def _classify_operation(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Classify the type of operation being performed."""
        if tool_name == "Write":
            content_length = len(tool_input.get("content", ""))
            if content_length > 1000:
                return "large_file_creation"
            return "file_creation"
        
        elif tool_name in ["Edit", "MultiEdit"]:
            return "file_modification"
        
        elif tool_name == "Read":
            return "file_reading"
        
        elif tool_name == "Bash":
            command = tool_input.get("command", "")
            if "git" in command:
                return "version_control"
            elif "npm" in command or "pip" in command:
                return "package_management"
            elif "mkdir" in command or "cp" in command:
                return "file_system_management"
            return "system_command"
        
        elif tool_name == "Task":
            return "agent_coordination"
        
        elif tool_name.startswith("mcp__zen__"):
            return "intelligent_coordination"
        
        elif tool_name.startswith("mcp__claude-flow__"):
            return "swarm_coordination"
        
        return "general_operation"
    
    def analyze_current_patterns(self) -> List[UsagePattern]:
        """Analyze current usage patterns and detect optimization opportunities."""
        if len(self.usage_history) < 3:
            return []
        
        patterns = []
        
        # Analyze tool sequences
        recent_tools = [usage.tool_name for usage in self.usage_history[-10:]]
        tool_sequence_str = " → ".join(recent_tools)
        
        # Check against known inefficient patterns
        for pattern_name, pattern_info in self.EFFICIENCY_PATTERNS.items():
            if re.search(pattern_info["pattern"], tool_sequence_str):
                efficiency_score = 1.0 - pattern_info["efficiency_loss"]
                patterns.append(UsagePattern(
                    pattern_type=pattern_name,
                    tool_sequence=recent_tools,
                    frequency=1,
                    efficiency_score=efficiency_score,
                    optimization_opportunities=[pattern_info["optimization"]],
                    success_rate=self._calculate_success_rate(recent_tools)
                ))
        
        # Analyze category distribution
        category_pattern = self._analyze_category_distribution()
        if category_pattern:
            patterns.append(category_pattern)
        
        # Analyze timing patterns
        timing_pattern = self._analyze_timing_patterns()
        if timing_pattern:
            patterns.append(timing_pattern)
        
        self.detected_patterns = patterns
        return patterns
    
    def _analyze_category_distribution(self) -> Optional[UsagePattern]:
        """Analyze distribution of tool categories."""
        if len(self.usage_history) < 5:
            return None
        
        category_counts = {}
        for usage in self.usage_history[-10:]:
            category_counts[usage.category] = category_counts.get(usage.category, 0) + 1
        
        # Check for lack of MCP coordination
        mcp_count = sum(count for category, count in category_counts.items() 
                       if category in [ToolCategory.MCP_ZEN, ToolCategory.MCP_CLAUDE_FLOW])
        total_count = sum(category_counts.values())
        
        if total_count > 5 and mcp_count / total_count < 0.2:  # Less than 20% MCP usage
            return UsagePattern(
                pattern_type="low_mcp_coordination",
                tool_sequence=[usage.tool_name for usage in self.usage_history[-5:]],
                frequency=1,
                efficiency_score=0.6,
                optimization_opportunities=[
                    "Increase MCP tool usage for better coordination",
                    "Use mcp__zen__planner for workflow optimization"
                ],
                success_rate=self._calculate_success_rate([usage.tool_name for usage in self.usage_history[-5:]])
            )
        
        return None
    
    def _analyze_timing_patterns(self) -> Optional[UsagePattern]:
        """Analyze timing patterns for performance insights."""
        if len(self.usage_history) < 5:
            return None
        
        recent_usages = self.usage_history[-5:]
        avg_execution_time = sum(usage.execution_time for usage in recent_usages) / len(recent_usages)
        
        if avg_execution_time > 5.0:  # More than 5 seconds average
            slow_tools = [usage.tool_name for usage in recent_usages if usage.execution_time > 3.0]
            return UsagePattern(
                pattern_type="performance_degradation",
                tool_sequence=slow_tools,
                frequency=len(slow_tools),
                efficiency_score=0.4,
                optimization_opportunities=[
                    "Consider using async execution patterns",
                    "Use mcp__claude-flow__ tools for parallel processing"
                ],
                success_rate=self._calculate_success_rate(slow_tools)
            )
        
        return None
    
    def _calculate_success_rate(self, tool_names: List[str]) -> float:
        """Calculate success rate for a list of tools."""
        if not tool_names:
            return 1.0
        
        relevant_usages = [usage for usage in self.usage_history if usage.tool_name in tool_names]
        if not relevant_usages:
            return 1.0
        
        successful = sum(1 for usage in relevant_usages if usage.success)
        return successful / len(relevant_usages)
    
    def generate_intelligent_feedback(self, current_tool: str, current_input: Dict[str, Any], 
                                    current_response: Dict[str, Any]) -> List[FeedbackMessage]:
        """Generate intelligent contextual feedback based on current patterns."""
        self.record_tool_usage(current_tool, current_input, current_response)
        patterns = self.analyze_current_patterns()
        
        feedback_messages = []
        
        # Generate pattern-based feedback
        for pattern in patterns:
            message = self._create_pattern_feedback(pattern, current_tool)
            if message:
                feedback_messages.append(message)
        
        # Generate tool-specific feedback
        tool_feedback = self._create_tool_specific_feedback(current_tool, current_input, current_response)
        if tool_feedback:
            feedback_messages.append(tool_feedback)
        
        # Generate progressive guidance
        progressive_feedback = self._create_progressive_feedback()
        if progressive_feedback:
            feedback_messages.append(progressive_feedback)
        
        # Sort by priority and limit to top 3
        feedback_messages.sort(key=lambda x: x.priority, reverse=True)
        return feedback_messages[:3]
    
    def _create_pattern_feedback(self, pattern: UsagePattern, current_tool: str) -> Optional[FeedbackMessage]:
        """Create feedback message for detected patterns."""
        if pattern.pattern_type == "sequential_file_ops":
            return FeedbackMessage(
                feedback_type=FeedbackType.OPTIMIZATION,
                priority=8,
                title="🚀 File Operation Batching Opportunity",
                message=f"Detected {len(pattern.tool_sequence)} sequential file operations. You could achieve ~3x performance improvement with batch processing.",
                actionable_steps=[
                    "Use mcp__filesystem__read_multiple_files for reading multiple files",
                    "Group related file operations in a single message",
                    "Consider using mcp__zen__planner to optimize the workflow"
                ],
                related_tools=["mcp__filesystem__read_multiple_files", "mcp__zen__planner"],
                expertise_level=self.user_expertise,
                show_technical_details=self.user_expertise in [UserExpertiseLevel.ADVANCED, UserExpertiseLevel.EXPERT]
            )
        
        elif pattern.pattern_type == "no_mcp_coordination":
            return FeedbackMessage(
                feedback_type=FeedbackType.GUIDANCE,
                priority=7,
                title="💡 MCP Coordination Opportunity",
                message="Your workflow could benefit from intelligent MCP coordination for better efficiency and error handling.",
                actionable_steps=[
                    "Start with mcp__zen__analyze for task analysis" if self.user_expertise == UserExpertiseLevel.BEGINNER else "Use mcp__zen__thinkdeep for complex analysis",
                    "Initialize swarm coordination with mcp__claude-flow__swarm_init",
                    "Leverage memory management with mcp__claude-flow__memory_usage"
                ],
                related_tools=["mcp__zen__analyze", "mcp__claude-flow__swarm_init"],
                expertise_level=self.user_expertise,
                show_technical_details=self.user_expertise != UserExpertiseLevel.BEGINNER
            )
        
        elif pattern.pattern_type == "performance_degradation":
            return FeedbackMessage(
                feedback_type=FeedbackType.WARNING,
                priority=9,
                title="⚠️ Performance Alert",
                message=f"Recent operations are taking longer than usual (avg: {pattern.efficiency_score*10:.1f}s). Consider optimization.",
                actionable_steps=[
                    "Review recent tool usage for bottlenecks",
                    "Use mcp__claude-flow__ tools for parallel execution",
                    "Check system resources if the pattern continues"
                ],
                related_tools=["mcp__claude-flow__swarm_init", "mcp__zen__analyze"],
                expertise_level=self.user_expertise,
                show_technical_details=True
            )
        
        return None
    
    def _create_tool_specific_feedback(self, tool_name: str, tool_input: Dict[str, Any], 
                                     tool_response: Dict[str, Any]) -> Optional[FeedbackMessage]:
        """Create tool-specific feedback messages."""
        success = tool_response.get("success", True)
        category = self.categorize_tool(tool_name)
        
        # MCP Tool Success Feedback
        if category in [ToolCategory.MCP_ZEN, ToolCategory.MCP_CLAUDE_FLOW] and success:
            return FeedbackMessage(
                feedback_type=FeedbackType.SUCCESS,
                priority=5,
                title="✅ MCP Coordination Active",
                message=f"Successfully using {tool_name} for intelligent coordination. You're leveraging the full power of the Claude Code ecosystem!",
                actionable_steps=[
                    "Continue using MCP tools for complex workflows",
                    "Consider combining with memory management for persistent sessions"
                ],
                related_tools=[tool_name],
                expertise_level=self.user_expertise,
                show_technical_details=False
            )
        
        # File Operation Optimization
        elif category == ToolCategory.FILE_OPERATIONS and tool_name in ["Read", "Write", "Edit"]:
            file_path = tool_input.get("file_path", "")
            if file_path and file_path.endswith(".py"):
                return FeedbackMessage(
                    feedback_type=FeedbackType.EDUCATIONAL,
                    priority=4,
                    title="🐍 Python File Operation",
                    message="Working with Python files. Consider using mcp__filesystem__ tools for enhanced file operations with better error handling.",
                    actionable_steps=[
                        "Use mcp__filesystem__read_file for robust file reading",
                        "Use mcp__filesystem__edit_file for safer modifications",
                        "Consider code analysis with mcp__zen__analyze"
                    ],
                    related_tools=["mcp__filesystem__read_file", "mcp__zen__analyze"],
                    expertise_level=self.user_expertise,
                    show_technical_details=self.user_expertise != UserExpertiseLevel.BEGINNER
                )
        
        # Error Handling Feedback
        elif not success:
            tool_response.get("error", "Unknown error")
            return FeedbackMessage(
                feedback_type=FeedbackType.WARNING,
                priority=8,
                title="🔧 Error Recovery Suggestion",
                message=f"Tool {tool_name} encountered an error. Here's how to handle it intelligently:",
                actionable_steps=[
                    "Use mcp__zen__debug for systematic error analysis",
                    "Check tool input parameters for correctness",
                    "Consider alternative approaches with MCP tools"
                ],
                related_tools=["mcp__zen__debug", "mcp__zen__analyze"],
                expertise_level=self.user_expertise,
                show_technical_details=True
            )
        
        return None
    
    def _create_progressive_feedback(self) -> Optional[FeedbackMessage]:
        """Create progressive feedback based on user expertise level."""
        if self.user_expertise == UserExpertiseLevel.BEGINNER and len(self.usage_history) > 5:
            mcp_usage = sum(1 for usage in self.usage_history[-10:] 
                           if usage.category in [ToolCategory.MCP_ZEN, ToolCategory.MCP_CLAUDE_FLOW])
            
            if mcp_usage == 0:
                return FeedbackMessage(
                    feedback_type=FeedbackType.EDUCATIONAL,
                    priority=6,
                    title="🎓 Unlock Advanced Capabilities",
                    message="You're using basic tools effectively! Ready to unlock advanced MCP capabilities for 10x productivity?",
                    actionable_steps=[
                        "Try mcp__zen__chat for interactive guidance",
                        "Use mcp__zen__analyze to understand complex code",
                        "Explore mcp__claude-flow__swarm_init for multi-agent workflows"
                    ],
                    related_tools=["mcp__zen__chat", "mcp__zen__analyze"],
                    expertise_level=self.user_expertise,
                    show_technical_details=False
                )
        
        elif self.user_expertise == UserExpertiseLevel.EXPERT:
            # Advanced optimization suggestions for experts
            recent_patterns = [pattern.pattern_type for pattern in self.detected_patterns]
            if "performance_degradation" not in recent_patterns and len(self.usage_history) > 10:
                return FeedbackMessage(
                    feedback_type=FeedbackType.OPTIMIZATION,
                    priority=3,
                    title="🔬 Expert Mode: Advanced Optimization",
                    message="Your workflow efficiency is excellent. Consider these cutting-edge optimizations:",
                    actionable_steps=[
                        "Implement custom MCP orchestration patterns",
                        "Use mcp__claude-flow__memory_usage for cross-session persistence",
                        "Explore neural pattern optimization with adaptive learning"
                    ],
                    related_tools=["mcp__claude-flow__memory_usage", "mcp__zen__consensus"],
                    expertise_level=self.user_expertise,
                    show_technical_details=True
                )
        
        return None
    
    def format_feedback_for_stderr(self, feedback_messages: List[FeedbackMessage]) -> str:
        """Format feedback messages for stderr output."""
        if not feedback_messages:
            return ""
        
        lines = []
        lines.append("\n" + "="*70)
        lines.append("🧠 CLAUDE CODE INTELLIGENCE FEEDBACK")
        lines.append("="*70)
        
        for i, feedback in enumerate(feedback_messages, 1):
            lines.append(f"\n{feedback.title}")
            lines.append("-" * len(feedback.title))
            lines.append(feedback.message)
            
            if feedback.actionable_steps:
                lines.append("\n📋 Next Steps:")
                for step in feedback.actionable_steps:
                    lines.append(f"  • {step}")
            
            if feedback.related_tools and feedback.show_technical_details:
                lines.append(f"\n🔧 Related Tools: {', '.join(feedback.related_tools)}")
            
            if i < len(feedback_messages):
                lines.append("")  # Spacing between messages
        
        lines.append("\n" + "="*70)
        lines.append("💡 Tip: This feedback adapts to your expertise level and usage patterns")
        lines.append("="*70 + "\n")
        
        return "\n".join(lines)
    
    def save_session_context(self):
        """Save session context for future adaptation."""
        context_dir = "/home/devcontainers/flowed/.claude/hooks/.session"
        os.makedirs(context_dir, exist_ok=True)
        
        context_file = os.path.join(context_dir, "tool_patterns.json")
        
        # Update session context
        mcp_usage = sum(1 for usage in self.usage_history 
                       if usage.category in [ToolCategory.MCP_ZEN, ToolCategory.MCP_CLAUDE_FLOW])
        error_count = sum(1 for usage in self.usage_history if not usage.success)
        
        self.session_context.update({
            "mcp_tool_usage": self.session_context.get("mcp_tool_usage", 0) + mcp_usage,
            "total_tool_usage": self.session_context.get("total_tool_usage", 0) + len(self.usage_history),
            "error_rate": error_count / len(self.usage_history) if self.usage_history else 0,
            "advanced_patterns_used": len(self.detected_patterns),
            "last_session": datetime.now().isoformat(),
            "expertise_level": self.user_expertise.value
        })
        
        try:
            with open(context_file, 'w') as f:
                json.dump(self.session_context, f, indent=2)
        except Exception:
            pass  # Silent failure for session context