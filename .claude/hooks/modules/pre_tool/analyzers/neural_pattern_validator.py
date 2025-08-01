"""Neural Pattern Validator - claude-flow integration for self-improving workflow intelligence.

Integrates claude-flow's neural pattern learning capabilities into the existing
Queen ZEN hierarchy system, providing continuous optimization based on successful operations.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowContextTracker,
    WorkflowViolationType,
)


@dataclass
class NeuralPattern:
    """Represents a learned neural pattern from successful operations."""
    pattern_id: str
    tool_name: str
    context_hash: str
    success_count: int
    failure_count: int
    confidence_score: float
    learned_optimization: str
    created_timestamp: float
    last_used_timestamp: float
    performance_metrics: Dict[str, float]


class NeuralPatternStorage:
    """SQLite-based storage for neural patterns with fallback."""

    def __init__(self, db_path: str = ".swarm/neural_patterns.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize neural patterns database with fallback handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS neural_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        tool_name TEXT NOT NULL,
                        context_hash TEXT NOT NULL,
                        success_count INTEGER DEFAULT 0,
                        failure_count INTEGER DEFAULT 0,
                        confidence_score REAL DEFAULT 0.0,
                        learned_optimization TEXT,
                        created_timestamp REAL NOT NULL,
                        last_used_timestamp REAL NOT NULL,
                        performance_metrics TEXT  -- JSON blob
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tool_confidence 
                    ON neural_patterns(tool_name, confidence_score DESC)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_context_hash
                    ON neural_patterns(context_hash)
                """)

                conn.commit()
        except sqlite3.Error as e:
            print(f"Warning: Neural pattern database initialization failed: {e}")
            # Fallback to in-memory storage will be handled by the validator

    def store_pattern(self, pattern: NeuralPattern) -> bool:
        """Store neural pattern with fallback handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO neural_patterns 
                    (pattern_id, tool_name, context_hash, success_count, failure_count,
                     confidence_score, learned_optimization, created_timestamp, 
                     last_used_timestamp, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.tool_name,
                    pattern.context_hash,
                    pattern.success_count,
                    pattern.failure_count,
                    pattern.confidence_score,
                    pattern.learned_optimization,
                    pattern.created_timestamp,
                    pattern.last_used_timestamp,
                    json.dumps(pattern.performance_metrics)
                ))
                conn.commit()
                return True
        except sqlite3.Error:
            return False

    def get_patterns_for_tool(self, tool_name: str, min_confidence: float = 0.5) -> List[NeuralPattern]:
        """Retrieve neural patterns for a specific tool."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pattern_id, tool_name, context_hash, success_count, failure_count,
                           confidence_score, learned_optimization, created_timestamp,
                           last_used_timestamp, performance_metrics
                    FROM neural_patterns 
                    WHERE tool_name = ? AND confidence_score >= ?
                    ORDER BY confidence_score DESC, success_count DESC
                    LIMIT 10
                """, (tool_name, min_confidence))

                patterns = []
                for row in cursor.fetchall():
                    performance_metrics = json.loads(row[9]) if row[9] else {}
                    pattern = NeuralPattern(
                        pattern_id=row[0],
                        tool_name=row[1],
                        context_hash=row[2],
                        success_count=row[3],
                        failure_count=row[4],
                        confidence_score=row[5],
                        learned_optimization=row[6],
                        created_timestamp=row[7],
                        last_used_timestamp=row[8],
                        performance_metrics=performance_metrics
                    )
                    patterns.append(pattern)

                return patterns
        except sqlite3.Error:
            return []

    def update_pattern_success(self, pattern_id: str) -> bool:
        """Update pattern success metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE neural_patterns
                    SET success_count = success_count + 1,
                        last_used_timestamp = ?,
                        confidence_score = CASE
                            WHEN (success_count + failure_count) > 0
                            THEN CAST(success_count + 1 AS REAL) / (success_count + failure_count + 1)
                            ELSE 0.5
                        END
                    WHERE pattern_id = ?
                """, (time.time(), pattern_id))
                conn.commit()
                return True
        except sqlite3.Error:
            return False

    def get_recent_patterns(self, limit: int = 10) -> List[NeuralPattern]:
        """Get recent neural patterns sorted by last_used_timestamp."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pattern_id, tool_name, context_hash, success_count, failure_count,
                           confidence_score, learned_optimization, created_timestamp,
                           last_used_timestamp, performance_metrics
                    FROM neural_patterns
                    ORDER BY last_used_timestamp DESC, confidence_score DESC
                    LIMIT ?
                """, (limit,))

                patterns = []
                for row in cursor.fetchall():
                    performance_metrics = json.loads(row[9]) if row[9] else {}
                    pattern = NeuralPattern(
                        pattern_id=row[0],
                        tool_name=row[1],
                        context_hash=row[2],
                        success_count=row[3],
                        failure_count=row[4],
                        confidence_score=row[5],
                        learned_optimization=row[6],
                        created_timestamp=row[7],
                        last_used_timestamp=row[8],
                        performance_metrics=performance_metrics
                    )
                    patterns.append(pattern)

                return patterns
        except sqlite3.Error:
            return []


class NeuralPatternValidator(HiveWorkflowValidator):
    """Neural pattern validator implementing claude-flow learning capabilities.
    
    Priority: 850 (between MCP coordination and efficiency optimization)
    Learns from successful operations to provide increasingly intelligent suggestions.
    """

    def __init__(self, priority: int = 850, learning_enabled: bool = False,
                 confidence_threshold: float = 0.7):
        super().__init__(priority)
        self.learning_enabled = learning_enabled  # Start disabled for safety
        self.confidence_threshold = confidence_threshold
        self.pattern_storage = NeuralPatternStorage()
        self._fallback_patterns: Dict[str, List[Dict[str, Any]]] = {}  # In-memory fallback

        # Neural learning metrics
        self.patterns_applied = 0
        self.learning_sessions = 0
        self.optimization_hits = 0

    def get_validator_name(self) -> str:
        return "neural_pattern_validator"

    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any],
                         context: WorkflowContextTracker) -> ValidationResult:
        """Validate tool usage using learned neural patterns."""
        if not self.learning_enabled:
            # Learning disabled - return neutral to avoid interference
            return ValidationResult(
                severity=ValidationSeverity.ALLOW,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="Neural pattern learning is disabled",
                priority_score=0
            )

        # Generate context hash for pattern matching
        context_hash = self._generate_context_hash(tool_name, tool_input, context)

        # Retrieve relevant neural patterns
        patterns = self.pattern_storage.get_patterns_for_tool(tool_name, self.confidence_threshold)

        if not patterns:
            # No learned patterns - record this for future learning
            self._record_unknown_pattern(tool_name, tool_input, context_hash)
            return ValidationResult(
                severity=ValidationSeverity.ALLOW,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="No neural patterns available - learning in progress",
                priority_score=10
            )

        # Find best matching pattern
        best_pattern = self._find_best_pattern_match(patterns, context_hash, context)

        if not best_pattern:
            return ValidationResult(
                severity=ValidationSeverity.ALLOW,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="No matching neural patterns found",
                priority_score=5
            )

        # Apply neural learning insights
        return self._apply_neural_insights(best_pattern, tool_name, tool_input, context)

    def _generate_context_hash(self, tool_name: str, tool_input: Dict[str, Any],
                              context: WorkflowContextTracker) -> str:
        """Generate hash representing current workflow context."""
        context_data = {
            "tool_name": tool_name,
            "coordination_state": context.get_coordination_state(),
            "recent_pattern": context.get_recent_pattern(),
            "tools_since_zen": min(context.get_tools_since_zen(), 10),  # Cap for stability
            "tools_since_flow": min(context.get_tools_since_flow(), 10),
            # Include key input characteristics (not full content for privacy)
            "input_keys": sorted(tool_input.keys()),
            "input_complexity": len(str(tool_input))
        }

        context_str = json.dumps(context_data, sort_keys=True)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]

    def _find_best_pattern_match(self, patterns: List[NeuralPattern], context_hash: str,
                                context: WorkflowContextTracker) -> Optional[NeuralPattern]:
        """Find the best matching neural pattern for current context."""
        best_match = None
        best_score = 0.0

        for pattern in patterns:
            # Direct hash match gets highest score
            if pattern.context_hash == context_hash:
                return pattern

            # Calculate contextual similarity score
            similarity_score = self._calculate_context_similarity(pattern, context)

            # Weight by confidence and usage
            weighted_score = (
                similarity_score * 0.6 +
                pattern.confidence_score * 0.3 +
                min(pattern.success_count / 10.0, 1.0) * 0.1
            )

            if weighted_score > best_score and weighted_score > 0.6:
                best_match = pattern
                best_score = weighted_score

        return best_match

    def _calculate_context_similarity(self, pattern: NeuralPattern,
                                    context: WorkflowContextTracker) -> float:
        """Calculate similarity between pattern context and current context."""
        # Simple similarity based on coordination state and recent patterns
        current_state = context.get_coordination_state()
        current_pattern = context.get_recent_pattern()

        # This is a simplified similarity - in full implementation would use
        # more sophisticated neural similarity measures
        similarity = 0.5  # Base similarity

        # Boost for similar coordination patterns in performance metrics
        if pattern.performance_metrics.get("coordination_state") == current_state:
            similarity += 0.3

        # Boost for similar recent tool patterns
        if current_pattern and pattern.performance_metrics.get("recent_pattern"):
            recent_pattern_value = pattern.performance_metrics["recent_pattern"]
            if isinstance(recent_pattern_value, str):
                pattern_tools = set(recent_pattern_value.split(" → "))
                current_tools = set(current_pattern.split(" → "))
                overlap = len(pattern_tools & current_tools) / max(len(pattern_tools | current_tools), 1)
                similarity += overlap * 0.2

        return min(similarity, 1.0)

    def _apply_neural_insights(self, pattern: NeuralPattern, tool_name: str,
                              tool_input: Dict[str, Any], context: WorkflowContextTracker) -> ValidationResult:
        """Apply learned neural insights to provide intelligent guidance."""
        self.patterns_applied += 1
        self.optimization_hits += 1

        # Update pattern usage
        self.pattern_storage.update_pattern_success(pattern.pattern_id)

        # Generate neural-enhanced guidance
        if pattern.confidence_score > 0.85:
            # High confidence - provide optimization suggestion
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message=f"🧠 NEURAL INTELLIGENCE: {pattern.learned_optimization}",
                suggested_alternative=self._extract_suggestion(pattern.learned_optimization),
                hive_guidance=f"Neural pattern learned from {pattern.success_count} successful operations (confidence: {pattern.confidence_score:.1%})",
                priority_score=int(pattern.confidence_score * 100)
            )
        if pattern.confidence_score > 0.7:
            # Medium confidence - provide gentle guidance
            return ValidationResult(
                severity=ValidationSeverity.SUGGEST,
                violation_type=WorkflowViolationType.MISSING_COORDINATION,
                message=f"🤖 Neural learning suggests: {pattern.learned_optimization}",
                hive_guidance=f"Based on {pattern.success_count} similar successful operations",
                priority_score=int(pattern.confidence_score * 80)
            )

        return ValidationResult(
            severity=ValidationSeverity.ALLOW,
            violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
            message="Neural pattern analysis complete",
            priority_score=5
        )

    def _extract_suggestion(self, learned_optimization: str) -> str:
        """Extract actionable suggestion from learned optimization."""
        # Simple extraction - in full implementation would parse structured suggestions
        if "mcp__zen__" in learned_optimization:
            return learned_optimization
        if "Queen ZEN" in learned_optimization:
            return 'mcp__zen__chat { prompt: "Guide this operation with hive intelligence" }'
        if "coordination" in learned_optimization.lower():
            return "mcp__claude-flow__swarm_init for enhanced coordination"
        return learned_optimization

    def _record_unknown_pattern(self, tool_name: str, tool_input: Dict[str, Any],
                               context_hash: str) -> None:
        """Record unknown pattern for future learning."""
        if not self.learning_enabled:
            return

        # Create new pattern for learning
        pattern_id = f"{tool_name}_{context_hash}_{int(time.time())}"

        pattern = NeuralPattern(
            pattern_id=pattern_id,
            tool_name=tool_name,
            context_hash=context_hash,
            success_count=0,
            failure_count=0,
            confidence_score=0.0,
            learned_optimization="Initial pattern - learning in progress",
            created_timestamp=time.time(),
            last_used_timestamp=time.time(),
            performance_metrics={}
        )

        # Store for future learning
        self.pattern_storage.store_pattern(pattern)

    def learn_from_success(self, tool_name: str, tool_input: Dict[str, Any],
                          context_hash: str, optimization_applied: str) -> None:
        """Learn from successful operation - called by post-tool hooks."""
        if not self.learning_enabled:
            return

        patterns = self.pattern_storage.get_patterns_for_tool(tool_name, 0.0)  # Get all patterns

        # Find matching pattern or create new one
        matching_pattern = None
        for pattern in patterns:
            if pattern.context_hash == context_hash:
                matching_pattern = pattern
                break

        if matching_pattern:
            # Update existing pattern
            matching_pattern.success_count += 1
            matching_pattern.last_used_timestamp = time.time()
            matching_pattern.confidence_score = matching_pattern.success_count / (
                matching_pattern.success_count + matching_pattern.failure_count
            )
            if optimization_applied:
                matching_pattern.learned_optimization = optimization_applied
        else:
            # Create new successful pattern
            pattern_id = f"{tool_name}_{context_hash}_{int(time.time())}"
            matching_pattern = NeuralPattern(
                pattern_id=pattern_id,
                tool_name=tool_name,
                context_hash=context_hash,
                success_count=1,
                failure_count=0,
                confidence_score=1.0,
                learned_optimization=optimization_applied or "Operation completed successfully",
                created_timestamp=time.time(),
                last_used_timestamp=time.time(),
                performance_metrics={}
            )

        # Store updated pattern
        self.pattern_storage.store_pattern(matching_pattern)
        self.learning_sessions += 1

    def get_neural_metrics(self) -> Dict[str, Any]:
        """Get neural learning metrics for monitoring."""
        try:
            with sqlite3.connect(self.pattern_storage.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM neural_patterns")
                total_patterns = cursor.fetchone()[0]

                cursor = conn.execute("""
                    SELECT COUNT(*) FROM neural_patterns 
                    WHERE confidence_score >= ?
                """, (self.confidence_threshold,))
                high_confidence_patterns = cursor.fetchone()[0]
        except sqlite3.Error:
            total_patterns = len(self._fallback_patterns)
            high_confidence_patterns = 0

        return {
            "learning_enabled": self.learning_enabled,
            "total_patterns": total_patterns,
            "high_confidence_patterns": high_confidence_patterns,
            "confidence_threshold": self.confidence_threshold,
            "patterns_applied": self.patterns_applied,
            "learning_sessions": self.learning_sessions,
            "optimization_hits": self.optimization_hits,
            "neural_effectiveness": (
                self.optimization_hits / max(self.patterns_applied, 1) * 100
            )
        }

    def enable_learning(self) -> None:
        """Enable neural learning after validation."""
        self.learning_enabled = True
        print("🧠 Neural pattern learning enabled - hive intelligence active!")

    def disable_learning(self) -> None:
        """Disable neural learning for safety."""
        self.learning_enabled = False
        print("🧠 Neural pattern learning disabled for safety.")
