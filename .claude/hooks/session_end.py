#!/usr/bin/env python3
"""Enhanced Session End Hook - Claude Flow Integration Phase 2

Comprehensive session cleanup with neural pattern learning storage,
performance metrics analysis, and workflow continuity preparation.
"""

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Define hooks directory for local usage
hooks_dir = Path(__file__).parent

# Import neural pattern classes with fallback
_HAS_NEURAL_MODULE = False

try:
    from modules.pre_tool.analyzers.neural_pattern_validator import (
        NeuralPattern as ImportedPattern,
    )
    from modules.pre_tool.analyzers.neural_pattern_validator import (
        NeuralPatternStorage as ImportedPatternStorage,
    )
    _HAS_NEURAL_MODULE = True
    # Use imported classes
    PatternStorageClass = ImportedPatternStorage
    PatternClass = ImportedPattern
except ImportError:
    # Fallback classes if modules aren't available
    from dataclasses import dataclass

    @dataclass
    class FallbackPattern:
        """Fallback NeuralPattern for when module isn't available."""
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

    class FallbackPatternStorage:
        """Fallback NeuralPatternStorage with compatible interface."""

        def __init__(self, db_path: Optional[str] = None):
            self._patterns: List[FallbackPattern] = []

        def store_pattern(self, pattern: FallbackPattern) -> bool:
            """Store pattern with compatible interface."""
            # Simple in-memory storage for fallback
            if isinstance(pattern, FallbackPattern):
                self._patterns.append(pattern)
                return True
            return False

        def get_recent_patterns(self, limit: int = 20) -> List[FallbackPattern]:
            """Get recent patterns with compatible interface."""
            return self._patterns[-limit:] if self._patterns else []

    # Use fallback classes
    PatternStorageClass = FallbackPatternStorage
    PatternClass = FallbackPattern

# Import memory integration
try:
    from modules.memory.hook_memory_integration import get_hook_memory_integration
    MEMORY_INTEGRATION = True
except ImportError:
    get_hook_memory_integration = None
    MEMORY_INTEGRATION = False


class SessionEndCoordinator:
    """Coordinates comprehensive session cleanup and learning storage."""

    def __init__(self):
        self.hooks_dir = Path(__file__).parent
        self.session_db_path = self.hooks_dir / ".session" / "session_state.db"
        self.neural_storage = PatternStorageClass()
        self.current_session_id = self._get_current_session_id()

    def _get_current_session_id(self) -> Optional[str]:
        """Get current session ID from database."""
        try:
            if not self.session_db_path.exists():
                return None

            with sqlite3.connect(self.session_db_path) as conn:
                cursor = conn.execute("""
                    SELECT session_id FROM session_states 
                    WHERE end_time IS NULL 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                return result[0] if result else None

        except Exception:
            return None

    def finalize_session(self) -> Dict[str, Any]:
        """Finalize session with comprehensive cleanup and learning."""
        session_summary = {
            "session_id": self.current_session_id,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "neural_patterns_learned": 0,
            "performance_score": 0.0,
            "workflow_efficiency": 0.0,
            "recommendations": [],
            "memories_persisted": 0
        }

        if not self.current_session_id:
            print("⚠️ No active session found", file=sys.stderr)
            return session_summary
        
        # Persist session memories
        if MEMORY_INTEGRATION and get_hook_memory_integration:
            try:
                memory_integration = get_hook_memory_integration()
                # Capture session end summary
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        memory_integration.capture_session_end_memory(session_summary)
                    )
                    # Get memory stats
                    memory_stats = memory_integration.get_session_stats()
                    session_summary["memories_persisted"] = memory_stats.get("memories_created", 0)
                finally:
                    loop.close()
                print(f"💾 Persisted {session_summary['memories_persisted']} memories to project namespace", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to persist session memories: {e}", file=sys.stderr)

        # Analyze session performance
        session_summary["performance_score"] = self._analyze_session_performance()

        # Store learned neural patterns
        session_summary["neural_patterns_learned"] = self._store_neural_learnings()

        # Calculate workflow efficiency
        session_summary["workflow_efficiency"] = self._calculate_workflow_efficiency()

        # Generate recommendations for next session
        session_summary["recommendations"] = self._generate_recommendations()

        # Update session database
        self._finalize_session_db(session_summary)

        # Prepare continuity data for next session
        self._prepare_session_continuity(session_summary)

        return session_summary

    def _analyze_session_performance(self) -> float:
        """Analyze overall session performance."""
        try:
            if not self.session_db_path.exists():
                return 0.0

            with sqlite3.connect(self.session_db_path) as conn:
                # Get session metrics
                cursor = conn.execute("""
                    SELECT metric_name, AVG(metric_value) as avg_value
                    FROM session_metrics 
                    WHERE session_id = ?
                    GROUP BY metric_name
                """, (self.current_session_id,))

                metrics = dict(cursor.fetchall())

                # Calculate weighted performance score
                performance_factors = {
                    "tool_efficiency": metrics.get("tool_efficiency", 0.5) * 0.3,
                    "coordination_ratio": metrics.get("coordination_ratio", 0.3) * 0.25,
                    "error_rate": (1.0 - metrics.get("error_rate", 0.1)) * 0.2,
                    "zen_usage": metrics.get("zen_usage", 0.2) * 0.15,
                    "workflow_optimization": metrics.get("workflow_optimization", 0.4) * 0.1
                }

                performance_score = sum(performance_factors.values())
                return min(1.0, max(0.0, performance_score))

        except Exception as e:
            print(f"Warning: Could not analyze performance: {e}", file=sys.stderr)
            return 0.5

    def _store_neural_learnings(self) -> int:
        """Store neural patterns learned during this session."""
        try:
            patterns_stored = 0

            # Get successful operations from session
            successful_operations = self._get_successful_operations()

            for operation in successful_operations:
                pattern_data = {
                    "tool_name": operation.get("tool_name", ""),
                    "context": operation.get("context", {}),
                    "success_metrics": operation.get("metrics", {}),
                    "session_id": self.current_session_id
                }

                # Create pattern object and store it
                pattern = PatternClass(
                    pattern_id=f"session_{self.current_session_id}_{patterns_stored}",
                    tool_name=pattern_data.get("tool_name", "unknown"),
                    context_hash=f"session_{self.current_session_id}",
                    success_count=1,
                    failure_count=0,
                    confidence_score=0.8,
                    learned_optimization=f"Session learning: {pattern_data.get('tool_name', 'operation')} successful",
                    created_timestamp=time.time(),
                    last_used_timestamp=time.time(),
                    performance_metrics=pattern_data.get("success_metrics", {})
                )

                # Store in neural pattern storage
                self.neural_storage.store_pattern(pattern)  # type: ignore[arg-type]
                patterns_stored += 1

            return patterns_stored

        except Exception as e:
            print(f"Warning: Could not store neural learnings: {e}", file=sys.stderr)
            return 0

    def _get_successful_operations(self) -> List[Dict[str, Any]]:
        """Get successful operations from current session."""
        try:
            if not self.session_db_path.exists():
                return []

            with sqlite3.connect(self.session_db_path) as conn:
                cursor = conn.execute("""
                    SELECT metric_name, metric_value 
                    FROM session_metrics 
                    WHERE session_id = ? AND metric_value > 0.7
                """, (self.current_session_id,))

                # Convert metrics to operation patterns
                operations = []
                for metric_name, value in cursor.fetchall():
                    operations.append({
                        "tool_name": metric_name.replace("_efficiency", ""),
                        "context": {"session_id": self.current_session_id},
                        "metrics": {"success_rate": value}
                    })

                return operations

        except Exception:
            return []

    def _calculate_workflow_efficiency(self) -> float:
        """Calculate overall workflow efficiency."""
        try:
            if not self.session_db_path.exists():
                return 0.0

            with sqlite3.connect(self.session_db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        tools_used,
                        zen_coordination_count,
                        flow_coordination_count,
                        github_operations
                    FROM session_states 
                    WHERE session_id = ?
                """, (self.current_session_id,))

                result = cursor.fetchone()
                if not result:
                    return 0.0

                tools_used, zen_count, flow_count, github_ops = result

                if tools_used == 0:
                    return 0.0

                # Calculate efficiency metrics
                coordination_ratio = (zen_count + flow_count) / tools_used
                github_efficiency = github_ops / max(tools_used, 1)

                # Weighted efficiency score
                efficiency = (
                    coordination_ratio * 0.6 +
                    github_efficiency * 0.4
                )

                return min(1.0, efficiency)

        except Exception as e:
            print(f"Warning: Could not calculate efficiency: {e}", file=sys.stderr)
            return 0.0

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for next session."""
        recommendations = []

        try:
            performance_score = self._analyze_session_performance()
            efficiency_score = self._calculate_workflow_efficiency()

            # Performance-based recommendations
            if performance_score < 0.6:
                recommendations.append("🎯 Consider using more Queen ZEN coordination for complex operations")
                recommendations.append("🔧 Use MCP tools instead of native tools for better efficiency")

            if efficiency_score < 0.5:
                recommendations.append("⚡ Batch related operations in single messages for better performance")
                recommendations.append("🧠 Enable neural pattern learning for workflow optimization")

            # GitHub-specific recommendations
            if self._has_github_context():
                recommendations.append("🐙 Utilize GitHub coordination analyzers for repository operations")
                recommendations.append("📋 Consider using GitHub swarm coordination for complex PR workflows")

            # Always include best practices
            recommendations.append("👑 Remember: MCP tools coordinate, Claude Code executes")
            recommendations.append("🚀 Use concurrent execution patterns for optimal performance")

        except Exception as e:
            print(f"Warning: Could not generate recommendations: {e}", file=sys.stderr)
            recommendations.append("📚 Review session logs for optimization opportunities")

        return recommendations

    def _has_github_context(self) -> bool:
        """Check if session involved GitHub operations."""
        try:
            if not self.session_db_path.exists():
                return False

            with sqlite3.connect(self.session_db_path) as conn:
                cursor = conn.execute("""
                    SELECT github_operations 
                    FROM session_states 
                    WHERE session_id = ?
                """, (self.current_session_id,))

                result = cursor.fetchone()
                return result and result[0] > 0

        except Exception:
            return False

    def _finalize_session_db(self, session_summary: Dict[str, Any]) -> None:
        """Update session database with final information."""
        try:
            with sqlite3.connect(self.session_db_path) as conn:
                conn.execute("""
                    UPDATE session_states 
                    SET 
                        end_time = ?,
                        performance_score = ?,
                        neural_patterns_learned = ?
                    WHERE session_id = ?
                """, (
                    session_summary["end_time"],
                    session_summary["performance_score"],
                    session_summary["neural_patterns_learned"],
                    self.current_session_id
                ))

        except Exception as e:
            print(f"Warning: Could not finalize session database: {e}", file=sys.stderr)

    def _prepare_session_continuity(self, session_summary: Dict[str, Any]) -> None:
        """Prepare continuity data for next session."""
        try:
            continuity_data = {
                "previous_performance": session_summary["performance_score"],
                "learned_patterns": session_summary["neural_patterns_learned"],
                "recommendations": session_summary["recommendations"],
                "github_context": self._has_github_context()
            }

            # Store for next session to pick up
            continuity_file = self.hooks_dir / ".session" / "continuity.json"
            continuity_file.parent.mkdir(exist_ok=True)
            with open(continuity_file, "w") as f:
                json.dump(continuity_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not prepare continuity data: {e}", file=sys.stderr)

    def display_session_summary(self, session_summary: Dict[str, Any]) -> None:
        """Display comprehensive session summary."""
        print("📊 Session Summary - Queen ZEN's Hive Intelligence", file=sys.stderr)
        print("=" * 55, file=sys.stderr)
        print(f"Session ID: {session_summary['session_id']}", file=sys.stderr)
        print(f"Performance Score: {session_summary['performance_score']:.2f}/1.00", file=sys.stderr)
        print(f"Workflow Efficiency: {session_summary['workflow_efficiency']:.2f}/1.00", file=sys.stderr)
        print(f"Neural Patterns Learned: {session_summary['neural_patterns_learned']}", file=sys.stderr)
        print("", file=sys.stderr)

        if session_summary["recommendations"]:
            print("🎯 Recommendations for Next Session:", file=sys.stderr)
            for rec in session_summary["recommendations"]:
                print(f"   {rec}", file=sys.stderr)
            print("", file=sys.stderr)

        print("👑 Queen ZEN's wisdom has been preserved for future sessions!", file=sys.stderr)


def main():
    """Main session end execution - maintain hook compatibility."""
    try:
        # Check if we're being called as a Claude Code hook
        if len(sys.argv) > 1 or not sys.stdin.isatty():
            # Hook mode - maintain JSON compatibility
            try:
                json.load(sys.stdin)
            except (json.JSONDecodeError, EOFError):
                pass

            # Perform enhanced session end processing
            coordinator = SessionEndCoordinator()
            session_summary = coordinator.finalize_session()

            # Create hook-compatible output
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionEnd",
                    "sessionSummary": session_summary,
                    "message": f"👑 Session completed - Performance: {session_summary['performance_score']:.2f}"
                }
            }

            print(json.dumps(output))
        else:
            # Direct execution mode
            coordinator = SessionEndCoordinator()
            session_summary = coordinator.finalize_session()
            coordinator.display_session_summary(session_summary)

        sys.exit(0)

    except Exception as e:
        print(f"Error during session end: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
