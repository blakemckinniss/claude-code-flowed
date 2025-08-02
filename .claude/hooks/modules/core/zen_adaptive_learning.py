#!/usr/bin/env python3
"""ZEN Adaptive Learning System - Integration with Neural Training Pipeline.

This module extends the existing neural training infrastructure to enable ZEN
to learn from consultation outcomes and improve recommendations over time.

Key Features:
- Real-time learning from ZEN consultation results
- Pattern-based agent allocation learning
- Context-aware recommendation improvements  
- Memory integration for cross-session learning
- Model updating for continuous improvement
"""

import json
import sqlite3
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Import existing neural infrastructure
from ..pre_tool.analyzers.neural_pattern_validator import NeuralPattern, NeuralPatternStorage
from .zen_consultant import ZenConsultant, ComplexityLevel, CoordinationType, AgentAllocation


@dataclass
class ZenLearningOutcome:
    """Represents a ZEN consultation outcome for learning purposes."""
    consultation_id: str
    prompt: str
    complexity: str
    coordination_type: str
    agents_allocated: int
    agent_types: List[str]
    mcp_tools: List[str]
    execution_success: bool
    user_satisfaction: float  # 0.0 to 1.0
    actual_agents_needed: int
    performance_metrics: Dict[str, float]
    lessons_learned: List[str]
    timestamp: float


@dataclass
class ZenLearningPattern:
    """Learned pattern from ZEN consultation outcomes."""
    pattern_id: str
    prompt_characteristics: Dict[str, Any]
    optimal_complexity: str
    optimal_coordination: str
    optimal_agent_count: int
    optimal_agent_types: List[str]
    success_rate: float
    confidence_score: float
    usage_count: int
    created_at: float
    updated_at: float


class ZenAdaptiveLearningEngine:
    """Main engine for ZEN adaptive learning integrated with neural training."""
    
    def __init__(self, db_path: str = ".claude/hooks/db/zen_learning.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_learning_database()
        
        # Integration with existing neural system
        self.neural_storage = NeuralPatternStorage()
        self.zen_consultant = ZenConsultant()
        
        # Learning metrics
        self.total_consultations = 0
        self.successful_learnings = 0
        self.pattern_improvements = 0
        
    def _init_learning_database(self) -> None:
        """Initialize ZEN learning database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # ZEN consultation outcomes table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS zen_outcomes (
                        consultation_id TEXT PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        complexity TEXT NOT NULL,
                        coordination_type TEXT NOT NULL,
                        agents_allocated INTEGER NOT NULL,
                        agent_types TEXT NOT NULL,  -- JSON array
                        mcp_tools TEXT NOT NULL,    -- JSON array
                        execution_success BOOLEAN NOT NULL,
                        user_satisfaction REAL NOT NULL,
                        actual_agents_needed INTEGER,
                        performance_metrics TEXT,   -- JSON object
                        lessons_learned TEXT,       -- JSON array
                        timestamp REAL NOT NULL
                    )
                """)
                
                # ZEN learning patterns table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS zen_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        prompt_characteristics TEXT NOT NULL,  -- JSON object
                        optimal_complexity TEXT NOT NULL,
                        optimal_coordination TEXT NOT NULL,
                        optimal_agent_count INTEGER NOT NULL,
                        optimal_agent_types TEXT NOT NULL,     -- JSON array
                        success_rate REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        usage_count INTEGER DEFAULT 0,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
                
                # Performance optimization indices
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_zen_outcomes_complexity
                    ON zen_outcomes(complexity, execution_success)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_zen_patterns_confidence
                    ON zen_patterns(confidence_score DESC, success_rate DESC)
                """)
                
                conn.commit()
                
        except sqlite3.Error as e:
            print(f"Warning: ZEN learning database initialization failed: {e}")
    
    def record_consultation_outcome(self, outcome: ZenLearningOutcome) -> bool:
        """Record a ZEN consultation outcome for learning."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO zen_outcomes
                    (consultation_id, prompt, complexity, coordination_type,
                     agents_allocated, agent_types, mcp_tools, execution_success,
                     user_satisfaction, actual_agents_needed, performance_metrics,
                     lessons_learned, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    outcome.consultation_id,
                    outcome.prompt,
                    outcome.complexity,
                    outcome.coordination_type,
                    outcome.agents_allocated,
                    json.dumps(outcome.agent_types),
                    json.dumps(outcome.mcp_tools),
                    outcome.execution_success,
                    outcome.user_satisfaction,
                    outcome.actual_agents_needed,
                    json.dumps(outcome.performance_metrics),
                    json.dumps(outcome.lessons_learned),
                    outcome.timestamp
                ))
                conn.commit()
                
            self.total_consultations += 1
            
            # Trigger pattern learning if outcome was successful
            if outcome.execution_success and outcome.user_satisfaction > 0.6:
                self._learn_from_successful_outcome(outcome)
                
            return True
            
        except sqlite3.Error as e:
            print(f"Failed to record ZEN consultation outcome: {e}")
            return False
    
    def _learn_from_successful_outcome(self, outcome: ZenLearningOutcome) -> None:
        """Learn patterns from successful consultation outcomes."""
        try:
            # Extract prompt characteristics for pattern matching
            characteristics = self._extract_prompt_characteristics(outcome.prompt)
            
            # Find existing pattern or create new one
            pattern = self._find_or_create_pattern(characteristics, outcome)
            
            if pattern:
                # Update pattern with new outcome
                self._update_learning_pattern(pattern, outcome)
                self.successful_learnings += 1
                
                # Also integrate with existing neural training
                self._integrate_with_neural_training(outcome, pattern)
                
        except Exception as e:
            print(f"Error learning from ZEN outcome: {e}")
    
    def _extract_prompt_characteristics(self, prompt: str) -> Dict[str, Any]:
        """Extract characteristics from prompt for pattern matching."""
        prompt_lower = prompt.lower()
        words = prompt.split()
        
        # Categorize prompt patterns
        categories = []
        complexity_indicators = []
        action_verbs = []
        
        # Development categories
        dev_keywords = {
            "code": ["code", "implement", "build", "create", "develop"],
            "test": ["test", "qa", "quality", "testing"],
            "debug": ["debug", "fix", "error", "issue", "problem", "bug"],
            "refactor": ["refactor", "clean", "improve", "optimize", "restructure"],
            "architecture": ["architecture", "design", "system", "structure"],
            "security": ["security", "audit", "vulnerability", "secure"],
            "performance": ["performance", "speed", "efficient", "optimize"],
            "documentation": ["document", "docs", "readme", "guide"],
            "deployment": ["deploy", "release", "production", "ci/cd"]
        }
        
        for category, keywords in dev_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                categories.append(category)
        
        # Complexity indicators
        simple_indicators = ["fix", "update", "add", "remove", "change"]
        complex_indicators = ["system", "architecture", "enterprise", "migrate", "scalable"]
        
        for indicator in simple_indicators:
            if indicator in prompt_lower:
                complexity_indicators.append("simple")
        
        for indicator in complex_indicators:
            if indicator in prompt_lower:
                complexity_indicators.append("complex")
        
        # Extract action verbs
        common_verbs = ["build", "create", "implement", "fix", "update", "refactor", "test", "deploy"]
        for verb in common_verbs:
            if verb in prompt_lower:
                action_verbs.append(verb)
        
        return {
            "word_count": len(words),
            "categories": categories,
            "complexity_indicators": complexity_indicators,
            "action_verbs": action_verbs,
            "has_multiple_tasks": "and" in prompt_lower or "," in prompt,
            "mentions_agents": "agent" in prompt_lower,
            "mentions_coordination": any(word in prompt_lower for word in ["coordinate", "orchestrate", "manage"]),
            "urgency_level": self._assess_urgency(prompt_lower)
        }
    
    def _assess_urgency(self, prompt_lower: str) -> str:
        """Assess urgency level from prompt language."""
        urgent_keywords = ["urgent", "asap", "immediately", "critical", "emergency"]
        normal_keywords = ["help", "need", "want", "would like"]
        
        if any(keyword in prompt_lower for keyword in urgent_keywords):
            return "high"
        elif any(keyword in prompt_lower for keyword in normal_keywords):
            return "normal"
        else:
            return "low"
    
    def _find_or_create_pattern(self, characteristics: Dict[str, Any], 
                               outcome: ZenLearningOutcome) -> Optional[ZenLearningPattern]:
        """Find existing pattern or create new one."""
        # Generate pattern hash for matching
        pattern_hash = self._generate_pattern_hash(characteristics)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM zen_patterns WHERE pattern_id = ?
                """, (pattern_hash,))
                
                row = cursor.fetchone()
                
                if row:
                    # Return existing pattern
                    return ZenLearningPattern(
                        pattern_id=row[0],
                        prompt_characteristics=json.loads(row[1]),
                        optimal_complexity=row[2],
                        optimal_coordination=row[3],
                        optimal_agent_count=row[4],
                        optimal_agent_types=json.loads(row[5]),
                        success_rate=row[6],
                        confidence_score=row[7],
                        usage_count=row[8],
                        created_at=row[9],
                        updated_at=row[10]
                    )
                else:
                    # Create new pattern
                    new_pattern = ZenLearningPattern(
                        pattern_id=pattern_hash,
                        prompt_characteristics=characteristics,
                        optimal_complexity=outcome.complexity,
                        optimal_coordination=outcome.coordination_type,
                        optimal_agent_count=outcome.actual_agents_needed or outcome.agents_allocated,
                        optimal_agent_types=outcome.agent_types,
                        success_rate=1.0,
                        confidence_score=outcome.user_satisfaction,
                        usage_count=1,
                        created_at=time.time(),
                        updated_at=time.time()
                    )
                    
                    # Store new pattern
                    self._store_learning_pattern(new_pattern)
                    return new_pattern
                    
        except sqlite3.Error:
            return None
    
    def _generate_pattern_hash(self, characteristics: Dict[str, Any]) -> str:
        """Generate consistent hash for pattern characteristics."""
        # Create stable representation
        stable_chars = {
            "categories": sorted(characteristics.get("categories", [])),
            "complexity_indicators": sorted(characteristics.get("complexity_indicators", [])),
            "action_verbs": sorted(characteristics.get("action_verbs", [])),
            "urgency_level": characteristics.get("urgency_level", "normal"),
            "has_multiple_tasks": characteristics.get("has_multiple_tasks", False)
        }
        
        stable_str = json.dumps(stable_chars, sort_keys=True)
        return hashlib.sha256(stable_str.encode()).hexdigest()[:16]
    
    def _store_learning_pattern(self, pattern: ZenLearningPattern) -> bool:
        """Store learning pattern in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO zen_patterns
                    (pattern_id, prompt_characteristics, optimal_complexity,
                     optimal_coordination, optimal_agent_count, optimal_agent_types,
                     success_rate, confidence_score, usage_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    json.dumps(pattern.prompt_characteristics),
                    pattern.optimal_complexity,
                    pattern.optimal_coordination,
                    pattern.optimal_agent_count,
                    json.dumps(pattern.optimal_agent_types),
                    pattern.success_rate,
                    pattern.confidence_score,
                    pattern.usage_count,
                    pattern.created_at,
                    pattern.updated_at
                ))
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            print(f"Failed to store learning pattern: {e}")
            return False
    
    def _update_learning_pattern(self, pattern: ZenLearningPattern, 
                                outcome: ZenLearningOutcome) -> None:
        """Update existing pattern with new outcome."""
        # Update usage count
        pattern.usage_count += 1
        pattern.updated_at = time.time()
        
        # Update success rate (exponential moving average)
        alpha = 0.2  # Learning rate
        new_success = 1.0 if outcome.execution_success else 0.0
        pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * new_success
        
        # Update confidence (weighted by user satisfaction)
        pattern.confidence_score = (
            0.7 * pattern.confidence_score + 0.3 * outcome.user_satisfaction
        )
        
        # Adjust optimal values based on actual needs
        if outcome.actual_agents_needed and outcome.execution_success:
            # Use exponential moving average for agent count optimization
            pattern.optimal_agent_count = int(
                0.8 * pattern.optimal_agent_count + 0.2 * outcome.actual_agents_needed
            )
            
            # Update agent types if new combination was more successful
            if outcome.user_satisfaction > pattern.confidence_score:
                pattern.optimal_agent_types = outcome.agent_types
        
        # Store updated pattern
        self._store_learning_pattern(pattern)
        self.pattern_improvements += 1
    
    def _integrate_with_neural_training(self, outcome: ZenLearningOutcome, 
                                      pattern: ZenLearningPattern) -> None:
        """Integrate ZEN learning with existing neural training pipeline."""
        try:
            # Create neural pattern for this ZEN consultation
            context_data = {
                "tool_name": "zen_consultation",
                "complexity": outcome.complexity,
                "coordination": outcome.coordination_type,
                "success": outcome.execution_success,
                "satisfaction": outcome.user_satisfaction
            }
            
            context_str = json.dumps(context_data, sort_keys=True)
            context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
            
            # Generate learned optimization message
            optimization = self._generate_optimization_message(outcome, pattern)
            
            # Create neural pattern
            neural_pattern = NeuralPattern(
                pattern_id=f"zen_{outcome.consultation_id}",
                tool_name="zen_consultation",
                context_hash=context_hash,
                success_count=1 if outcome.execution_success else 0,
                failure_count=0 if outcome.execution_success else 1,
                confidence_score=outcome.user_satisfaction,
                learned_optimization=optimization,
                created_timestamp=outcome.timestamp,
                last_used_timestamp=outcome.timestamp,
                performance_metrics={
                    "complexity": outcome.complexity,
                    "coordination": outcome.coordination_type,
                    "agents_allocated": outcome.agents_allocated,
                    "actual_agents_needed": outcome.actual_agents_needed or outcome.agents_allocated,
                    "user_satisfaction": outcome.user_satisfaction
                }
            )
            
            # Store in neural system
            self.neural_storage.store_pattern(neural_pattern)
            
        except Exception as e:
            print(f"Failed to integrate with neural training: {e}")
    
    def _generate_optimization_message(self, outcome: ZenLearningOutcome, 
                                     pattern: ZenLearningPattern) -> str:
        """Generate optimization message from learning outcome."""
        if outcome.user_satisfaction > 0.8:
            return f"ZEN consultation optimized: {pattern.optimal_coordination} with {pattern.optimal_agent_count} agents achieved {outcome.user_satisfaction*100:.0f}% satisfaction"
        elif outcome.actual_agents_needed and outcome.actual_agents_needed != outcome.agents_allocated:
            diff = outcome.actual_agents_needed - outcome.agents_allocated
            if diff > 0:
                return f"ZEN learning: Allocate {diff} additional agents for similar tasks"
            else:
                return f"ZEN learning: {abs(diff)} fewer agents sufficient for similar tasks"
        else:
            return f"ZEN consultation completed with {outcome.coordination_type} coordination"
    
    def get_adaptive_recommendation(self, prompt: str) -> Dict[str, Any]:
        """Get adaptive recommendation based on learned patterns."""
        characteristics = self._extract_prompt_characteristics(prompt)
        pattern_hash = self._generate_pattern_hash(characteristics)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Look for exact pattern match first
                cursor = conn.execute("""
                    SELECT * FROM zen_patterns 
                    WHERE pattern_id = ? AND confidence_score > 0.6
                """, (pattern_hash,))
                
                row = cursor.fetchone()
                
                if row:
                    # Use learned pattern
                    pattern = ZenLearningPattern(
                        pattern_id=row[0],
                        prompt_characteristics=json.loads(row[1]),
                        optimal_complexity=row[2],
                        optimal_coordination=row[3],
                        optimal_agent_count=row[4],
                        optimal_agent_types=json.loads(row[5]),
                        success_rate=row[6],
                        confidence_score=row[7],
                        usage_count=row[8],
                        created_at=row[9],
                        updated_at=row[10]
                    )
                    
                    return {
                        "source": "adaptive_learning",
                        "confidence": pattern.confidence_score,
                        "complexity": pattern.optimal_complexity,
                        "coordination": pattern.optimal_coordination,
                        "agent_count": pattern.optimal_agent_count,
                        "agent_types": pattern.optimal_agent_types,
                        "success_rate": pattern.success_rate,
                        "usage_count": pattern.usage_count,
                        "learning_available": True
                    }
                else:
                    # Look for similar patterns
                    similar_pattern = self._find_similar_pattern(characteristics)
                    if similar_pattern:
                        return {
                            "source": "similar_pattern",
                            "confidence": similar_pattern.confidence_score * 0.8,  # Reduced confidence
                            "complexity": similar_pattern.optimal_complexity,
                            "coordination": similar_pattern.optimal_coordination,
                            "agent_count": similar_pattern.optimal_agent_count,
                            "agent_types": similar_pattern.optimal_agent_types,
                            "success_rate": similar_pattern.success_rate,
                            "learning_available": True
                        }
                    else:
                        # No learning data available
                        return {
                            "source": "no_learning",
                            "confidence": 0.5,
                            "learning_available": False
                        }
                        
        except sqlite3.Error:
            return {
                "source": "error",
                "confidence": 0.0,
                "learning_available": False
            }
    
    def _find_similar_pattern(self, characteristics: Dict[str, Any]) -> Optional[ZenLearningPattern]:
        """Find similar pattern based on characteristics overlap."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM zen_patterns 
                    WHERE confidence_score > 0.6 
                    ORDER BY success_rate DESC, confidence_score DESC 
                    LIMIT 20
                """)
                
                best_match = None
                best_similarity = 0.0
                
                for row in cursor.fetchall():
                    pattern_chars = json.loads(row[1])
                    similarity = self._calculate_similarity(characteristics, pattern_chars)
                    
                    if similarity > best_similarity and similarity > 0.6:
                        best_similarity = similarity
                        best_match = ZenLearningPattern(
                            pattern_id=row[0],
                            prompt_characteristics=pattern_chars,
                            optimal_complexity=row[2],
                            optimal_coordination=row[3],
                            optimal_agent_count=row[4],
                            optimal_agent_types=json.loads(row[5]),
                            success_rate=row[6],
                            confidence_score=row[7],
                            usage_count=row[8],
                            created_at=row[9],
                            updated_at=row[10]
                        )
                
                return best_match
                
        except sqlite3.Error:
            return None
    
    def _calculate_similarity(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Calculate similarity between characteristic sets."""
        similarity = 0.0
        weights = {
            "categories": 0.4,
            "complexity_indicators": 0.2,
            "action_verbs": 0.2,
            "urgency_level": 0.1,
            "has_multiple_tasks": 0.1
        }
        
        for key, weight in weights.items():
            if key in chars1 and key in chars2:
                if key in ["categories", "complexity_indicators", "action_verbs"]:
                    # List overlap
                    set1 = set(chars1[key])
                    set2 = set(chars2[key])
                    if set1 or set2:
                        overlap = len(set1 & set2) / len(set1 | set2)
                        similarity += weight * overlap
                elif key == "urgency_level":
                    # String match
                    if chars1[key] == chars2[key]:
                        similarity += weight
                elif key == "has_multiple_tasks":
                    # Boolean match
                    if chars1[key] == chars2[key]:
                        similarity += weight
        
        return similarity
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive learning metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Outcome metrics
                cursor = conn.execute("SELECT COUNT(*) FROM zen_outcomes")
                total_outcomes = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM zen_outcomes WHERE execution_success = 1
                """)
                successful_outcomes = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT AVG(user_satisfaction) FROM zen_outcomes WHERE execution_success = 1
                """)
                avg_satisfaction = cursor.fetchone()[0] or 0.0
                
                # Pattern metrics
                cursor = conn.execute("SELECT COUNT(*) FROM zen_patterns")
                total_patterns = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM zen_patterns WHERE confidence_score > 0.7
                """)
                high_confidence_patterns = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT AVG(success_rate) FROM zen_patterns
                """)
                avg_success_rate = cursor.fetchone()[0] or 0.0
                
                return {
                    "total_consultations": self.total_consultations,
                    "total_outcomes": total_outcomes,
                    "successful_outcomes": successful_outcomes,
                    "success_rate": successful_outcomes / max(total_outcomes, 1),
                    "avg_user_satisfaction": avg_satisfaction,
                    "successful_learnings": self.successful_learnings,
                    "pattern_improvements": self.pattern_improvements,
                    "total_patterns": total_patterns,
                    "high_confidence_patterns": high_confidence_patterns,
                    "avg_pattern_success_rate": avg_success_rate,
                    "learning_effectiveness": self.successful_learnings / max(self.total_consultations, 1),
                    "adaptive_intelligence_active": total_patterns > 0
                }
                
        except sqlite3.Error:
            return {
                "total_consultations": self.total_consultations,
                "successful_learnings": self.successful_learnings,
                "pattern_improvements": self.pattern_improvements,
                "learning_effectiveness": 0.0,
                "adaptive_intelligence_active": False
            }
    
    def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for model training."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Export outcomes
                cursor = conn.execute("SELECT * FROM zen_outcomes ORDER BY timestamp DESC LIMIT 100")
                outcomes = []
                for row in cursor.fetchall():
                    outcomes.append({
                        "consultation_id": row[0],
                        "prompt": row[1],
                        "complexity": row[2],
                        "coordination_type": row[3],
                        "agents_allocated": row[4],
                        "agent_types": json.loads(row[5]),
                        "mcp_tools": json.loads(row[6]),
                        "execution_success": bool(row[7]),
                        "user_satisfaction": row[8],
                        "actual_agents_needed": row[9],
                        "performance_metrics": json.loads(row[10]) if row[10] else {},
                        "lessons_learned": json.loads(row[11]) if row[11] else [],
                        "timestamp": row[12]
                    })
                
                # Export patterns
                cursor = conn.execute("SELECT * FROM zen_patterns ORDER BY confidence_score DESC")
                patterns = []
                for row in cursor.fetchall():
                    patterns.append({
                        "pattern_id": row[0],
                        "prompt_characteristics": json.loads(row[1]),
                        "optimal_complexity": row[2],
                        "optimal_coordination": row[3],
                        "optimal_agent_count": row[4],
                        "optimal_agent_types": json.loads(row[5]),
                        "success_rate": row[6],
                        "confidence_score": row[7],
                        "usage_count": row[8],
                        "created_at": row[9],
                        "updated_at": row[10]
                    })
                
                return {
                    "outcomes": outcomes,
                    "patterns": patterns,
                    "export_timestamp": time.time(),
                    "metrics": self.get_learning_metrics()
                }
                
        except sqlite3.Error as e:
            return {
                "error": f"Failed to export learning data: {e}",
                "outcomes": [],
                "patterns": []
            }


# Integration with ZEN Consultant
class AdaptiveZenConsultant(ZenConsultant):
    """Enhanced ZEN consultant with adaptive learning capabilities."""
    
    def __init__(self):
        super().__init__()
        self.learning_engine = ZenAdaptiveLearningEngine()
        self.use_adaptive_learning = True  # Enable by default
    
    def get_adaptive_directive(self, prompt: str) -> Dict[str, Any]:
        """Generate directive using adaptive learning if available."""
        if not self.use_adaptive_learning:
            return self.get_concise_directive(prompt)
        
        # Get adaptive recommendation
        adaptive_rec = self.learning_engine.get_adaptive_recommendation(prompt)
        
        if adaptive_rec.get("learning_available") and adaptive_rec.get("confidence", 0) > 0.6:
            # Use learned patterns
            return {
                "source": "adaptive_learning",
                "hive": adaptive_rec["coordination"],
                "swarm": f"{adaptive_rec['agent_count']} agents (learned)",
                "agents": adaptive_rec["agent_types"][:3],
                "tools": ["mcp__claude-flow__swarm_init", "mcp__zen__thinkdeep"],
                "confidence": adaptive_rec["confidence"],
                "success_rate": adaptive_rec.get("success_rate", 0.0),
                "session_id": self.session_id,
                "thinking_mode": adaptive_rec.get("complexity", "medium"),
                "learning_note": f"Based on {adaptive_rec.get('usage_count', 0)} similar successful consultations"
            }
        else:
            # Fall back to standard consultation
            standard_rec = self.get_concise_directive(prompt)
            standard_rec["source"] = "standard_consultation"
            standard_rec["learning_note"] = "No adaptive learning data available - building knowledge base"
            return standard_rec
    
    def record_consultation_outcome(self, consultation_id: str, prompt: str, 
                                  outcome_data: Dict[str, Any]) -> None:
        """Record consultation outcome for learning."""
        if not self.use_adaptive_learning:
            return
        
        # Convert outcome data to ZenLearningOutcome
        outcome = ZenLearningOutcome(
            consultation_id=consultation_id,
            prompt=prompt,
            complexity=outcome_data.get("complexity", "medium"),
            coordination_type=outcome_data.get("coordination_type", "SWARM"),
            agents_allocated=outcome_data.get("agents_allocated", 0),
            agent_types=outcome_data.get("agent_types", []),
            mcp_tools=outcome_data.get("mcp_tools", []),
            execution_success=outcome_data.get("execution_success", False),
            user_satisfaction=outcome_data.get("user_satisfaction", 0.5),
            actual_agents_needed=outcome_data.get("actual_agents_needed"),
            performance_metrics=outcome_data.get("performance_metrics", {}),
            lessons_learned=outcome_data.get("lessons_learned", []),
            timestamp=time.time()
        )
        
        # Record for learning
        self.learning_engine.record_consultation_outcome(outcome)