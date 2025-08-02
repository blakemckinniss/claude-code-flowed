#!/usr/bin/env python3
"""ZEN Memory Pipeline - Training data pipeline from memory system.

This module creates a pipeline from the existing memory system to provide
rich training data for ZEN adaptive learning and neural training models.

Key Features:
- Memory data extraction and processing
- Pattern recognition in memory entries
- Training data preparation from memory patterns
- Cross-session learning continuity
- Real-time memory integration
"""

import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import ZEN learning components
from .zen_adaptive_learning import ZenLearningOutcome, ZenAdaptiveLearningEngine
from .zen_neural_training import ZenNeuralTrainingPipeline


@dataclass
class MemoryPattern:
    """Represents a learned pattern from memory entries."""
    pattern_id: str
    memory_entries: List[Dict[str, Any]]
    extracted_knowledge: Dict[str, Any]
    confidence_score: float
    pattern_type: str  # consultation, operation, optimization
    timestamp: float


class ZenMemoryPipeline:
    """Pipeline for extracting training data from memory system."""
    
    def __init__(self, memory_store_path: str = ".claude/hooks/memory/memory-store.json"):
        self.memory_store_path = Path(memory_store_path)
        self.learning_engine = ZenAdaptiveLearningEngine()
        self.neural_pipeline = ZenNeuralTrainingPipeline()
        
        # Pattern extraction configuration
        self.min_confidence_threshold = 0.6
        self.max_memory_entries_per_pattern = 50
        
    def extract_training_data_from_memory(self) -> Dict[str, Any]:
        """Extract comprehensive training data from memory system."""
        if not self.memory_store_path.exists():
            return {"error": "Memory store not found", "training_data": []}
        
        try:
            with open(self.memory_store_path) as f:
                memory_data = json.load(f)
            
            # Process memory entries
            zen_entries = self._filter_zen_related_entries(memory_data)
            consultation_patterns = self._extract_consultation_patterns(zen_entries)
            operation_patterns = self._extract_operation_patterns(zen_entries)
            optimization_patterns = self._extract_optimization_patterns(zen_entries)
            
            # Convert to training data
            training_outcomes = []
            
            for pattern in consultation_patterns + operation_patterns + optimization_patterns:
                outcome = self._convert_pattern_to_outcome(pattern)
                if outcome:
                    training_outcomes.append(outcome)
            
            return {
                "total_memory_entries": len(memory_data),
                "zen_related_entries": len(zen_entries),
                "extracted_patterns": {
                    "consultation": len(consultation_patterns),
                    "operation": len(operation_patterns),
                    "optimization": len(optimization_patterns)
                },
                "training_outcomes": len(training_outcomes),
                "training_data": training_outcomes
            }
            
        except Exception as e:
            return {"error": f"Memory extraction failed: {e}", "training_data": []}
    
    def _filter_zen_related_entries(self, memory_data: List[Dict]) -> List[Dict]:
        """Filter memory entries related to ZEN operations."""
        zen_entries = []
        
        zen_keywords = [
            "zen", "consultation", "agent", "coordination", "swarm", "hive",
            "mcp__zen", "neural", "learning", "optimization", "adaptive"
        ]
        
        for entry in memory_data:
            if not isinstance(entry, dict):
                continue
                
            # Check if entry is ZEN-related
            entry_text = json.dumps(entry).lower()
            
            if any(keyword in entry_text for keyword in zen_keywords):
                zen_entries.append(entry)
        
        return zen_entries
    
    def _extract_consultation_patterns(self, zen_entries: List[Dict]) -> List[MemoryPattern]:
        """Extract consultation patterns from memory entries."""
        patterns = []
        
        # Group entries by consultation sessions
        consultation_groups = self._group_entries_by_session(zen_entries, "consultation")
        
        for session_id, entries in consultation_groups.items():
            if len(entries) < 2:  # Need at least 2 entries for a pattern
                continue
            
            # Extract consultation knowledge
            knowledge = self._extract_consultation_knowledge(entries)
            
            if knowledge and knowledge.get("confidence_score", 0) > self.min_confidence_threshold:
                pattern = MemoryPattern(
                    pattern_id=f"consultation_{session_id}",
                    memory_entries=entries,
                    extracted_knowledge=knowledge,
                    confidence_score=knowledge["confidence_score"],
                    pattern_type="consultation",
                    timestamp=time.time()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_operation_patterns(self, zen_entries: List[Dict]) -> List[MemoryPattern]:
        """Extract operation patterns from memory entries."""
        patterns = []
        
        # Look for successful operation sequences
        operation_sequences = self._find_operation_sequences(zen_entries)
        
        for seq_id, sequence in enumerate(operation_sequences):
            if len(sequence) < 3:  # Need meaningful sequence
                continue
            
            knowledge = self._extract_operation_knowledge(sequence)
            
            if knowledge and knowledge.get("success_indicator", False):
                pattern = MemoryPattern(
                    pattern_id=f"operation_{seq_id}",
                    memory_entries=sequence,
                    extracted_knowledge=knowledge,
                    confidence_score=knowledge.get("confidence_score", 0.5),
                    pattern_type="operation",
                    timestamp=time.time()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_optimization_patterns(self, zen_entries: List[Dict]) -> List[MemoryPattern]:
        """Extract optimization patterns from memory entries."""
        patterns = []
        
        # Look for optimization-related entries
        optimization_entries = []
        for entry in zen_entries:
            entry_text = json.dumps(entry).lower()
            if any(word in entry_text for word in ["optimize", "improve", "enhance", "better", "efficient"]):
                optimization_entries.append(entry)
        
        if len(optimization_entries) < 5:  # Need sufficient data
            return patterns
        
        # Group optimizations by type
        optimization_groups = self._group_optimizations_by_type(optimization_entries)
        
        for opt_type, entries in optimization_groups.items():
            if len(entries) >= 3:
                knowledge = self._extract_optimization_knowledge(entries, opt_type)
                
                if knowledge:
                    pattern = MemoryPattern(
                        pattern_id=f"optimization_{opt_type}",
                        memory_entries=entries,
                        extracted_knowledge=knowledge,
                        confidence_score=knowledge.get("confidence_score", 0.4),
                        pattern_type="optimization",
                        timestamp=time.time()
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _group_entries_by_session(self, entries: List[Dict], pattern_type: str) -> Dict[str, List[Dict]]:
        """Group memory entries by session or related context."""
        groups = {}
        
        for entry in entries:
            # Extract session identifier
            session_id = self._extract_session_id(entry)
            
            if session_id not in groups:
                groups[session_id] = []
            
            groups[session_id].append(entry)
        
        return groups
    
    def _extract_session_id(self, entry: Dict) -> str:
        """Extract session identifier from memory entry."""
        # Try various session ID patterns
        entry_str = json.dumps(entry)
        
        # Look for session patterns
        session_patterns = [
            r'session_(\d{8}_\d{6})',
            r'session[_\-]([a-zA-Z0-9]+)',
            r'consultation[_\-]([a-zA-Z0-9]+)',
            r'zen[_\-]([a-zA-Z0-9]+)'
        ]
        
        for pattern in session_patterns:
            match = re.search(pattern, entry_str)
            if match:
                return match.group(1)
        
        # Fallback to timestamp-based grouping
        timestamp = entry.get("timestamp", time.time())
        return f"group_{int(timestamp // 3600)}"  # Group by hour
    
    def _extract_consultation_knowledge(self, entries: List[Dict]) -> Optional[Dict[str, Any]]:
        """Extract knowledge from consultation entries."""
        try:
            # Analyze consultation patterns
            prompts = []
            complexities = []
            coordinations = []
            agent_counts = []
            satisfactions = []
            
            for entry in entries:
                entry_str = json.dumps(entry).lower()
                
                # Extract prompts
                if "prompt" in entry_str:
                    prompt_match = re.search(r'"prompt":\s*"([^"]+)"', json.dumps(entry))
                    if prompt_match:
                        prompts.append(prompt_match.group(1))
                
                # Extract complexity indicators
                if any(word in entry_str for word in ["complex", "simple", "medium", "enterprise"]):
                    for complexity in ["simple", "medium", "complex", "enterprise"]:
                        if complexity in entry_str:
                            complexities.append(complexity)
                            break
                
                # Extract coordination patterns
                if "hive" in entry_str:
                    coordinations.append("HIVE")
                elif "swarm" in entry_str:
                    coordinations.append("SWARM")
                
                # Extract agent information
                agent_match = re.search(r'(\d+)\s*agent', entry_str)
                if agent_match:
                    agent_counts.append(int(agent_match.group(1)))
                
                # Extract satisfaction indicators
                if any(word in entry_str for word in ["success", "complete", "good", "excellent"]):
                    satisfactions.append(0.8)
                elif any(word in entry_str for word in ["partial", "ok", "acceptable"]):
                    satisfactions.append(0.6)
                elif any(word in entry_str for word in ["failed", "error", "problem"]):
                    satisfactions.append(0.2)
            
            if not prompts:
                return None
            
            # Synthesize knowledge
            return {
                "prompt_sample": prompts[0] if prompts else "",
                "dominant_complexity": max(set(complexities), key=complexities.count) if complexities else "medium",
                "dominant_coordination": max(set(coordinations), key=coordinations.count) if coordinations else "SWARM",
                "avg_agent_count": sum(agent_counts) / len(agent_counts) if agent_counts else 2,
                "avg_satisfaction": sum(satisfactions) / len(satisfactions) if satisfactions else 0.5,
                "consultation_count": len(entries),
                "confidence_score": min(0.9, len(entries) / 10.0)  # Confidence based on data volume
            }
            
        except Exception:
            return None
    
    def _find_operation_sequences(self, entries: List[Dict]) -> List[List[Dict]]:
        """Find sequences of related operations."""
        sequences = []
        
        # Sort entries by timestamp
        sorted_entries = sorted(entries, key=lambda x: x.get("timestamp", 0))
        
        current_sequence = []
        last_timestamp = 0
        
        for entry in sorted_entries:
            timestamp = entry.get("timestamp", 0)
            
            # If gap is too large, start new sequence
            if timestamp - last_timestamp > 3600:  # 1 hour gap
                if len(current_sequence) >= 3:
                    sequences.append(current_sequence)
                current_sequence = [entry]
            else:
                current_sequence.append(entry)
            
            last_timestamp = timestamp
        
        # Add final sequence
        if len(current_sequence) >= 3:
            sequences.append(current_sequence)
        
        return sequences
    
    def _extract_operation_knowledge(self, sequence: List[Dict]) -> Optional[Dict[str, Any]]:
        """Extract knowledge from operation sequence."""
        try:
            # Look for success indicators
            success_indicators = []
            tools_used = []
            
            for entry in sequence:
                entry_str = json.dumps(entry).lower()
                
                # Check for success
                if any(word in entry_str for word in ["success", "complete", "done"]):
                    success_indicators.append(True)
                elif any(word in entry_str for word in ["error", "failed", "problem"]):
                    success_indicators.append(False)
                
                # Extract tools
                tool_match = re.search(r'"tool":\s*"([^"]+)"', json.dumps(entry))
                if tool_match:
                    tools_used.append(tool_match.group(1))
            
            success_rate = sum(success_indicators) / len(success_indicators) if success_indicators else 0.5
            
            return {
                "sequence_length": len(sequence),
                "success_indicator": success_rate > 0.6,
                "success_rate": success_rate,
                "tools_used": list(set(tools_used)),
                "confidence_score": min(0.8, success_rate)
            }
            
        except Exception:
            return None
    
    def _group_optimizations_by_type(self, entries: List[Dict]) -> Dict[str, List[Dict]]:
        """Group optimization entries by type."""
        groups = {
            "performance": [],
            "coordination": [],
            "agent_selection": [],
            "neural": [],
            "general": []
        }
        
        for entry in entries:
            entry_str = json.dumps(entry).lower()
            
            if any(word in entry_str for word in ["performance", "speed", "efficient"]):
                groups["performance"].append(entry)
            elif any(word in entry_str for word in ["coordination", "hive", "swarm"]):
                groups["coordination"].append(entry)
            elif any(word in entry_str for word in ["agent", "specialist"]):
                groups["agent_selection"].append(entry)
            elif any(word in entry_str for word in ["neural", "learning", "pattern"]):
                groups["neural"].append(entry)
            else:
                groups["general"].append(entry)
        
        return groups
    
    def _extract_optimization_knowledge(self, entries: List[Dict], opt_type: str) -> Optional[Dict[str, Any]]:
        """Extract knowledge from optimization entries."""
        try:
            # Extract optimization patterns
            optimizations = []
            
            for entry in entries:
                entry_str = json.dumps(entry)
                
                # Look for optimization descriptions
                opt_patterns = [
                    r'optimization[^"]*"([^"]+)"',
                    r'improve[^"]*"([^"]+)"',
                    r'enhance[^"]*"([^"]+)"',
                    r'better[^"]*"([^"]+)"'
                ]
                
                for pattern in opt_patterns:
                    matches = re.findall(pattern, entry_str, re.IGNORECASE)
                    optimizations.extend(matches)
            
            if not optimizations:
                return None
            
            return {
                "optimization_type": opt_type,
                "optimization_count": len(optimizations),
                "sample_optimizations": optimizations[:3],
                "confidence_score": min(0.7, len(entries) / 10.0)
            }
            
        except Exception:
            return None
    
    def _convert_pattern_to_outcome(self, pattern: MemoryPattern) -> Optional[ZenLearningOutcome]:
        """Convert memory pattern to ZEN learning outcome."""
        try:
            knowledge = pattern.extracted_knowledge
            
            if pattern.pattern_type == "consultation":
                return ZenLearningOutcome(
                    consultation_id=pattern.pattern_id,
                    prompt=knowledge.get("prompt_sample", "Memory-derived consultation"),
                    complexity=knowledge.get("dominant_complexity", "medium"),
                    coordination_type=knowledge.get("dominant_coordination", "SWARM"),
                    agents_allocated=int(knowledge.get("avg_agent_count", 2)),
                    agent_types=["coder", "reviewer"],  # Default
                    mcp_tools=[],
                    execution_success=knowledge.get("avg_satisfaction", 0.5) > 0.5,
                    user_satisfaction=knowledge.get("avg_satisfaction", 0.5),
                    actual_agents_needed=int(knowledge.get("avg_agent_count", 2)),
                    performance_metrics={
                        "memory_pattern_confidence": pattern.confidence_score,
                        "consultation_count": knowledge.get("consultation_count", 1)
                    },
                    lessons_learned=[f"Pattern learned from {knowledge.get('consultation_count', 1)} memory entries"],
                    timestamp=pattern.timestamp
                )
            
            # Add more pattern type conversions as needed
            return None
            
        except Exception as e:
            print(f"Error converting pattern to outcome: {e}")
            return None
    
    def train_from_memory(self) -> Dict[str, Any]:
        """Train ZEN models using memory data."""
        print("🧠 Extracting training data from memory system...")
        
        # Extract training data
        memory_analysis = self.extract_training_data_from_memory()
        
        if memory_analysis.get("error"):
            return memory_analysis
        
        training_outcomes = memory_analysis["training_data"]
        
        if not training_outcomes:
            return {
                "status": "no_training_data",
                "message": "No suitable training data found in memory"
            }
        
        # Record outcomes in learning engine
        recorded_count = 0
        for outcome in training_outcomes:
            success = self.learning_engine.record_consultation_outcome(outcome)
            if success:
                recorded_count += 1
        
        # Train neural models
        training_results = self.neural_pipeline.train_all_models()
        
        return {
            "status": "success",
            "memory_analysis": memory_analysis,
            "outcomes_recorded": recorded_count,
            "neural_training_results": training_results,
            "learning_metrics": self.learning_engine.get_learning_metrics()
        }
    
    def get_memory_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of intelligence extracted from memory."""
        memory_analysis = self.extract_training_data_from_memory()
        
        if memory_analysis.get("error"):
            return {"error": memory_analysis["error"]}
        
        return {
            "memory_intelligence": {
                "total_entries_analyzed": memory_analysis["total_memory_entries"],
                "zen_related_entries": memory_analysis["zen_related_entries"],
                "patterns_extracted": memory_analysis["extracted_patterns"],
                "training_outcomes_generated": memory_analysis["training_outcomes"]
            },
            "intelligence_quality": {
                "pattern_density": memory_analysis["zen_related_entries"] / max(memory_analysis["total_memory_entries"], 1),
                "conversion_rate": memory_analysis["training_outcomes"] / max(memory_analysis["zen_related_entries"], 1),
                "data_richness_score": min(1.0, memory_analysis["training_outcomes"] / 20.0)  # 20+ outcomes = rich
            },
            "recommendations": self._generate_memory_recommendations(memory_analysis)
        }
    
    def _generate_memory_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on memory analysis."""
        recommendations = []
        
        if analysis["training_outcomes"] < 10:
            recommendations.append("Increase ZEN consultation usage to build richer training data")
        
        if analysis["zen_related_entries"] < analysis["total_memory_entries"] * 0.1:
            recommendations.append("More ZEN-related operations needed for better pattern learning")
        
        pattern_counts = analysis["extracted_patterns"]
        if pattern_counts["consultation"] < 5:
            recommendations.append("More consultation patterns needed for consultation intelligence")
        
        if pattern_counts["optimization"] < 3:
            recommendations.append("Document more optimization outcomes for learning improvement")
        
        if not recommendations:
            recommendations.append("Memory intelligence is well-developed - continue current usage patterns")
        
        return recommendations


if __name__ == "__main__":
    # Test the memory pipeline
    pipeline = ZenMemoryPipeline()
    
    print("Testing ZEN Memory Pipeline...")
    summary = pipeline.get_memory_intelligence_summary()
    print(f"Memory Intelligence Summary: {json.dumps(summary, indent=2)}")
    
    print("\nTraining from memory...")
    results = pipeline.train_from_memory()
    print(f"Training Results: {json.dumps(results, indent=2)}")