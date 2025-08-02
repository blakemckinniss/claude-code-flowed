#!/usr/bin/env python3
"""ZEN Neural Training Pipeline - Enhanced neural training with ZEN integration.

This module extends the existing neural training infrastructure to provide
specialized models for ZEN consultation patterns, real-time learning updates,
and comprehensive model management for adaptive intelligence.

Key Features:
- ZEN-specific neural models (task-predictor, agent-selector, performance-optimizer)
- Real-time model updates from consultation outcomes
- Memory data pipeline integration  
- Continuous learning feedback loops
- Model validation and performance tracking
"""

import json
import sqlite3
import time
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

# Import existing neural infrastructure
from ..pre_tool.analyzers.neural_pattern_validator import NeuralPatternStorage
from .zen_adaptive_learning import ZenAdaptiveLearningEngine, ZenLearningOutcome


@dataclass
class ZenModelMetrics:
    """Metrics for ZEN neural models."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    last_updated: float
    performance_trend: List[float]


@dataclass
class TrainingConfig:
    """Configuration for neural training."""
    min_samples_for_training: int = 20
    min_accuracy_threshold: float = 0.7
    retrain_interval_hours: int = 24
    max_model_age_days: int = 7
    validation_split: float = 0.2
    enable_online_learning: bool = True


class ZenTaskPredictor:
    """Neural model for predicting task complexity and coordination needs."""
    
    def __init__(self):
        self.complexity_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.coordination_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.agent_count_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.label_encoders = {}
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.metrics: Optional[ZenModelMetrics] = None
    
    def extract_features(self, prompt: str, context: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from prompt and context."""
        # Text-based features
        words = prompt.lower().split()
        word_count = len(words)
        char_count = len(prompt)
        
        # Complexity indicators
        simple_keywords = ["fix", "update", "add", "remove", "change", "help"]
        complex_keywords = ["architecture", "system", "enterprise", "migrate", "scalable", "performance"]
        
        simple_score = sum(1 for keyword in simple_keywords if keyword in prompt.lower())
        complex_score = sum(1 for keyword in complex_keywords if keyword in prompt.lower())
        
        # Task category features
        categories = {
            "development": ["code", "implement", "build", "create"],
            "testing": ["test", "qa", "quality"],
            "debugging": ["debug", "fix", "error", "issue"],
            "architecture": ["architecture", "design", "system"],
            "security": ["security", "audit", "vulnerability"],
            "performance": ["performance", "optimize", "speed"]
        }
        
        category_scores = []
        for _category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in prompt.lower())
            category_scores.append(score)
        
        # Context features
        has_urgency = any(word in prompt.lower() for word in ["urgent", "asap", "immediately"])
        has_multiple_tasks = "and" in prompt.lower() or "," in prompt
        mentions_agents = "agent" in prompt.lower()
        
        # Historical context (if available)
        recent_complexity = context.get("recent_complexity", 0.5)
        recent_success_rate = context.get("recent_success_rate", 0.5)
        user_experience_level = context.get("user_experience_level", 0.5)
        
        # Combine all features
        features = [word_count, char_count, simple_score, complex_score, int(has_urgency), int(has_multiple_tasks), int(mentions_agents), recent_complexity, recent_success_rate, user_experience_level, *category_scores]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Tuple[str, Dict[str, Any], str, str, int]]) -> bool:
        """Train the task prediction models."""
        if len(training_data) < 20:  # Minimum samples
            return False
        
        try:
            # Prepare training data
            X_features = []
            y_complexity = []
            y_coordination = []
            y_agent_count = []
            
            for prompt, context, complexity, coordination, agent_count in training_data:
                features = self.extract_features(prompt, context).flatten()
                X_features.append(features)
                y_complexity.append(complexity)
                y_coordination.append(coordination)
                y_agent_count.append(agent_count)
            
            X = np.array(X_features)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train complexity model
            self.complexity_model.fit(X_scaled, y_complexity)
            
            # Train coordination model  
            self.coordination_model.fit(X_scaled, y_coordination)
            
            # Train agent count model
            self.agent_count_model.fit(X_scaled, y_agent_count)
            
            self.is_trained = True
            
            # Calculate metrics
            self._calculate_metrics(X_scaled, y_complexity, y_coordination, y_agent_count)
            
            return True
            
        except Exception as e:
            print(f"Error training task predictor: {e}")
            return False
    
    def predict(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict task characteristics."""
        if not self.is_trained:
            return {
                "complexity": "medium",
                "coordination": "SWARM",
                "agent_count": 2,
                "confidence": 0.5
            }
        
        try:
            features = self.extract_features(prompt, context)
            features_scaled = self.feature_scaler.transform(features)
            
            # Get predictions
            complexity_pred = self.complexity_model.predict(features_scaled)[0]
            coordination_pred = self.coordination_model.predict(features_scaled)[0]
            agent_count_pred = max(0, round(self.agent_count_model.predict(features_scaled)[0]))
            
            # Get confidence scores
            complexity_proba = max(self.complexity_model.predict_proba(features_scaled)[0])
            coordination_proba = max(self.coordination_model.predict_proba(features_scaled)[0])
            
            confidence = (complexity_proba + coordination_proba) / 2
            
            return {
                "complexity": complexity_pred,
                "coordination": coordination_pred,
                "agent_count": agent_count_pred,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Error in task prediction: {e}")
            return {
                "complexity": "medium",
                "coordination": "SWARM", 
                "agent_count": 2,
                "confidence": 0.5
            }
    
    def _calculate_metrics(self, X: np.ndarray, y_complexity: List[str], 
                          y_coordination: List[str], y_agent_count: List[int]) -> None:
        """Calculate model performance metrics."""
        try:
            # Split data for validation
            X_train, X_val, y_comp_train, y_comp_val = train_test_split(
                X, y_complexity, test_size=0.2, random_state=42
            )
            _, _, y_coord_train, y_coord_val = train_test_split(
                X, y_coordination, test_size=0.2, random_state=42
            )
            _, _, y_count_train, y_count_val = train_test_split(
                X, y_agent_count, test_size=0.2, random_state=42
            )
            
            # Predictions
            comp_pred = self.complexity_model.predict(X_val)
            coord_pred = self.coordination_model.predict(X_val)
            count_pred = self.agent_count_model.predict(X_val)
            
            # Calculate accuracies
            comp_accuracy = accuracy_score(y_comp_val, comp_pred)
            coord_accuracy = accuracy_score(y_coord_val, coord_pred)
            mean_squared_error(y_count_val, count_pred)
            
            # Overall metrics
            overall_accuracy = (comp_accuracy + coord_accuracy) / 2
            
            self.metrics = ZenModelMetrics(
                model_name="task_predictor",
                accuracy=overall_accuracy,
                precision=overall_accuracy,  # Simplified
                recall=overall_accuracy,
                f1_score=overall_accuracy,
                training_samples=len(X),
                last_updated=time.time(),
                performance_trend=[overall_accuracy]
            )
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")


class ZenAgentSelector:
    """Neural model for selecting optimal agent combinations."""
    
    def __init__(self):
        self.agent_type_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.specialist_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.metrics: Optional[ZenModelMetrics] = None
        
        # Agent type mapping
        self.agent_types = [
            "coder", "reviewer", "tester", "debugger", "architect",
            "security-auditor", "performance-optimizer", "documentation-specialist",
            "deployment-engineer", "data-engineer", "frontend-developer", "backend-developer"
        ]
    
    def extract_agent_features(self, prompt: str, context: Dict[str, Any], 
                              task_prediction: Dict[str, Any]) -> np.ndarray:
        """Extract features for agent selection."""
        # Task characteristics
        complexity_score = {"simple": 1, "medium": 2, "complex": 3, "enterprise": 4}.get(
            task_prediction.get("complexity", "medium"), 2
        )
        
        coordination_score = 1 if task_prediction.get("coordination") == "SWARM" else 2
        predicted_agent_count = task_prediction.get("agent_count", 2)
        
        # Domain-specific features
        domains = {
            "coding": ["code", "implement", "build", "create", "develop"],
            "testing": ["test", "qa", "quality", "testing"],
            "debugging": ["debug", "fix", "error", "issue", "bug"],
            "architecture": ["architecture", "design", "system", "structure"],
            "security": ["security", "audit", "vulnerability", "secure"],
            "performance": ["performance", "optimize", "speed", "efficient"],
            "documentation": ["document", "docs", "readme", "guide"],
            "deployment": ["deploy", "release", "production", "ci/cd"]
        }
        
        domain_scores = []
        for _domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in prompt.lower())
            domain_scores.append(score)
        
        # Historical success patterns
        historical_success = context.get("historical_agent_success", {})
        avg_success_rates = []
        for agent_type in self.agent_types:
            success_rate = historical_success.get(agent_type, 0.5)
            avg_success_rates.append(success_rate)
        
        # Combine features
        features = [
            complexity_score,
            coordination_score,
            predicted_agent_count,
            len(prompt.split()),  # Word count
        ] + domain_scores + avg_success_rates
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Tuple[str, Dict[str, Any], List[str], float]]) -> bool:
        """Train agent selection models."""
        if len(training_data) < 15:
            return False
        
        try:
            X_features = []
            y_primary_agent = []
            y_success_score = []
            
            for prompt, context, agent_types, success_score in training_data:
                # Mock task prediction for feature extraction
                task_pred = {"complexity": "medium", "coordination": "SWARM", "agent_count": len(agent_types)}
                features = self.extract_agent_features(prompt, context, task_pred).flatten()
                
                X_features.append(features)
                primary_agent = agent_types[0] if agent_types else "coder"
                y_primary_agent.append(primary_agent)
                y_success_score.append(success_score)
            
            X = np.array(X_features)
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train primary agent selector
            self.agent_type_model.fit(X_scaled, y_primary_agent)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training agent selector: {e}")
            return False
    
    def select_agents(self, prompt: str, context: Dict[str, Any], 
                     task_prediction: Dict[str, Any]) -> List[str]:
        """Select optimal agents for the task."""
        if not self.is_trained:
            # Fallback to rule-based selection
            return self._rule_based_selection(prompt, task_prediction.get("agent_count", 2))
        
        try:
            features = self.extract_agent_features(prompt, context, task_prediction)
            features_scaled = self.feature_scaler.transform(features)
            
            # Predict primary agent
            primary_agent = self.agent_type_model.predict(features_scaled)[0]
            
            # Get agent count from task prediction
            agent_count = task_prediction.get("agent_count", 2)
            
            # Select additional agents based on domain analysis
            selected_agents = [primary_agent]
            remaining_agents = self._select_complementary_agents(
                prompt, primary_agent, agent_count - 1
            )
            
            selected_agents.extend(remaining_agents)
            return selected_agents[:agent_count]
            
        except Exception as e:
            print(f"Error in agent selection: {e}")
            return self._rule_based_selection(prompt, task_prediction.get("agent_count", 2))
    
    def _rule_based_selection(self, prompt: str, agent_count: int) -> List[str]:
        """Fallback rule-based agent selection."""
        prompt_lower = prompt.lower()
        selected = []
        
        # Priority-based selection
        if any(word in prompt_lower for word in ["debug", "fix", "error", "issue"]):
            selected.append("debugger")
        if any(word in prompt_lower for word in ["test", "qa", "quality"]):
            selected.append("tester")
        if any(word in prompt_lower for word in ["security", "audit", "vulnerability"]):
            selected.append("security-auditor")
        if any(word in prompt_lower for word in ["performance", "optimize", "speed"]):
            selected.append("performance-optimizer")
        if any(word in prompt_lower for word in ["architecture", "design", "system"]):
            selected.append("architect")
        
        # Fill remaining slots
        while len(selected) < agent_count:
            if len(selected) == 0:
                selected.append("coder")
            elif "reviewer" not in selected:
                selected.append("reviewer")
            else:
                selected.append("coder")
        
        return selected[:agent_count]
    
    def _select_complementary_agents(self, prompt: str, primary_agent: str, 
                                   remaining_count: int) -> List[str]:
        """Select complementary agents based on primary agent."""
        prompt_lower = prompt.lower()
        complementary = []
        
        # Complementary agent rules
        if primary_agent == "coder":
            complementary.extend(["reviewer", "tester"])
        elif primary_agent == "debugger":
            complementary.extend(["coder", "tester"])
        elif primary_agent == "architect":
            complementary.extend(["coder", "security-auditor"])
        elif primary_agent == "security-auditor":
            complementary.extend(["coder", "reviewer"])
        elif primary_agent == "performance-optimizer":
            complementary.extend(["coder", "tester"])
        else:
            complementary.extend(["coder", "reviewer", "tester"])
        
        # Domain-specific additions
        if any(word in prompt_lower for word in ["frontend", "ui", "interface"]):
            if "frontend-developer" not in complementary:
                complementary.insert(0, "frontend-developer")
        
        if any(word in prompt_lower for word in ["backend", "api", "server"]):
            if "backend-developer" not in complementary:
                complementary.insert(0, "backend-developer")
        
        # Remove duplicates and primary agent
        complementary = [agent for agent in complementary if agent != primary_agent]
        return complementary[:remaining_count]


class ZenNeuralTrainingPipeline:
    """Main neural training pipeline for ZEN adaptive learning."""
    
    def __init__(self, db_path: str = ".claude/hooks/db/zen_neural_training.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.task_predictor = ZenTaskPredictor()
        self.agent_selector = ZenAgentSelector()
        
        # Data sources
        self.learning_engine = ZenAdaptiveLearningEngine()
        self.neural_storage = NeuralPatternStorage()
        
        # Configuration
        self.config = TrainingConfig()
        
        # Initialize database
        self._init_training_database()
        
        # Load existing models if available
        self._load_models()
    
    def _init_training_database(self) -> None:
        """Initialize neural training database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_versions (
                        model_name TEXT PRIMARY KEY,
                        version INTEGER NOT NULL,
                        accuracy REAL NOT NULL,
                        training_samples INTEGER NOT NULL,
                        created_at REAL NOT NULL,
                        model_data BLOB NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        training_samples INTEGER NOT NULL,
                        accuracy_before REAL,
                        accuracy_after REAL,
                        training_duration REAL NOT NULL,
                        timestamp REAL NOT NULL
                    )
                """)
                
                conn.commit()
                
        except sqlite3.Error as e:
            print(f"Warning: Neural training database initialization failed: {e}")
    
    def train_all_models(self, force_retrain: bool = False) -> Dict[str, bool]:
        """Train all ZEN neural models."""
        results = {}
        
        # Get training data
        training_data = self._prepare_training_data()
        
        if not training_data or len(training_data) < self.config.min_samples_for_training:
            print(f"Insufficient training data: {len(training_data) if training_data else 0} samples")
            return {"task_predictor": False, "agent_selector": False}
        
        # Train task predictor
        start_time = time.time()
        task_data = self._prepare_task_prediction_data(training_data)
        
        if len(task_data) >= self.config.min_samples_for_training:
            success = self.task_predictor.train(task_data)
            results["task_predictor"] = success
            
            if success:
                self._save_model("task_predictor", self.task_predictor)
                self._log_training("task_predictor", len(task_data), 
                                 self.task_predictor.metrics.accuracy if self.task_predictor.metrics else 0.0,
                                 time.time() - start_time)
        else:
            results["task_predictor"] = False
        
        # Train agent selector
        start_time = time.time()
        agent_data = self._prepare_agent_selection_data(training_data)
        
        if len(agent_data) >= self.config.min_samples_for_training // 2:  # Lower threshold
            success = self.agent_selector.train(agent_data)
            results["agent_selector"] = success
            
            if success:
                self._save_model("agent_selector", self.agent_selector)
                self._log_training("agent_selector", len(agent_data), 0.8,  # Estimated
                                 time.time() - start_time)
        else:
            results["agent_selector"] = False
        
        return results
    
    def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data from all sources."""
        training_data = []
        
        # Get data from ZEN learning engine
        zen_data = self.learning_engine.export_learning_data()
        
        for outcome in zen_data.get("outcomes", []):
            if outcome.get("execution_success") and outcome.get("user_satisfaction", 0) > 0.5:
                training_data.append({
                    "prompt": outcome["prompt"],
                    "complexity": outcome["complexity"],
                    "coordination_type": outcome["coordination_type"],
                    "agents_allocated": outcome["agents_allocated"],
                    "agent_types": outcome["agent_types"],
                    "actual_agents_needed": outcome.get("actual_agents_needed", outcome["agents_allocated"]),
                    "user_satisfaction": outcome["user_satisfaction"],
                    "performance_metrics": outcome.get("performance_metrics", {}),
                    "timestamp": outcome["timestamp"]
                })
        
        # Get data from neural patterns
        recent_patterns = self.neural_storage.get_recent_patterns(50)
        for pattern in recent_patterns:
            if pattern.success_count > 0 and pattern.confidence_score > 0.6:
                # Convert neural pattern to training data format
                training_data.append({
                    "prompt": f"Neural pattern: {pattern.learned_optimization}",
                    "complexity": pattern.performance_metrics.get("complexity", "medium"),
                    "coordination_type": pattern.performance_metrics.get("coordination", "SWARM"),
                    "agents_allocated": pattern.performance_metrics.get("agents_allocated", 2),
                    "agent_types": ["coder", "reviewer"],  # Default
                    "actual_agents_needed": pattern.performance_metrics.get("actual_agents_needed", 2),
                    "user_satisfaction": pattern.confidence_score,
                    "performance_metrics": pattern.performance_metrics,
                    "timestamp": pattern.last_used_timestamp
                })
        
        return training_data
    
    def _prepare_task_prediction_data(self, training_data: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any], str, str, int]]:
        """Prepare data for task prediction model."""
        task_data = []
        
        for data in training_data:
            context = {
                "recent_complexity": 0.5,  # Could be enhanced with actual history
                "recent_success_rate": data.get("user_satisfaction", 0.5),
                "user_experience_level": 0.5  # Could be enhanced with user profiling
            }
            
            task_data.append((
                data["prompt"],
                context,
                data["complexity"],
                data["coordination_type"],
                data["actual_agents_needed"]
            ))
        
        return task_data
    
    def _prepare_agent_selection_data(self, training_data: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any], List[str], float]]:
        """Prepare data for agent selection model."""
        agent_data = []
        
        for data in training_data:
            context = {
                "historical_agent_success": {}  # Could be enhanced with actual history
            }
            
            agent_data.append((
                data["prompt"],
                context,
                data["agent_types"],
                data["user_satisfaction"]
            ))
        
        return agent_data
    
    def _save_model(self, model_name: str, model_obj: Any) -> bool:
        """Save trained model to database."""
        try:
            model_data = pickle.dumps(model_obj)
            accuracy = model_obj.metrics.accuracy if hasattr(model_obj, 'metrics') and model_obj.metrics else 0.0
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_versions
                    (model_name, version, accuracy, training_samples, created_at, model_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    int(time.time()),  # Use timestamp as version
                    accuracy,
                    getattr(model_obj.metrics, 'training_samples', 0) if hasattr(model_obj, 'metrics') and model_obj.metrics else 0,
                    time.time(),
                    model_data
                ))
                conn.commit()
                
            return True
            
        except Exception as e:
            print(f"Error saving model {model_name}: {e}")
            return False
    
    def _load_models(self) -> None:
        """Load existing models from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load task predictor
                cursor = conn.execute("""
                    SELECT model_data FROM model_versions 
                    WHERE model_name = 'task_predictor' 
                    ORDER BY version DESC LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    self.task_predictor = pickle.loads(row[0])
                
                # Load agent selector
                cursor = conn.execute("""
                    SELECT model_data FROM model_versions 
                    WHERE model_name = 'agent_selector' 
                    ORDER BY version DESC LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    self.agent_selector = pickle.loads(row[0])
                    
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _log_training(self, model_name: str, training_samples: int, 
                     accuracy_after: float, duration: float) -> None:
        """Log training session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO training_logs
                    (model_name, training_samples, accuracy_after, training_duration, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (model_name, training_samples, accuracy_after, duration, time.time()))
                conn.commit()
                
        except sqlite3.Error as e:
            print(f"Error logging training: {e}")
    
    def get_enhanced_prediction(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced prediction using all neural models."""
        # Task prediction
        task_pred = self.task_predictor.predict(prompt, context)
        
        # Agent selection
        selected_agents = self.agent_selector.select_agents(prompt, context, task_pred)
        
        # Combine predictions
        return {
            "complexity": task_pred["complexity"],
            "coordination": task_pred["coordination"],
            "agent_count": task_pred["agent_count"],
            "agent_types": selected_agents,
            "confidence": task_pred["confidence"],
            "source": "neural_prediction",
            "models_used": {
                "task_predictor_trained": self.task_predictor.is_trained,
                "agent_selector_trained": self.agent_selector.is_trained
            }
        }
    
    def update_models_from_outcome(self, outcome: ZenLearningOutcome) -> None:
        """Update models with new outcome data (online learning)."""
        if not self.config.enable_online_learning:
            return
        
        # Record outcome in learning engine
        self.learning_engine.record_consultation_outcome(outcome)
        
        # Check if retraining is needed
        if self._should_retrain():
            print("🧠 Triggering model retraining due to new data...")
            self.train_all_models()
    
    def _should_retrain(self) -> bool:
        """Determine if models should be retrained."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT MAX(timestamp) FROM training_logs
                """)
                last_training = cursor.fetchone()[0]
                
                if not last_training:
                    return True
                
                hours_since_training = (time.time() - last_training) / 3600
                return hours_since_training > self.config.retrain_interval_hours
                
        except sqlite3.Error:
            return True
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        metrics = {
            "task_predictor": {
                "trained": self.task_predictor.is_trained,
                "metrics": asdict(self.task_predictor.metrics) if self.task_predictor.metrics else None
            },
            "agent_selector": {
                "trained": self.agent_selector.is_trained,
                "metrics": asdict(self.agent_selector.metrics) if self.agent_selector.metrics else None
            },
            "learning_engine_metrics": self.learning_engine.get_learning_metrics(),
            "training_config": asdict(self.config)
        }
        
        # Add training history
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT model_name, COUNT(*), AVG(accuracy_after), MAX(timestamp)
                    FROM training_logs
                    GROUP BY model_name
                """)
                
                training_history = {}
                for row in cursor.fetchall():
                    training_history[row[0]] = {
                        "training_sessions": row[1],
                        "avg_accuracy": row[2],
                        "last_training": row[3]
                    }
                
                metrics["training_history"] = training_history
                
        except sqlite3.Error:
            metrics["training_history"] = {}
        
        return metrics


# Integration function for existing neural training hook
def integrate_zen_neural_training(operation_data: Dict[str, Any]) -> None:
    """Integration point for existing neural training hook."""
    if operation_data.get("tool_name") == "zen_consultation":
        # Create ZEN outcome from operation data
        outcome = ZenLearningOutcome(
            consultation_id=operation_data.get("consultation_id", f"zen_{int(time.time())}"),
            prompt=operation_data.get("prompt", ""),
            complexity=operation_data.get("complexity", "medium"),
            coordination_type=operation_data.get("coordination_type", "SWARM"),
            agents_allocated=operation_data.get("agents_allocated", 0),
            agent_types=operation_data.get("agent_types", []),
            mcp_tools=operation_data.get("mcp_tools", []),
            execution_success=operation_data.get("success", False),
            user_satisfaction=operation_data.get("user_satisfaction", 0.5),
            actual_agents_needed=operation_data.get("actual_agents_needed"),
            performance_metrics=operation_data.get("performance_metrics", {}),
            lessons_learned=operation_data.get("lessons_learned", []),
            timestamp=time.time()
        )
        
        # Update models
        pipeline = ZenNeuralTrainingPipeline()
        pipeline.update_models_from_outcome(outcome)
        
        print("🧠 ZEN Neural Training: Updated models with consultation outcome")


if __name__ == "__main__":
    # Test the neural training pipeline
    pipeline = ZenNeuralTrainingPipeline()
    results = pipeline.train_all_models()
    print(f"Training results: {results}")
    
    metrics = pipeline.get_training_metrics()
    print(f"Training metrics: {json.dumps(metrics, indent=2)}")