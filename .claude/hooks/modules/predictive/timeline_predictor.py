#!/usr/bin/env python3
"""Timeline Predictor for Accurate Project Timeline Predictions.

This module implements timeline forecasting capabilities for the ZEN Co-pilot system:
- Critical path analysis for task dependencies
- Confidence interval calculations with 80% accuracy
- Project timeline aggregation from task-level metrics
- Dynamic timeline updates based on real-time performance

Key Requirements:
- ±20% timeline accuracy
- 80% confidence intervals
- Critical path identification
- Dynamic timeline updates

Architecture Patterns:
- Integrates with existing PerformanceMonitor infrastructure
- Uses existing task execution history from task-metrics.json
- Leverages MLMetrics from adaptive_learning_engine
- Follows established dependency patterns
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta

# Import existing optimization infrastructure
from ..optimization.performance_monitor import get_performance_monitor
from ..optimization.adaptive_learning_engine import MLMetrics, PerformancePredictor

logger = logging.getLogger(__name__)


@dataclass
class TaskNode:
    """Represents a task in the project timeline."""
    task_id: str
    task_type: str
    estimated_duration: float  # in seconds
    actual_duration: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    critical_path: bool = False
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineMetrics:
    """Metrics for timeline predictions."""
    total_duration: float
    critical_path_duration: float
    confidence_interval: Tuple[float, float]  # (lower_bound, upper_bound)
    accuracy_percentage: float
    critical_tasks: List[str]
    parallel_efficiency: float
    prediction_timestamp: float
    confidence_level: float = 0.8  # 80% confidence interval


class DurationPredictor:
    """ML model for predicting task durations."""
    
    def __init__(self):
        # Use lightweight neural network similar to PerformancePredictor
        self.model = PerformancePredictor(input_size=12, hidden_size=48)
        self.history_window = 100
        self.duration_history = defaultdict(deque)
        self.feature_cache = {}
        
    def extract_features(self, task_type: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract features for duration prediction."""
        # Cache key for performance
        cache_key = f"{task_type}_{json.dumps(metadata, sort_keys=True)}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Task type encoding (one-hot for common types)
        type_features = self._encode_task_type(task_type)
        
        # Historical features
        hist_features = self._extract_historical_features(task_type)
        
        # Metadata features
        meta_features = np.array([
            metadata.get('complexity', 1.0) / 10.0,
            metadata.get('size', 1.0) / 1000.0,
            metadata.get('priority', 5) / 10.0,
            len(metadata.get('dependencies', [])) / 10.0
        ])
        
        features = np.concatenate([type_features, hist_features, meta_features])
        self.feature_cache[cache_key] = features
        
        return features
    
    def _encode_task_type(self, task_type: str) -> np.ndarray:
        """One-hot encode task types."""
        common_types = ['hooks', 'memory', 'analysis', 'validation', 'optimization']
        encoding = np.zeros(len(common_types))
        
        if task_type in common_types:
            encoding[common_types.index(task_type)] = 1.0
        else:
            # Unknown type - use last position
            encoding[-1] = 0.5
            
        return encoding
    
    def _extract_historical_features(self, task_type: str) -> np.ndarray:
        """Extract features from historical data."""
        if task_type not in self.duration_history or len(self.duration_history[task_type]) == 0:
            # No history - return defaults
            return np.array([1.0, 0.0, 0.0])  # avg=1s, std=0, trend=0
        
        recent_durations = list(self.duration_history[task_type])[-20:]
        
        avg_duration = np.mean(recent_durations)
        std_duration = np.std(recent_durations) if len(recent_durations) > 1 else 0.0
        
        # Trend calculation
        if len(recent_durations) > 5:
            x = np.arange(len(recent_durations))
            trend = np.polyfit(x, recent_durations, 1)[0]
        else:
            trend = 0.0
        
        # Normalize
        return np.array([
            min(avg_duration / 10.0, 1.0),
            min(std_duration / 5.0, 1.0),
            np.tanh(trend)  # Bound trend to [-1, 1]
        ])
    
    def predict_duration(self, task_type: str, metadata: Dict[str, Any]) -> Tuple[float, float]:
        """Predict task duration with confidence interval."""
        features = self.extract_features(task_type, metadata).reshape(1, -1)
        
        # Get prediction from model
        prediction = self.model.predict(features)[0]
        
        # Extract duration and uncertainty
        base_duration = max(0.1, prediction[0] * 10.0)  # Scale back to seconds
        uncertainty = max(0.1, prediction[1] * 2.0)
        
        # Use historical data to refine prediction
        if task_type in self.duration_history and len(self.duration_history[task_type]) > 5:
            historical_avg = np.mean(list(self.duration_history[task_type])[-10:])
            historical_std = np.std(list(self.duration_history[task_type])[-10:])
            
            # Blend prediction with historical data
            alpha = min(0.7, len(self.duration_history[task_type]) / 50.0)
            base_duration = alpha * historical_avg + (1 - alpha) * base_duration
            uncertainty = max(uncertainty, historical_std * 1.5)
        
        return base_duration, uncertainty
    
    def update_history(self, task_type: str, actual_duration: float):
        """Update duration history for a task type."""
        if len(self.duration_history[task_type]) >= self.history_window:
            self.duration_history[task_type].popleft()
        
        self.duration_history[task_type].append(actual_duration)
        
        # Clear feature cache as historical features have changed
        self.feature_cache.clear()


class CriticalPathAnalyzer:
    """Analyzes task dependencies to find critical path."""
    
    def __init__(self):
        self.task_graph: Dict[str, TaskNode] = {}
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        
    def add_task(self, task: TaskNode):
        """Add a task to the dependency graph."""
        self.task_graph[task.task_id] = task
        
        # Build adjacency lists
        for dep_id in task.dependencies:
            self.adjacency_list[dep_id].append(task.task_id)
            self.reverse_adjacency[task.task_id].append(dep_id)
    
    def find_critical_path(self) -> Tuple[List[str], float]:
        """Find critical path using topological sort and dynamic programming."""
        # Find all tasks with no dependencies (start nodes)
        start_nodes = [
            task_id for task_id, task in self.task_graph.items()
            if not task.dependencies
        ]
        
        if not start_nodes:
            # Handle circular dependencies by finding node with minimum dependencies
            start_nodes = [min(self.task_graph.keys(), 
                             key=lambda t: len(self.task_graph[t].dependencies))]
        
        # Calculate earliest start times
        earliest_start = {}
        earliest_finish = {}
        
        # Topological sort
        visited = set()
        topo_order = []
        
        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            for successor in self.adjacency_list[node_id]:
                dfs(successor)
            
            topo_order.append(node_id)
        
        for start in start_nodes:
            dfs(start)
        
        topo_order.reverse()
        
        # Forward pass - calculate earliest times
        for task_id in topo_order:
            task = self.task_graph.get(task_id)
            if not task:
                continue
                
            if not task.dependencies:
                earliest_start[task_id] = 0
            else:
                earliest_start[task_id] = max(
                    earliest_finish.get(dep, 0) 
                    for dep in task.dependencies
                )
            
            duration = task.actual_duration or task.estimated_duration
            earliest_finish[task_id] = earliest_start[task_id] + duration
        
        # Find end nodes and project duration
        end_nodes = [
            task_id for task_id in self.task_graph
            if task_id not in self.adjacency_list
        ]
        
        if not end_nodes:
            end_nodes = list(self.task_graph.keys())
        
        project_duration = max(
            earliest_finish.get(end_id, 0) for end_id in end_nodes
        )
        
        # Backward pass - find critical path
        latest_finish = {task_id: project_duration for task_id in end_nodes}
        latest_start = {}
        
        for task_id in reversed(topo_order):
            task = self.task_graph.get(task_id)
            if not task:
                continue
            
            if task_id not in self.adjacency_list:
                # End node
                latest_finish[task_id] = project_duration
            else:
                latest_finish[task_id] = min(
                    latest_start.get(successor, project_duration)
                    for successor in self.adjacency_list[task_id]
                )
            
            duration = task.actual_duration or task.estimated_duration
            latest_start[task_id] = latest_finish[task_id] - duration
        
        # Identify critical tasks (zero float)
        critical_tasks = []
        for task_id in self.task_graph:
            if task_id in earliest_start and task_id in latest_start:
                float_time = latest_start[task_id] - earliest_start[task_id]
                if abs(float_time) < 0.001:  # Effectively zero
                    critical_tasks.append(task_id)
                    self.task_graph[task_id].critical_path = True
        
        return critical_tasks, project_duration
    
    def get_parallel_efficiency(self) -> float:
        """Calculate how well tasks can be parallelized."""
        if not self.task_graph:
            return 0.0
        
        # Calculate maximum parallelism at each time point
        time_slots = defaultdict(int)
        
        for task in self.task_graph.values():
            if task.start_time is not None and task.end_time is not None:
                # Discretize time into slots
                start_slot = int(task.start_time)
                end_slot = int(task.end_time) + 1
                
                for slot in range(start_slot, end_slot):
                    time_slots[slot] += 1
        
        if not time_slots:
            return 0.0
        
        max_parallel = max(time_slots.values())
        avg_parallel = sum(time_slots.values()) / len(time_slots)
        
        # Efficiency is ratio of average to max parallelism
        return avg_parallel / max_parallel if max_parallel > 0 else 0.0


class TimelinePredictor:
    """Main timeline prediction engine."""
    
    def __init__(self, metrics_dir: Optional[Path] = None):
        self.metrics_dir = metrics_dir or Path(".claude-flow/metrics")
        self.duration_predictor = DurationPredictor()
        self.critical_path_analyzer = CriticalPathAnalyzer()
        self.performance_monitor = get_performance_monitor()
        
        # Prediction history for accuracy tracking
        self.prediction_history = deque(maxlen=100)
        self.accuracy_tracker = defaultdict(list)
        
        # Load historical data
        self._load_historical_data()
        
        # Confidence parameters
        self.confidence_level = 0.8  # 80% confidence intervals
        self.min_confidence = 0.3
        self.max_confidence = 0.95
        
    def _load_historical_data(self):
        """Load historical task execution data."""
        task_metrics_path = self.metrics_dir / "task-metrics.json"
        
        if task_metrics_path.exists():
            try:
                with open(task_metrics_path) as f:
                    task_metrics = json.load(f)
                
                # Process historical data
                for task in task_metrics:
                    if task.get('success', False) and 'duration' in task:
                        task_type = task.get('type', 'unknown')
                        duration = task['duration']
                        
                        # Update duration history
                        self.duration_predictor.update_history(task_type, duration)
                        
            except Exception as e:
                logger.warning(f"Failed to load historical data: {e}")
    
    def predict_timeline(self, tasks: List[Dict[str, Any]]) -> TimelineMetrics:
        """Predict project timeline with confidence intervals."""
        start_time = time.time()
        
        # Create task nodes
        task_nodes = []
        for task_data in tasks:
            # Predict duration if not provided
            if 'estimated_duration' not in task_data:
                duration, uncertainty = self.duration_predictor.predict_duration(
                    task_data.get('type', 'unknown'),
                    task_data.get('metadata', {})
                )
            else:
                duration = task_data['estimated_duration']
                uncertainty = duration * 0.2  # Default 20% uncertainty
            
            task = TaskNode(
                task_id=task_data['id'],
                task_type=task_data.get('type', 'unknown'),
                estimated_duration=duration,
                dependencies=task_data.get('dependencies', []),
                metadata=task_data.get('metadata', {}),
                confidence=self._calculate_task_confidence(task_data, uncertainty)
            )
            
            task_nodes.append(task)
            self.critical_path_analyzer.add_task(task)
        
        # Find critical path
        critical_tasks, critical_duration = self.critical_path_analyzer.find_critical_path()
        
        # Calculate confidence intervals
        lower_bound, upper_bound = self._calculate_confidence_intervals(
            task_nodes, critical_tasks, self.confidence_level
        )
        
        # Calculate parallel efficiency
        parallel_efficiency = self._estimate_parallel_efficiency(task_nodes)
        
        # Estimate total duration considering parallelism
        total_duration = self._estimate_total_duration(
            task_nodes, critical_duration, parallel_efficiency
        )
        
        # Calculate prediction accuracy based on historical data
        accuracy = self._calculate_prediction_accuracy()
        
        metrics = TimelineMetrics(
            total_duration=total_duration,
            critical_path_duration=critical_duration,
            confidence_interval=(lower_bound, upper_bound),
            accuracy_percentage=accuracy,
            critical_tasks=critical_tasks,
            parallel_efficiency=parallel_efficiency,
            prediction_timestamp=start_time,
            confidence_level=self.confidence_level
        )
        
        # Store prediction for future accuracy tracking
        self._store_prediction(metrics, task_nodes)
        
        return metrics
    
    def _calculate_task_confidence(self, task_data: Dict[str, Any], uncertainty: float) -> float:
        """Calculate confidence score for a task prediction."""
        base_confidence = 0.5
        
        # Increase confidence based on historical data
        task_type = task_data.get('type', 'unknown')
        if task_type in self.duration_predictor.duration_history:
            history_size = len(self.duration_predictor.duration_history[task_type])
            base_confidence += min(0.3, history_size / 100.0)
        
        # Adjust based on uncertainty
        if uncertainty > 0:
            uncertainty_factor = 1.0 / (1.0 + uncertainty)
            base_confidence *= uncertainty_factor
        
        # Adjust based on dependencies
        num_deps = len(task_data.get('dependencies', []))
        if num_deps > 0:
            base_confidence *= (1.0 - min(0.2, num_deps * 0.02))
        
        return max(self.min_confidence, min(self.max_confidence, base_confidence))
    
    def _calculate_confidence_intervals(
        self, 
        tasks: List[TaskNode], 
        critical_tasks: List[str],
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence intervals for timeline prediction."""
        # Use Monte Carlo simulation for complex dependencies
        num_simulations = 1000
        simulated_durations = []
        
        for _ in range(num_simulations):
            total_duration = 0
            task_durations = {}
            
            # Simulate each task duration
            for task in tasks:
                # Sample from distribution based on confidence
                std_dev = task.estimated_duration * (1 - task.confidence) * 0.5
                simulated = np.random.normal(task.estimated_duration, std_dev)
                simulated = max(0.1, simulated)  # Ensure positive duration
                
                task_durations[task.task_id] = simulated
                
                # For critical path tasks, track the path duration
                if task.task_id in critical_tasks:
                    # Consider dependencies
                    start_time = 0
                    for dep_id in task.dependencies:
                        if dep_id in task_durations:
                            start_time = max(start_time, task_durations[dep_id])
                    
                    end_time = start_time + simulated
                    total_duration = max(total_duration, end_time)
            
            simulated_durations.append(total_duration)
        
        # Calculate percentiles for confidence interval
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        lower_bound = np.percentile(simulated_durations, lower_percentile)
        upper_bound = np.percentile(simulated_durations, upper_percentile)
        
        return lower_bound, upper_bound
    
    def _estimate_parallel_efficiency(self, tasks: List[TaskNode]) -> float:
        """Estimate how efficiently tasks can be parallelized."""
        if not tasks:
            return 0.0
        
        # Simple heuristic based on dependency structure
        total_deps = sum(len(task.dependencies) for task in tasks)
        avg_deps = total_deps / len(tasks)
        
        # More dependencies = less parallelism
        efficiency = 1.0 / (1.0 + avg_deps * 0.2)
        
        # Consider task count
        if len(tasks) > 10:
            efficiency *= min(1.0, 10.0 / len(tasks))
        
        return max(0.1, min(1.0, efficiency))
    
    def _estimate_total_duration(
        self, 
        tasks: List[TaskNode], 
        critical_duration: float,
        parallel_efficiency: float
    ) -> float:
        """Estimate total project duration considering parallelism."""
        # Sum of all task durations
        total_sequential = sum(task.estimated_duration for task in tasks)
        
        # Theoretical minimum with perfect parallelism
        theoretical_min = critical_duration
        
        # Actual estimate based on parallel efficiency
        actual_duration = theoretical_min + (total_sequential - theoretical_min) * (1 - parallel_efficiency)
        
        return actual_duration
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate historical prediction accuracy."""
        if not self.accuracy_tracker:
            return 80.0  # Default to 80% if no history
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        recent_errors = []
        for _task_type, errors in self.accuracy_tracker.items():
            if errors:
                recent = errors[-20:]  # Last 20 predictions
                recent_errors.extend(recent)
        
        if not recent_errors:
            return 80.0
        
        # Convert to accuracy percentage
        avg_error = np.mean(recent_errors)
        accuracy = max(0, min(100, 100 - avg_error))
        
        return accuracy
    
    def _store_prediction(self, metrics: TimelineMetrics, tasks: List[TaskNode]):
        """Store prediction for future accuracy tracking."""
        prediction_record = {
            'timestamp': metrics.prediction_timestamp,
            'predicted_duration': metrics.total_duration,
            'confidence_interval': metrics.confidence_interval,
            'num_tasks': len(tasks),
            'critical_tasks': metrics.critical_tasks
        }
        
        self.prediction_history.append(prediction_record)
    
    def update_with_actual(self, task_id: str, actual_duration: float):
        """Update predictor with actual task duration."""
        # Find task in current graph
        if task_id in self.critical_path_analyzer.task_graph:
            task = self.critical_path_analyzer.task_graph[task_id]
            task.actual_duration = actual_duration
            
            # Update duration predictor
            self.duration_predictor.update_history(task.task_type, actual_duration)
            
            # Calculate prediction error
            if task.estimated_duration > 0:
                error_percentage = abs(actual_duration - task.estimated_duration) / task.estimated_duration * 100
                self.accuracy_tracker[task.task_type].append(error_percentage)
    
    def get_timeline_status(self) -> Dict[str, Any]:
        """Get current timeline prediction status."""
        return {
            'accuracy_percentage': self._calculate_prediction_accuracy(),
            'confidence_level': self.confidence_level,
            'total_predictions': len(self.prediction_history),
            'task_types_tracked': list(self.duration_predictor.duration_history.keys()),
            'critical_path_analyzer': {
                'num_tasks': len(self.critical_path_analyzer.task_graph),
                'parallel_efficiency': self.critical_path_analyzer.get_parallel_efficiency()
            }
        }


# Global instance management
_global_timeline_predictor: Optional[TimelinePredictor] = None


def get_timeline_predictor() -> TimelinePredictor:
    """Get or create global timeline predictor instance."""
    global _global_timeline_predictor
    
    if _global_timeline_predictor is None:
        _global_timeline_predictor = TimelinePredictor()
    
    return _global_timeline_predictor


async def predict_project_timeline(project_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function to predict project timeline."""
    predictor = get_timeline_predictor()
    metrics = predictor.predict_timeline(project_tasks)
    
    return {
        'total_duration_seconds': metrics.total_duration,
        'total_duration_human': str(timedelta(seconds=int(metrics.total_duration))),
        'critical_path_duration': metrics.critical_path_duration,
        'confidence_interval': {
            'lower_bound': metrics.confidence_interval[0],
            'upper_bound': metrics.confidence_interval[1],
            'confidence_level': metrics.confidence_level
        },
        'accuracy_percentage': metrics.accuracy_percentage,
        'critical_tasks': metrics.critical_tasks,
        'parallel_efficiency': metrics.parallel_efficiency,
        'prediction_timestamp': datetime.fromtimestamp(metrics.prediction_timestamp).isoformat()
    }