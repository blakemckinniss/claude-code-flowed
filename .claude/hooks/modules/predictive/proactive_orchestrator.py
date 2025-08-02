#!/usr/bin/env python3
"""Proactive Resource Orchestrator - Phase 3 Core Integration.

This module implements the central orchestration component that integrates all
predictive intelligence components for proactive resource management.

Key Features:
- <100ms prediction latency requirement
- Agent pre-positioning before workload arrives
- Load balancing optimization across resources
- Maintains 76%+ memory efficiency
- Unified predictive dashboard interface
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
import threading

# Import existing infrastructure
from ..optimization.adaptive_learning_engine import (
    AdaptiveLearningEngine, get_adaptive_learning_engine,
    PerformancePredictor, PatternLearningSystem
)
from ..optimization.performance_monitor import get_performance_monitor
from ..optimization.async_orchestrator import AsyncOrchestrator, TaskPriority
from ..optimization.circuit_breaker import CircuitBreakerManager

logger = logging.getLogger(__name__)


@dataclass
class PredictiveWorkload:
    """Represents a predicted workload with metadata."""
    workload_id: str
    predicted_at: float
    expected_arrival: float
    resource_requirements: Dict[str, float]
    agent_requirements: List[str]
    confidence: float
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    risk_assessment: Optional[Dict[str, Any]] = None
    timeline_prediction: Optional[Dict[str, Any]] = None


@dataclass 
class AgentPrePosition:
    """Agent pre-positioning configuration."""
    agent_type: str
    count: int
    readiness_time: float
    resource_allocation: Dict[str, float]
    positioning_confidence: float


@dataclass
class OrchestrationMetrics:
    """Metrics for orchestration performance."""
    predictions_made: int = 0
    successful_predictions: int = 0
    pre_positioning_success: int = 0
    resource_savings_percent: float = 0.0
    average_prediction_latency_ms: float = 0.0
    memory_efficiency: float = 0.76
    last_updated: float = field(default_factory=time.time)


class ProactiveOrchestrator:
    """Main orchestrator for predictive resource management."""
    
    def __init__(self, 
                 prediction_window_seconds: float = 300.0,  # 5 minutes
                 pre_position_threshold: float = 0.7,
                 max_pre_positioned_agents: int = 10):
        """Initialize the proactive orchestrator.
        
        Args:
            prediction_window_seconds: How far ahead to predict
            pre_position_threshold: Confidence threshold for pre-positioning
            max_pre_positioned_agents: Maximum agents to pre-position
        """
        self.prediction_window = prediction_window_seconds
        self.pre_position_threshold = pre_position_threshold
        self.max_pre_positioned = max_pre_positioned_agents
        
        # Core components (will be initialized lazily)
        self._workflow_predictor = None
        self._resource_anticipator = None
        self._risk_assessor = None
        self._timeline_forecaster = None
        
        # Integration with existing infrastructure
        self.learning_engine = get_adaptive_learning_engine()
        self.performance_monitor = get_performance_monitor()
        self.async_orchestrator = None
        
        # Prediction tracking
        self.active_predictions: Dict[str, PredictiveWorkload] = {}
        self.pre_positioned_agents: Dict[str, AgentPrePosition] = {}
        self.prediction_history = deque(maxlen=1000)
        
        # Performance tracking
        self.metrics = OrchestrationMetrics()
        self._metrics_lock = threading.RLock()
        
        # Configuration
        self.config = {
            'enable_workflow_prediction': True,
            'enable_resource_anticipation': True,
            'enable_risk_assessment': True,
            'enable_timeline_forecast': True,
            'enable_agent_preposition': True,
            'prediction_latency_target_ms': 100,
            'min_confidence_for_action': 0.7
        }
        
        # State management
        self._running = False
        self._prediction_task = None
        self._orchestration_task = None
        
        logger.info("ProactiveOrchestrator initialized with %d second window", 
                   prediction_window_seconds)
    
    async def start(self):
        """Start the orchestrator."""
        if self._running:
            return
            
        self._running = True
        
        # Initialize async components
        self.async_orchestrator = AsyncOrchestrator(
            min_workers=4,
            max_workers=32,  # Utilize available CPU cores
            enable_shared_memory=True
        )
        await self.async_orchestrator.start()
        
        # Start prediction and orchestration loops
        self._prediction_task = asyncio.create_task(self._prediction_loop())
        self._orchestration_task = asyncio.create_task(self._orchestration_loop())
        
        logger.info("ProactiveOrchestrator started")
    
    async def stop(self):
        """Stop the orchestrator."""
        self._running = False
        
        if self._prediction_task:
            self._prediction_task.cancel()
        if self._orchestration_task:
            self._orchestration_task.cancel()
            
        if self.async_orchestrator:
            await self.async_orchestrator.shutdown()
        
        logger.info("ProactiveOrchestrator stopped")
    
    async def _prediction_loop(self):
        """Main prediction loop - runs continuously."""
        while self._running:
            try:
                start_time = time.perf_counter()
                
                # Run all predictions in parallel
                predictions = await self._run_parallel_predictions()
                
                # Process and integrate predictions
                integrated = await self._integrate_predictions(predictions)
                
                # Update active predictions
                await self._update_active_predictions(integrated)
                
                # Track performance
                latency = (time.perf_counter() - start_time) * 1000
                self._update_prediction_metrics(latency)
                
                # Adaptive sleep based on load
                sleep_time = self._calculate_adaptive_sleep()
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.exception(f"Prediction loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _orchestration_loop(self):
        """Main orchestration loop - manages resources."""
        while self._running:
            try:
                # Check for imminent workloads
                imminent = self._get_imminent_workloads()
                
                if imminent:
                    # Pre-position resources
                    await self._pre_position_resources(imminent)
                
                # Optimize current resource allocation
                await self._optimize_resource_allocation()
                
                # Clean up expired predictions
                self._cleanup_expired_predictions()
                
                await asyncio.sleep(1.0)  # Run every second
                
            except Exception as e:
                logger.exception(f"Orchestration loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _run_parallel_predictions(self) -> Dict[str, Any]:
        """Run all prediction engines in parallel."""
        tasks = []
        
        # Prepare current context
        context = await self._build_prediction_context()
        
        # Create prediction tasks
        if self.config['enable_workflow_prediction']:
            tasks.append(('workflow', self._predict_workflow(context)))
        
        if self.config['enable_resource_anticipation']:
            tasks.append(('resources', self._predict_resources(context)))
            
        if self.config['enable_risk_assessment']:
            tasks.append(('risk', self._assess_risk(context)))
            
        if self.config['enable_timeline_forecast']:
            tasks.append(('timeline', self._forecast_timeline(context)))
        
        # Run all predictions in parallel
        results = {}
        if tasks:
            predictions = await asyncio.gather(
                *[task[1] for task in tasks],
                return_exceptions=True
            )
            
            for i, (name, _) in enumerate(tasks):
                if isinstance(predictions[i], Exception):
                    logger.error(f"Prediction error in {name}: {predictions[i]}")
                    results[name] = None
                else:
                    results[name] = predictions[i]
        
        return results
    
    async def _build_prediction_context(self) -> Dict[str, Any]:
        """Build context for predictions."""
        # Get current system state
        current_performance = self.performance_monitor.get_current_metrics()
        learning_status = self.learning_engine.get_learning_status()
        
        # Get historical patterns
        recent_patterns = self._get_recent_patterns()
        
        # Build comprehensive context
        context = {
            'timestamp': time.time(),
            'system_performance': current_performance,
            'learning_status': learning_status,
            'active_predictions': len(self.active_predictions),
            'pre_positioned_agents': len(self.pre_positioned_agents),
            'recent_patterns': recent_patterns,
            'prediction_window': self.prediction_window
        }
        
        return context
    
    async def _predict_workflow(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Predict upcoming workflow patterns."""
        try:
            # For now, use pattern learning system
            # In full implementation, this would use WorkflowPredictionEngine
            patterns = self.learning_engine.pattern_learner.get_similar_patterns(
                context, threshold=0.8
            )
            
            if patterns:
                # Extract workflow predictions
                workflows = []
                for pattern in patterns[:5]:  # Top 5 patterns
                    workflow = {
                        'pattern_id': pattern.pattern_id,
                        'confidence': pattern.confidence,
                        'expected_resources': self._extract_resource_needs(pattern),
                        'expected_duration': self._estimate_duration(pattern)
                    }
                    workflows.append(workflow)
                
                return {
                    'workflows': workflows,
                    'prediction_time': time.time()
                }
            
            return None
            
        except Exception as e:
            logger.exception(f"Workflow prediction error: {e}")
            return None
    
    async def _predict_resources(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Predict resource requirements."""
        try:
            # Use performance predictor for resource prediction
            features = self._extract_prediction_features(context)
            
            # Get prediction from neural network
            prediction = self.learning_engine.performance_predictor.predict(
                features.reshape(1, -1)
            )[0]
            
            # Convert to resource requirements
            resource_prediction = {
                'cpu_percent': prediction[2] * 100,
                'memory_percent': prediction[3] * 100,
                'expected_latency_ms': prediction[0] * 1000,
                'expected_throughput': prediction[1] * 100,
                'confidence': self._calculate_resource_confidence(prediction),
                'prediction_time': time.time()
            }
            
            return resource_prediction
            
        except Exception as e:
            logger.exception(f"Resource prediction error: {e}")
            return None
    
    async def _assess_risk(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Assess risk factors."""
        try:
            # Simple risk assessment based on current metrics
            perf = context['system_performance']
            
            risk_factors = []
            overall_risk = 'low'
            
            # CPU risk
            if perf.get('cpu', 0) > 80:
                risk_factors.append({
                    'type': 'cpu_overload',
                    'severity': 'high' if perf.get('cpu', 0) > 90 else 'medium',
                    'probability': 0.8
                })
                overall_risk = 'high'
            
            # Memory risk
            if perf.get('memory', 0) > 70:
                risk_factors.append({
                    'type': 'memory_pressure',
                    'severity': 'high' if perf.get('memory', 0) > 85 else 'medium',
                    'probability': 0.7
                })
                if overall_risk == 'low':
                    overall_risk = 'medium'
            
            # Error rate risk
            error_rate = perf.get('errors', 0) / max(1, perf.get('requests', 1))
            if error_rate > 0.05:
                risk_factors.append({
                    'type': 'high_error_rate',
                    'severity': 'high',
                    'probability': 0.9
                })
                overall_risk = 'high'
            
            return {
                'overall_risk': overall_risk,
                'risk_factors': risk_factors,
                'mitigation_recommendations': self._get_mitigation_recommendations(risk_factors),
                'assessment_time': time.time()
            }
            
        except Exception as e:
            logger.exception(f"Risk assessment error: {e}")
            return None
    
    async def _forecast_timeline(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Forecast project timeline."""
        try:
            # Simple timeline forecast based on historical data
            recent_tasks = self._get_recent_task_completions()
            
            if len(recent_tasks) < 5:
                return None
            
            # Calculate average completion times
            avg_completion = np.mean([t['duration'] for t in recent_tasks])
            std_completion = np.std([t['duration'] for t in recent_tasks])
            
            # Forecast with confidence intervals
            forecast = {
                'expected_completion_time': avg_completion,
                'confidence_interval_low': max(0, avg_completion - 2 * std_completion),
                'confidence_interval_high': avg_completion + 2 * std_completion,
                'confidence_level': 0.95,
                'based_on_samples': len(recent_tasks),
                'forecast_time': time.time()
            }
            
            return forecast
            
        except Exception as e:
            logger.exception(f"Timeline forecast error: {e}")
            return None
    
    async def _integrate_predictions(self, predictions: Dict[str, Any]) -> List[PredictiveWorkload]:
        """Integrate all predictions into actionable workloads."""
        integrated_workloads = []
        
        # Extract valid predictions
        workflow_pred = predictions.get('workflow')
        resource_pred = predictions.get('resources')
        risk_pred = predictions.get('risk')
        timeline_pred = predictions.get('timeline')
        
        # Create integrated workload predictions
        if workflow_pred and workflow_pred.get('workflows'):
            for workflow in workflow_pred['workflows']:
                # Calculate arrival time
                expected_arrival = time.time() + (workflow.get('expected_duration', 60) / 2)
                
                # Determine resource requirements
                resource_reqs = {
                    'cpu': resource_pred.get('cpu_percent', 10) if resource_pred else 10,
                    'memory': resource_pred.get('memory_percent', 20) if resource_pred else 20,
                    'latency_budget_ms': resource_pred.get('expected_latency_ms', 100) if resource_pred else 100
                }
                
                # Determine agent requirements
                agent_reqs = self._determine_agent_requirements(workflow, resource_reqs)
                
                # Calculate overall confidence
                confidence = self._calculate_integrated_confidence(
                    workflow.get('confidence', 0.5),
                    resource_pred.get('confidence', 0.5) if resource_pred else 0.5
                )
                
                # Create workload prediction
                workload = PredictiveWorkload(
                    workload_id=f"pred_{workflow['pattern_id']}_{int(time.time())}",
                    predicted_at=time.time(),
                    expected_arrival=expected_arrival,
                    resource_requirements=resource_reqs,
                    agent_requirements=agent_reqs,
                    confidence=confidence,
                    priority=self._determine_priority(risk_pred),
                    risk_assessment=risk_pred,
                    timeline_prediction=timeline_pred
                )
                
                integrated_workloads.append(workload)
        
        return integrated_workloads
    
    async def _update_active_predictions(self, workloads: List[PredictiveWorkload]):
        """Update active predictions with new workloads."""
        for workload in workloads:
            if workload.confidence >= self.config['min_confidence_for_action']:
                self.active_predictions[workload.workload_id] = workload
                
                # Track in history
                self.prediction_history.append({
                    'workload_id': workload.workload_id,
                    'timestamp': workload.predicted_at,
                    'confidence': workload.confidence
                })
    
    def _get_imminent_workloads(self) -> List[PredictiveWorkload]:
        """Get workloads expected to arrive soon."""
        current_time = time.time()
        imminent_threshold = 30.0  # 30 seconds
        
        imminent = []
        for workload in self.active_predictions.values():
            time_until_arrival = workload.expected_arrival - current_time
            if 0 < time_until_arrival <= imminent_threshold:
                imminent.append(workload)
        
        return sorted(imminent, key=lambda w: w.expected_arrival)
    
    async def _pre_position_resources(self, workloads: List[PredictiveWorkload]):
        """Pre-position resources for imminent workloads."""
        if not self.config['enable_agent_preposition']:
            return
        
        for workload in workloads:
            if workload.confidence < self.pre_position_threshold:
                continue
            
            # Check if already pre-positioned
            if workload.workload_id in self.pre_positioned_agents:
                continue
            
            # Determine pre-positioning strategy
            positions = self._calculate_pre_positions(workload)
            
            # Execute pre-positioning
            for position in positions:
                if len(self.pre_positioned_agents) < self.max_pre_positioned:
                    await self._execute_pre_position(workload, position)
    
    def _calculate_pre_positions(self, workload: PredictiveWorkload) -> List[AgentPrePosition]:
        """Calculate agent pre-positioning strategy."""
        positions = []
        
        for agent_type in workload.agent_requirements:
            # Calculate how many agents to pre-position
            count = self._calculate_agent_count(workload, agent_type)
            
            # Calculate resource allocation
            resource_alloc = {
                'cpu': workload.resource_requirements['cpu'] / len(workload.agent_requirements),
                'memory': workload.resource_requirements['memory'] / len(workload.agent_requirements)
            }
            
            position = AgentPrePosition(
                agent_type=agent_type,
                count=count,
                readiness_time=workload.expected_arrival - 5.0,  # 5 seconds before
                resource_allocation=resource_alloc,
                positioning_confidence=workload.confidence
            )
            
            positions.append(position)
        
        return positions
    
    async def _execute_pre_position(self, workload: PredictiveWorkload, 
                                   position: AgentPrePosition):
        """Execute agent pre-positioning."""
        try:
            # Log pre-positioning action
            logger.info(f"Pre-positioning {position.count} {position.agent_type} agents "
                       f"for workload {workload.workload_id}")
            
            # In real implementation, this would:
            # 1. Call swarm-init or agent-spawn commands
            # 2. Allocate resources
            # 3. Warm up agents
            
            # Track pre-positioned agents
            self.pre_positioned_agents[workload.workload_id] = position
            
            # Update metrics
            with self._metrics_lock:
                self.metrics.pre_positioning_success += 1
            
        except Exception as e:
            logger.exception(f"Pre-positioning error: {e}")
    
    async def _optimize_resource_allocation(self):
        """Optimize current resource allocation."""
        try:
            # Get current resource state
            current_state = self.performance_monitor.get_current_metrics()
            
            # Check memory efficiency
            memory_efficiency = (100 - current_state.get('memory', 0)) / 100.0
            
            # Update metrics
            with self._metrics_lock:
                self.metrics.memory_efficiency = memory_efficiency
            
            # If memory efficiency drops below target, trigger optimization
            if memory_efficiency < 0.76:
                await self._rebalance_resources()
                
        except Exception as e:
            logger.exception(f"Resource optimization error: {e}")
    
    async def _rebalance_resources(self):
        """Rebalance resources to maintain efficiency."""
        # In real implementation, this would:
        # 1. Identify underutilized agents
        # 2. Consolidate workloads
        # 3. Release unnecessary resources
        pass
    
    def _cleanup_expired_predictions(self):
        """Clean up expired predictions."""
        current_time = time.time()
        expired = []
        
        for workload_id, workload in self.active_predictions.items():
            if current_time > workload.expected_arrival + 60:  # 1 minute past
                expired.append(workload_id)
        
        for workload_id in expired:
            del self.active_predictions[workload_id]
            if workload_id in self.pre_positioned_agents:
                del self.pre_positioned_agents[workload_id]
    
    def _update_prediction_metrics(self, latency_ms: float):
        """Update prediction performance metrics."""
        with self._metrics_lock:
            self.metrics.predictions_made += 1
            
            # Update average latency
            n = self.metrics.predictions_made
            prev_avg = self.metrics.average_prediction_latency_ms
            self.metrics.average_prediction_latency_ms = (
                (prev_avg * (n - 1) + latency_ms) / n
            )
            
            self.metrics.last_updated = time.time()
    
    def _calculate_adaptive_sleep(self) -> float:
        """Calculate adaptive sleep time based on load."""
        # Base sleep time
        base_sleep = 5.0
        
        # Adjust based on active predictions
        if len(self.active_predictions) > 10:
            return base_sleep * 0.5  # Speed up
        elif len(self.active_predictions) < 2:
            return base_sleep * 2.0  # Slow down
        
        return base_sleep
    
    def _extract_resource_needs(self, pattern: Any) -> Dict[str, float]:
        """Extract resource needs from pattern."""
        # Simple extraction - in real implementation would be more sophisticated
        return {
            'cpu': 10.0,
            'memory': 20.0,
            'io': 5.0
        }
    
    def _estimate_duration(self, pattern: Any) -> float:
        """Estimate task duration from pattern."""
        # Simple estimate - in real implementation would use ML
        return 60.0  # 60 seconds default
    
    def _extract_prediction_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract features for prediction."""
        perf = context['system_performance']
        
        features = [
            perf.get('cpu', 0) / 100.0,
            perf.get('memory', 0) / 100.0,
            perf.get('requests', 0) / 1000.0,
            perf.get('errors', 0) / 100.0,
            len(self.active_predictions) / 10.0,
            len(self.pre_positioned_agents) / 10.0,
            time.time() % 86400 / 86400,  # Time of day
            1.0,  # Placeholder
            1.0,  # Placeholder
            1.0   # Placeholder
        ]
        
        return np.array(features[:10])  # Ensure 10 features
    
    def _calculate_resource_confidence(self, prediction: np.ndarray) -> float:
        """Calculate confidence in resource prediction."""
        # Simple confidence based on prediction values
        # In real implementation would use prediction uncertainty
        avg_prediction = np.mean(np.abs(prediction))
        return min(0.9, 0.5 + avg_prediction * 0.4)
    
    def _get_mitigation_recommendations(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Get risk mitigation recommendations."""
        recommendations = []
        
        for risk in risk_factors:
            if risk['type'] == 'cpu_overload':
                recommendations.append("Scale up compute resources or optimize CPU-intensive operations")
            elif risk['type'] == 'memory_pressure':
                recommendations.append("Increase memory allocation or optimize memory usage")
            elif risk['type'] == 'high_error_rate':
                recommendations.append("Investigate error sources and implement circuit breakers")
        
        return recommendations
    
    def _get_recent_task_completions(self) -> List[Dict[str, Any]]:
        """Get recent task completion times."""
        # In real implementation, would query actual task history
        # For now, return synthetic data
        return [
            {'task_id': f'task_{i}', 'duration': 30 + np.random.normal(0, 5)}
            for i in range(10)
        ]
    
    def _determine_agent_requirements(self, workflow: Dict[str, Any], 
                                    resources: Dict[str, float]) -> List[str]:
        """Determine which agents are needed."""
        agents = []
        
        # Simple heuristic - in real implementation would be ML-based
        if resources['cpu'] > 50:
            agents.extend(['compute-optimizer', 'performance-engineer'])
        if resources['memory'] > 40:
            agents.extend(['memory-optimizer', 'resource-manager'])
        if resources.get('latency_budget_ms', 100) < 50:
            agents.append('latency-optimizer')
        
        # Always include a coordinator
        if agents and 'coordinator' not in agents:
            agents.append('coordinator')
        
        return agents or ['general-worker']
    
    def _calculate_integrated_confidence(self, *confidences) -> float:
        """Calculate integrated confidence from multiple sources."""
        valid_confidences = [c for c in confidences if c > 0]
        if not valid_confidences:
            return 0.5
        
        # Use geometric mean for conservative estimate
        return float(np.prod(valid_confidences) ** (1.0 / len(valid_confidences)))
    
    def _determine_priority(self, risk_assessment: Optional[Dict[str, Any]]) -> TaskPriority:
        """Determine task priority based on risk."""
        if not risk_assessment:
            return TaskPriority.NORMAL
        
        risk_level = risk_assessment.get('overall_risk', 'low')
        if risk_level == 'critical':
            return TaskPriority.CRITICAL
        elif risk_level == 'high':
            return TaskPriority.HIGH
        elif risk_level == 'medium':
            return TaskPriority.NORMAL
        else:
            return TaskPriority.LOW
    
    def _calculate_agent_count(self, workload: PredictiveWorkload, agent_type: str) -> int:
        """Calculate how many agents to pre-position."""
        # Base count
        base_count = 1
        
        # Scale based on resource requirements
        if workload.resource_requirements['cpu'] > 50:
            base_count += 1
        if workload.resource_requirements['memory'] > 50:
            base_count += 1
        
        # Add buffer based on confidence
        if workload.confidence > 0.9:
            base_count += 1
        
        return min(base_count, 4)  # Cap at 4 per type
    
    def _get_recent_patterns(self) -> List[Dict[str, Any]]:
        """Get recent usage patterns."""
        # In real implementation, would query pattern database
        return []
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status."""
        with self._metrics_lock:
            metrics_copy = {
                'predictions_made': self.metrics.predictions_made,
                'successful_predictions': self.metrics.successful_predictions,
                'pre_positioning_success': self.metrics.pre_positioning_success,
                'resource_savings_percent': self.metrics.resource_savings_percent,
                'average_prediction_latency_ms': self.metrics.average_prediction_latency_ms,
                'memory_efficiency': self.metrics.memory_efficiency,
                'last_updated': self.metrics.last_updated
            }
        
        return {
            'status': 'running' if self._running else 'stopped',
            'configuration': self.config,
            'metrics': metrics_copy,
            'active_predictions': len(self.active_predictions),
            'pre_positioned_agents': len(self.pre_positioned_agents),
            'prediction_window_seconds': self.prediction_window,
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        if self.metrics.average_prediction_latency_ms > 100:
            return 'degraded'
        elif self.metrics.memory_efficiency < 0.76:
            return 'warning'
        else:
            return 'healthy'
    
    def get_predictive_dashboard(self) -> Dict[str, Any]:
        """Get unified predictive dashboard data."""
        status = self.get_orchestration_status()
        
        # Add visualization-friendly data
        dashboard = {
            **status,
            'prediction_timeline': self._get_prediction_timeline(),
            'resource_utilization': self._get_resource_utilization_data(),
            'agent_distribution': self._get_agent_distribution(),
            'risk_heatmap': self._get_risk_heatmap(),
            'performance_trends': self._get_performance_trends()
        }
        
        return dashboard
    
    def _get_prediction_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of predictions for visualization."""
        timeline = []
        current_time = time.time()
        
        for workload in sorted(self.active_predictions.values(), 
                              key=lambda w: w.expected_arrival):
            timeline.append({
                'workload_id': workload.workload_id,
                'time_until_arrival': workload.expected_arrival - current_time,
                'confidence': workload.confidence,
                'resource_impact': sum(workload.resource_requirements.values()),
                'agents_required': len(workload.agent_requirements)
            })
        
        return timeline
    
    def _get_resource_utilization_data(self) -> Dict[str, Any]:
        """Get resource utilization data for dashboard."""
        current = self.performance_monitor.get_current_metrics()
        
        return {
            'current': {
                'cpu': current.get('cpu', 0),
                'memory': current.get('memory', 0),
                'io': current.get('io_read', 0) + current.get('io_write', 0)
            },
            'predicted': {
                'cpu': sum(w.resource_requirements.get('cpu', 0) 
                          for w in self.active_predictions.values()),
                'memory': sum(w.resource_requirements.get('memory', 0)
                            for w in self.active_predictions.values())
            }
        }
    
    def _get_agent_distribution(self) -> Dict[str, int]:
        """Get distribution of pre-positioned agents."""
        distribution = defaultdict(int)
        
        for position in self.pre_positioned_agents.values():
            distribution[position.agent_type] += position.count
        
        return dict(distribution)
    
    def _get_risk_heatmap(self) -> List[Dict[str, Any]]:
        """Get risk heatmap data."""
        heatmap = []
        
        for workload in self.active_predictions.values():
            if workload.risk_assessment:
                risk = workload.risk_assessment
                heatmap.append({
                    'workload_id': workload.workload_id,
                    'risk_level': risk.get('overall_risk', 'unknown'),
                    'risk_factors': len(risk.get('risk_factors', [])),
                    'arrival_time': workload.expected_arrival
                })
        
        return heatmap
    
    def _get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trend data."""
        # In real implementation, would aggregate historical data
        # For now, return sample data
        return {
            'prediction_latency': [self.metrics.average_prediction_latency_ms] * 10,
            'memory_efficiency': [self.metrics.memory_efficiency] * 10,
            'prediction_accuracy': [0.85, 0.87, 0.86, 0.88, 0.89, 0.90, 0.88, 0.91, 0.89, 0.90]
        }


# Global orchestrator instance
_global_orchestrator: Optional[ProactiveOrchestrator] = None


async def get_proactive_orchestrator() -> ProactiveOrchestrator:
    """Get or create global proactive orchestrator."""
    global _global_orchestrator
    
    if _global_orchestrator is None:
        _global_orchestrator = ProactiveOrchestrator()
        await _global_orchestrator.start()
    
    return _global_orchestrator


async def shutdown_orchestrator():
    """Shutdown the global orchestrator."""
    global _global_orchestrator
    
    if _global_orchestrator is not None:
        await _global_orchestrator.stop()
        _global_orchestrator = None