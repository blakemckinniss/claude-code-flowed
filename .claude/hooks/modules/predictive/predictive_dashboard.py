#!/usr/bin/env python3
"""Predictive Intelligence Dashboard - Phase 3 Visualization.

This module provides a unified dashboard interface for monitoring and visualizing
all predictive intelligence components.

Features:
- Real-time prediction monitoring
- Resource utilization forecasting
- Risk assessment visualization
- Agent pre-positioning status
- Timeline prediction accuracy
"""

import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import logging

# Import orchestrator
from .proactive_orchestrator import get_proactive_orchestrator

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetric:
    """Individual dashboard metric."""
    name: str
    value: Any
    unit: str
    status: str  # 'good', 'warning', 'critical'
    trend: str  # 'up', 'down', 'stable'
    history: deque = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=100)


class PredictiveDashboard:
    """Unified dashboard for predictive intelligence."""
    
    def __init__(self):
        self.metrics: Dict[str, DashboardMetric] = {}
        self.last_update = 0
        self.update_interval = 1.0  # 1 second
        
        # Initialize core metrics
        self._initialize_metrics()
        
        # Visualization templates
        self.display_templates = {
            'summary': self._generate_summary_view,
            'predictions': self._generate_predictions_view,
            'resources': self._generate_resources_view,
            'risks': self._generate_risks_view,
            'timeline': self._generate_timeline_view,
            'agents': self._generate_agents_view
        }
    
    def _initialize_metrics(self):
        """Initialize dashboard metrics."""
        # Prediction metrics
        self.metrics['prediction_latency'] = DashboardMetric(
            name="Prediction Latency",
            value=0.0,
            unit="ms",
            status="good",
            trend="stable"
        )
        
        self.metrics['active_predictions'] = DashboardMetric(
            name="Active Predictions",
            value=0,
            unit="count",
            status="good",
            trend="stable"
        )
        
        # Resource metrics
        self.metrics['memory_efficiency'] = DashboardMetric(
            name="Memory Efficiency",
            value=76.0,
            unit="%",
            status="good",
            trend="stable"
        )
        
        self.metrics['cpu_utilization'] = DashboardMetric(
            name="CPU Utilization",
            value=2.2,
            unit="%",
            status="good",
            trend="stable"
        )
        
        # Risk metrics
        self.metrics['risk_level'] = DashboardMetric(
            name="Overall Risk Level",
            value="low",
            unit="level",
            status="good",
            trend="stable"
        )
        
        # Agent metrics
        self.metrics['pre_positioned_agents'] = DashboardMetric(
            name="Pre-Positioned Agents",
            value=0,
            unit="agents",
            status="good",
            trend="stable"
        )
    
    async def update(self):
        """Update dashboard with latest data."""
        current_time = time.time()
        
        if current_time - self.last_update < self.update_interval:
            return
        
        try:
            # Get orchestrator status
            orchestrator = await get_proactive_orchestrator()
            status = orchestrator.get_orchestration_status()
            
            # Update metrics
            self._update_metric('prediction_latency', 
                              status['metrics']['average_prediction_latency_ms'],
                              target_max=100)
            
            self._update_metric('active_predictions',
                              status['active_predictions'],
                              target_range=(5, 20))
            
            self._update_metric('memory_efficiency',
                              status['metrics']['memory_efficiency'] * 100,
                              target_min=76)
            
            # Get performance monitor data
            perf_monitor = orchestrator.performance_monitor
            current_perf = perf_monitor.get_current_metrics()
            
            self._update_metric('cpu_utilization',
                              current_perf.get('cpu', 0),
                              target_max=80)
            
            # Update risk level
            risk_counts = self._analyze_risk_levels(orchestrator)
            overall_risk = self._determine_overall_risk(risk_counts)
            self._update_metric('risk_level', overall_risk, categorical=True)
            
            self._update_metric('pre_positioned_agents',
                              status['pre_positioned_agents'],
                              target_range=(2, 10))
            
            self.last_update = current_time
            
        except Exception as e:
            logger.exception(f"Dashboard update error: {e}")
    
    def _update_metric(self, name: str, value: Any, 
                      target_min: Optional[float] = None,
                      target_max: Optional[float] = None,
                      target_range: Optional[Tuple[float, float]] = None,
                      categorical: bool = False):
        """Update a metric with status and trend calculation."""
        if name not in self.metrics:
            return
        
        metric = self.metrics[name]
        old_value = metric.value
        metric.value = value
        
        # Add to history
        metric.history.append({
            'timestamp': time.time(),
            'value': value
        })
        
        # Calculate status
        if categorical:
            if value in ['low', 'good', 'healthy']:
                metric.status = 'good'
            elif value in ['medium', 'warning']:
                metric.status = 'warning'
            elif value in ['high', 'critical', 'degraded']:
                metric.status = 'critical'
        else:
            if target_min is not None and value < target_min:
                metric.status = 'critical'
            elif target_max is not None and value > target_max:
                metric.status = 'critical'
            elif target_range is not None:
                if value < target_range[0]:
                    metric.status = 'warning'
                elif value > target_range[1]:
                    metric.status = 'warning'
                else:
                    metric.status = 'good'
            else:
                metric.status = 'good'
        
        # Calculate trend
        if not categorical and isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
            if value > old_value * 1.05:
                metric.trend = 'up'
            elif value < old_value * 0.95:
                metric.trend = 'down'
            else:
                metric.trend = 'stable'
    
    def _analyze_risk_levels(self, orchestrator) -> Dict[str, int]:
        """Analyze risk levels from active predictions."""
        risk_counts = defaultdict(int)
        
        for prediction in orchestrator.active_predictions.values():
            if prediction.risk_assessment:
                risk_level = prediction.risk_assessment.get('overall_risk', 'unknown')
                risk_counts[risk_level] += 1
        
        return dict(risk_counts)
    
    def _determine_overall_risk(self, risk_counts: Dict[str, int]) -> str:
        """Determine overall risk level."""
        if risk_counts.get('critical', 0) > 0:
            return 'critical'
        elif risk_counts.get('high', 0) > 2:
            return 'high'
        elif risk_counts.get('medium', 0) > 5:
            return 'medium'
        else:
            return 'low'
    
    async def get_dashboard_view(self, view_type: str = 'summary') -> str:
        """Get formatted dashboard view."""
        # Update data first
        await self.update()
        
        # Generate requested view
        if view_type in self.display_templates:
            return await self.display_templates[view_type]()
        else:
            return await self._generate_summary_view()
    
    async def _generate_summary_view(self) -> str:
        """Generate summary dashboard view."""
        lines = []
        lines.append("=" * 80)
        lines.append("🎯 PREDICTIVE INTELLIGENCE DASHBOARD - PHASE 3")
        lines.append("=" * 80)
        lines.append("")
        
        # System Health
        lines.append("📊 SYSTEM HEALTH")
        lines.append("-" * 40)
        
        for metric_name in ['memory_efficiency', 'cpu_utilization', 'prediction_latency']:
            metric = self.metrics[metric_name]
            status_icon = self._get_status_icon(metric.status)
            trend_icon = self._get_trend_icon(metric.trend)
            
            lines.append(f"{status_icon} {metric.name}: {metric.value:.1f}{metric.unit} {trend_icon}")
        
        lines.append("")
        
        # Predictions
        lines.append("🔮 PREDICTIONS")
        lines.append("-" * 40)
        
        active = self.metrics['active_predictions']
        lines.append(f"Active Predictions: {active.value}")
        
        # Get prediction timeline
        orchestrator = await get_proactive_orchestrator()
        timeline = orchestrator._get_prediction_timeline()
        
        if timeline:
            lines.append("\nUpcoming Workloads:")
            for i, pred in enumerate(timeline[:5]):  # Show top 5
                arrival = pred['time_until_arrival']
                confidence = pred['confidence'] * 100
                lines.append(f"  {i+1}. In {arrival:.0f}s (Confidence: {confidence:.0f}%)")
        
        lines.append("")
        
        # Resources
        lines.append("💾 RESOURCE FORECAST")
        lines.append("-" * 40)
        
        resource_data = orchestrator._get_resource_utilization_data()
        current = resource_data['current']
        predicted = resource_data['predicted']
        
        lines.append(f"CPU:    Current: {current['cpu']:.1f}% → Predicted: +{predicted['cpu']:.1f}%")
        lines.append(f"Memory: Current: {current['memory']:.1f}% → Predicted: +{predicted['memory']:.1f}%")
        
        lines.append("")
        
        # Agents
        lines.append("🤖 AGENT PRE-POSITIONING")
        lines.append("-" * 40)
        
        agents = self.metrics['pre_positioned_agents']
        lines.append(f"Pre-Positioned: {agents.value} agents")
        
        distribution = orchestrator._get_agent_distribution()
        if distribution:
            lines.append("Distribution:")
            for agent_type, count in distribution.items():
                lines.append(f"  - {agent_type}: {count}")
        
        lines.append("")
        
        # Risk Assessment
        lines.append("⚠️  RISK ASSESSMENT")
        lines.append("-" * 40)
        
        risk = self.metrics['risk_level']
        risk_icon = self._get_status_icon(risk.status)
        lines.append(f"{risk_icon} Overall Risk: {risk.value.upper()}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        
        return "\n".join(lines)
    
    async def _generate_predictions_view(self) -> str:
        """Generate detailed predictions view."""
        lines = []
        lines.append("🔮 PREDICTION DETAILS")
        lines.append("=" * 80)
        
        orchestrator = await get_proactive_orchestrator()
        
        # Active predictions
        lines.append(f"\nActive Predictions: {len(orchestrator.active_predictions)}")
        lines.append("-" * 40)
        
        for workload in list(orchestrator.active_predictions.values())[:10]:
            lines.append(f"\nWorkload ID: {workload.workload_id}")
            lines.append(f"  Confidence: {workload.confidence:.2%}")
            lines.append(f"  Expected Arrival: {workload.expected_arrival - time.time():.1f}s")
            lines.append(f"  Priority: {workload.priority.name}")
            lines.append("  Resource Requirements:")
            for resource, amount in workload.resource_requirements.items():
                lines.append(f"    - {resource}: {amount:.1f}")
            lines.append(f"  Agents Required: {', '.join(workload.agent_requirements)}")
        
        # Prediction accuracy
        lines.append("\n" + "=" * 40)
        lines.append("PREDICTION ACCURACY")
        lines.append("-" * 40)
        
        if orchestrator.prediction_history:
            recent = list(orchestrator.prediction_history)[-20:]
            avg_confidence = sum(p['confidence'] for p in recent) / len(recent)
            lines.append(f"Average Confidence: {avg_confidence:.2%}")
            lines.append(f"Predictions Made: {orchestrator.metrics.predictions_made}")
        
        return "\n".join(lines)
    
    async def _generate_resources_view(self) -> str:
        """Generate resource utilization view."""
        lines = []
        lines.append("💾 RESOURCE UTILIZATION & FORECAST")
        lines.append("=" * 80)
        
        orchestrator = await get_proactive_orchestrator()
        resource_data = orchestrator._get_resource_utilization_data()
        
        # Current utilization
        lines.append("\nCURRENT UTILIZATION")
        lines.append("-" * 40)
        
        current = resource_data['current']
        for resource, value in current.items():
            bar = self._generate_bar(value, 100)
            lines.append(f"{resource.upper():6} [{bar}] {value:.1f}%")
        
        # Predicted additional load
        lines.append("\nPREDICTED ADDITIONAL LOAD")
        lines.append("-" * 40)
        
        predicted = resource_data['predicted']
        for resource in ['cpu', 'memory']:
            if resource in predicted:
                current_val = current.get(resource, 0)
                predicted_val = predicted[resource]
                total = current_val + predicted_val
                
                lines.append(f"{resource.upper():6} +{predicted_val:.1f}% → Total: {total:.1f}%")
                
                if total > 80:
                    lines.append(f"        ⚠️  WARNING: {resource.upper()} may exceed safe threshold!")
        
        # Memory efficiency
        lines.append("\nMEMORY EFFICIENCY")
        lines.append("-" * 40)
        
        efficiency = self.metrics['memory_efficiency']
        lines.append(f"Current: {efficiency.value:.1f}%")
        lines.append("Target:  76.0% (minimum)")
        lines.append(f"Status:  {efficiency.status.upper()}")
        
        return "\n".join(lines)
    
    async def _generate_risks_view(self) -> str:
        """Generate risk assessment view."""
        lines = []
        lines.append("⚠️  RISK ASSESSMENT DETAILS")
        lines.append("=" * 80)
        
        orchestrator = await get_proactive_orchestrator()
        
        # Risk heatmap
        heatmap = orchestrator._get_risk_heatmap()
        
        if heatmap:
            lines.append("\nRISK HEATMAP")
            lines.append("-" * 40)
            
            # Group by risk level
            by_level = defaultdict(list)
            for item in heatmap:
                by_level[item['risk_level']].append(item)
            
            for level in ['critical', 'high', 'medium', 'low']:
                if level in by_level:
                    lines.append(f"\n{level.upper()} ({len(by_level[level])} workloads)")
                    for item in by_level[level][:5]:
                        arrival = item['arrival_time'] - time.time()
                        lines.append(f"  - {item['workload_id']}: {item['risk_factors']} factors, "
                                   f"arrives in {arrival:.0f}s")
        
        # Mitigation strategies
        lines.append("\nACTIVE MITIGATIONS")
        lines.append("-" * 40)
        
        if orchestrator.pre_positioned_agents:
            lines.append(f"✓ {len(orchestrator.pre_positioned_agents)} agent groups pre-positioned")
        else:
            lines.append("  No active mitigations")
        
        return "\n".join(lines)
    
    async def _generate_timeline_view(self) -> str:
        """Generate timeline forecast view."""
        lines = []
        lines.append("📅 TIMELINE FORECAST")
        lines.append("=" * 80)
        
        orchestrator = await get_proactive_orchestrator()
        timeline = orchestrator._get_prediction_timeline()
        
        if timeline:
            lines.append("\nWORKLOAD TIMELINE")
            lines.append("-" * 40)
            
            # Create timeline visualization
            time.time()
            for pred in timeline[:10]:
                arrival_time = pred['time_until_arrival']
                
                # Create visual timeline
                if arrival_time < 0:
                    marker = "✓"  # Already arrived
                elif arrival_time < 30:
                    marker = "⚡"  # Imminent
                elif arrival_time < 60:
                    marker = "⏰"  # Soon
                else:
                    marker = "📍"  # Future
                
                lines.append(f"{marker} T+{arrival_time:6.1f}s | "
                           f"Confidence: {pred['confidence']:.0%} | "
                           f"Agents: {pred['agents_required']}")
        
        # Performance trends
        lines.append("\nPERFORMANCE TRENDS")
        lines.append("-" * 40)
        
        trends = orchestrator._get_performance_trends()
        if 'prediction_accuracy' in trends:
            accuracy = trends['prediction_accuracy']
            if accuracy:
                recent_avg = sum(accuracy[-5:]) / 5
                lines.append(f"Recent Prediction Accuracy: {recent_avg:.1%}")
        
        return "\n".join(lines)
    
    async def _generate_agents_view(self) -> str:
        """Generate agent management view."""
        lines = []
        lines.append("🤖 AGENT PRE-POSITIONING STATUS")
        lines.append("=" * 80)
        
        orchestrator = await get_proactive_orchestrator()
        
        # Pre-positioned agents
        if orchestrator.pre_positioned_agents:
            lines.append("\nPRE-POSITIONED AGENTS")
            lines.append("-" * 40)
            
            for workload_id, position in orchestrator.pre_positioned_agents.items():
                lines.append(f"\nWorkload: {workload_id}")
                lines.append(f"  Agent Type: {position.agent_type}")
                lines.append(f"  Count: {position.count}")
                lines.append(f"  Ready At: T-{position.readiness_time - time.time():.1f}s")
                lines.append("  Resources:")
                for resource, amount in position.resource_allocation.items():
                    lines.append(f"    - {resource}: {amount:.1f}")
        else:
            lines.append("\nNo agents currently pre-positioned")
        
        # Agent distribution
        distribution = orchestrator._get_agent_distribution()
        if distribution:
            lines.append("\nAGENT DISTRIBUTION")
            lines.append("-" * 40)
            
            total = sum(distribution.values())
            for agent_type, count in sorted(distribution.items(), 
                                          key=lambda x: x[1], reverse=True):
                percent = (count / total * 100) if total > 0 else 0
                bar = self._generate_bar(percent, 100, width=20)
                lines.append(f"{agent_type:20} [{bar}] {count} ({percent:.0f}%)")
        
        return "\n".join(lines)
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for status."""
        icons = {
            'good': '✅',
            'warning': '⚠️ ',
            'critical': '🚨'
        }
        return icons.get(status, '•')
    
    def _get_trend_icon(self, trend: str) -> str:
        """Get icon for trend."""
        icons = {
            'up': '↑',
            'down': '↓',
            'stable': '→'
        }
        return icons.get(trend, '')
    
    def _generate_bar(self, value: float, max_value: float, width: int = 30) -> str:
        """Generate ASCII progress bar."""
        if max_value == 0:
            return ' ' * width
        
        filled = int((value / max_value) * width)
        filled = max(0, min(width, filled))
        
        return '█' * filled + '░' * (width - filled)
    
    async def get_dashboard_json(self) -> Dict[str, Any]:
        """Get dashboard data as JSON."""
        await self.update()
        
        orchestrator = await get_proactive_orchestrator()
        
        return {
            'timestamp': time.time(),
            'metrics': {
                name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'status': metric.status,
                    'trend': metric.trend
                }
                for name, metric in self.metrics.items()
            },
            'predictions': {
                'active': len(orchestrator.active_predictions),
                'timeline': orchestrator._get_prediction_timeline()
            },
            'resources': orchestrator._get_resource_utilization_data(),
            'agents': {
                'pre_positioned': len(orchestrator.pre_positioned_agents),
                'distribution': orchestrator._get_agent_distribution()
            },
            'risks': orchestrator._get_risk_heatmap(),
            'performance': orchestrator._get_performance_trends()
        }


# Global dashboard instance
_global_dashboard: Optional[PredictiveDashboard] = None


async def get_predictive_dashboard() -> PredictiveDashboard:
    """Get or create global predictive dashboard."""
    global _global_dashboard
    
    if _global_dashboard is None:
        _global_dashboard = PredictiveDashboard()
    
    return _global_dashboard


async def display_dashboard(view_type: str = 'summary'):
    """Display dashboard view."""
    dashboard = await get_predictive_dashboard()
    view = await dashboard.get_dashboard_view(view_type)
    print(view)


async def get_dashboard_data() -> Dict[str, Any]:
    """Get dashboard data as JSON."""
    dashboard = await get_predictive_dashboard()
    return await dashboard.get_dashboard_json()