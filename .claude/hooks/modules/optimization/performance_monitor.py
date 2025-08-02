"""Comprehensive Performance Monitoring for Hook System.

Features:
- Real-time metrics collection
- Distributed tracing
- Anomaly detection
- Performance profiling
- Resource usage tracking
"""

import time
import asyncio
import threading
import psutil
import os
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
from datetime import datetime, timezone
from contextlib import contextmanager
import traceback
import inspect
import functools


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "in_progress"
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, window_size: int = 300):  # 5 min window
        self.window_size = window_size
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        
        # Start background cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def record(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        
        with self._lock:
            self._metrics[metric_name].append(point)
    
    def increment(self, metric_name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record(metric_name, value, labels)
    
    def gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        self.record(metric_name, value, labels)
    
    def histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self.record(metric_name, value, labels)
    
    def get_metrics(self, metric_name: str, 
                   last_n_seconds: Optional[int] = None) -> List[MetricPoint]:
        """Get metric values."""
        with self._lock:
            if metric_name not in self._metrics:
                return []
            
            points = list(self._metrics[metric_name])
            
            if last_n_seconds:
                cutoff = time.time() - last_n_seconds
                points = [p for p in points if p.timestamp > cutoff]
            
            return points
    
    def get_aggregated_metrics(self, metric_name: str, 
                             aggregation: str = "mean",
                             last_n_seconds: int = 60) -> Dict[str, float]:
        """Get aggregated metrics."""
        points = self.get_metrics(metric_name, last_n_seconds)
        
        if not points:
            return {
                "count": 0,
                "mean": 0,
                "min": 0,
                "max": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0
            }
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "p50": statistics.quantiles(values, n=2)[0] if len(values) > 1 else values[0],
            "p95": statistics.quantiles(values, n=20)[18] if len(values) > 19 else max(values),
            "p99": statistics.quantiles(values, n=100)[98] if len(values) > 99 else max(values)
        }
    
    def _cleanup_loop(self):
        """Clean up old metrics periodically."""
        while True:
            time.sleep(60)  # Cleanup every minute
            
            cutoff = time.time() - self.window_size
            
            with self._lock:
                for _metric_name, points in self._metrics.items():
                    # Remove old points
                    while points and points[0].timestamp < cutoff:
                        points.popleft()


class DistributedTracer:
    """Distributed tracing implementation."""
    
    def __init__(self):
        self._traces: Dict[str, List[Span]] = defaultdict(list)
        self._active_spans: Dict[str, Span] = {}
        self._lock = threading.Lock()
        self._trace_counter = 0
        self._span_counter = 0
    
    def start_trace(self, operation_name: str) -> Tuple[str, str]:
        """Start a new trace."""
        with self._lock:
            self._trace_counter += 1
            trace_id = f"trace_{self._trace_counter}_{int(time.time() * 1000)}"
            span_id = self._create_span_id()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        with self._lock:
            self._traces[trace_id].append(span)
            self._active_spans[span_id] = span
        
        return trace_id, span_id
    
    def start_span(self, trace_id: str, parent_span_id: str, 
                   operation_name: str) -> str:
        """Start a new span within a trace."""
        with self._lock:
            span_id = self._create_span_id()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        with self._lock:
            self._traces[trace_id].append(span)
            self._active_spans[span_id] = span
        
        return span_id
    
    def end_span(self, span_id: str, status: str = "success"):
        """End a span."""
        with self._lock:
            if span_id in self._active_spans:
                span = self._active_spans[span_id]
                span.end_time = time.time()
                span.status = status
                del self._active_spans[span_id]
    
    def add_span_tag(self, span_id: str, key: str, value: Any):
        """Add tag to span."""
        with self._lock:
            if span_id in self._active_spans:
                self._active_spans[span_id].tags[key] = value
    
    def add_span_log(self, span_id: str, message: str, level: str = "info"):
        """Add log to span."""
        with self._lock:
            if span_id in self._active_spans:
                log_entry = {
                    "timestamp": time.time(),
                    "level": level,
                    "message": message
                }
                self._active_spans[span_id].logs.append(log_entry)
    
    @contextmanager
    def trace_operation(self, operation_name: str):
        """Context manager for tracing."""
        trace_id, span_id = self.start_trace(operation_name)
        
        try:
            yield trace_id, span_id
            self.end_span(span_id, "success")
        except Exception as e:
            self.add_span_tag(span_id, "error", True)
            self.add_span_tag(span_id, "error.message", str(e))
            self.add_span_log(span_id, traceback.format_exc(), "error")
            self.end_span(span_id, "error")
            raise
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return list(self._traces.get(trace_id, []))
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get trace summary."""
        spans = self.get_trace(trace_id)
        
        if not spans:
            return {}
        
        root_span = next((s for s in spans if s.parent_span_id is None), None)
        if not root_span:
            return {}
        
        total_duration = root_span.duration_ms
        span_count = len(spans)
        error_count = sum(1 for s in spans if s.status == "error")
        
        # Build span tree
        span_tree = self._build_span_tree(spans)
        
        return {
            "trace_id": trace_id,
            "root_operation": root_span.operation_name,
            "total_duration_ms": total_duration,
            "span_count": span_count,
            "error_count": error_count,
            "span_tree": span_tree
        }
    
    def _create_span_id(self) -> str:
        """Create unique span ID."""
        self._span_counter += 1
        return f"span_{self._span_counter}_{int(time.time() * 1000000)}"
    
    def _build_span_tree(self, spans: List[Span]) -> Dict[str, Any]:
        """Build hierarchical span tree."""
        span_map = {s.span_id: s for s in spans}
        tree = {}
        
        for span in spans:
            if span.parent_span_id is None:
                tree[span.span_id] = {
                    "operation": span.operation_name,
                    "duration_ms": span.duration_ms,
                    "status": span.status,
                    "children": self._get_child_spans(span.span_id, span_map)
                }
        
        return tree
    
    def _get_child_spans(self, parent_id: str, span_map: Dict[str, Span]) -> Dict[str, Any]:
        """Get child spans recursively."""
        children = {}
        
        for span_id, span in span_map.items():
            if span.parent_span_id == parent_id:
                children[span_id] = {
                    "operation": span.operation_name,
                    "duration_ms": span.duration_ms,
                    "status": span.status,
                    "children": self._get_child_spans(span_id, span_map)
                }
        
        return children


class ResourceMonitor:
    """Monitors system resource usage."""
    
    def __init__(self):
        self._process = psutil.Process()
        self._last_cpu_time = None
        self._last_check_time = None
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        # Memory usage
        memory_info = self._process.memory_info()
        memory_percent = self._process.memory_percent()
        
        # CPU usage
        cpu_percent = self._process.cpu_percent(interval=0.1)
        
        # File descriptors
        try:
            num_fds = self._process.num_fds()
        except AttributeError:
            # Windows doesn't support num_fds
            num_fds = len(self._process.open_files())
        
        # Thread count
        num_threads = self._process.num_threads()
        
        return {
            "memory_rss_mb": memory_info.rss / (1024 * 1024),
            "memory_vms_mb": memory_info.vms / (1024 * 1024),
            "memory_percent": memory_percent,
            "cpu_percent": cpu_percent,
            "num_fds": num_fds,
            "num_threads": num_threads,
            "timestamp": time.time()
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "memory_total_mb": psutil.virtual_memory().total / (1024 * 1024),
            "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }


class AnomalyDetector:
    """Detects performance anomalies."""
    
    def __init__(self, sensitivity: float = 3.0):
        self.sensitivity = sensitivity  # Number of std devs for anomaly
        self._baselines: Dict[str, Tuple[float, float]] = {}  # mean, std
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def update_baseline(self, metric_name: str, values: List[float]):
        """Update baseline statistics for a metric."""
        if len(values) < 10:
            return
        
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        
        self._baselines[metric_name] = (mean, std)
    
    def check_anomaly(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """Check if value is anomalous."""
        # Add to history
        self._history[metric_name].append(value)
        
        # Update baseline periodically
        if len(self._history[metric_name]) % 100 == 0:
            self.update_baseline(metric_name, list(self._history[metric_name]))
        
        # Check against baseline
        if metric_name in self._baselines:
            mean, std = self._baselines[metric_name]
            
            if std > 0:
                z_score = abs(value - mean) / std
                
                if z_score > self.sensitivity:
                    return {
                        "metric": metric_name,
                        "value": value,
                        "baseline_mean": mean,
                        "baseline_std": std,
                        "z_score": z_score,
                        "severity": "high" if z_score > 5 else "medium"
                    }
        
        return None


class PerformanceProfiler:
    """Profile code performance."""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = defaultdict(list)
    
    def profile(self, name: str):
        """Decorator for profiling functions."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    self.profiles[name].append(duration * 1000)  # Convert to ms
            
            return wrapper
        return decorator
    
    @contextmanager
    def profile_block(self, name: str):
        """Context manager for profiling code blocks."""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.profiles[name].append(duration * 1000)
    
    def get_profile_stats(self, name: str) -> Dict[str, float]:
        """Get profiling statistics."""
        durations = self.profiles.get(name, [])
        
        if not durations:
            return {}
        
        return {
            "count": len(durations),
            "total_ms": sum(durations),
            "mean_ms": statistics.mean(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p50_ms": statistics.quantiles(durations, n=2)[0] if len(durations) > 1 else durations[0],
            "p95_ms": statistics.quantiles(durations, n=20)[18] if len(durations) > 19 else max(durations),
            "p99_ms": statistics.quantiles(durations, n=100)[98] if len(durations) > 99 else max(durations)
        }
    
    def get_all_profiles(self) -> Dict[str, Dict[str, float]]:
        """Get all profile statistics."""
        return {name: self.get_profile_stats(name) for name in self.profiles}


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.tracer = DistributedTracer()
        self.resource_monitor = ResourceMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.profiler = PerformanceProfiler()
        
        # Start background monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Collect resource metrics
                usage = self.resource_monitor.get_current_usage()
                
                # Record metrics
                self.metrics.gauge("process.memory.rss_mb", usage["memory_rss_mb"])
                self.metrics.gauge("process.cpu.percent", usage["cpu_percent"])
                self.metrics.gauge("process.threads", usage["num_threads"])
                self.metrics.gauge("process.fds", usage["num_fds"])
                
                # Check for anomalies
                for metric, value in [
                    ("memory", usage["memory_rss_mb"]),
                    ("cpu", usage["cpu_percent"])
                ]:
                    anomaly = self.anomaly_detector.check_anomaly(f"resource.{metric}", value)
                    if anomaly:
                        self.metrics.increment("anomalies.detected", labels={"metric": metric})
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)
    
    def record_hook_execution(self, hook_name: str, duration_ms: float, 
                            success: bool, error: Optional[str] = None):
        """Record hook execution metrics."""
        labels = {
            "hook": hook_name,
            "status": "success" if success else "error"
        }
        
        self.metrics.histogram("hook.duration_ms", duration_ms, labels)
        self.metrics.increment("hook.executions", labels=labels)
        
        if error:
            self.metrics.increment("hook.errors", labels={"hook": hook_name, "error": error})
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resource_usage": self.resource_monitor.get_current_usage(),
            "system_info": self.resource_monitor.get_system_info(),
            "hook_metrics": {
                "executions": self.metrics.get_aggregated_metrics("hook.executions", last_n_seconds=300),
                "duration": self.metrics.get_aggregated_metrics("hook.duration_ms", last_n_seconds=300),
                "errors": self.metrics.get_aggregated_metrics("hook.errors", last_n_seconds=300)
            },
            "profiles": self.profiler.get_all_profiles(),
            "anomalies": self.metrics.get_aggregated_metrics("anomalies.detected", last_n_seconds=3600)
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in various formats."""
        data = self.get_dashboard_data()
        
        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "prometheus":
            # Simplified Prometheus format
            lines = []
            
            for metric_type in ["executions", "duration", "errors"]:
                stats = data["hook_metrics"][metric_type]
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"hook_{metric_type}_{stat_name} {value}")
            
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def shutdown(self):
        """Shutdown monitoring."""
        self._monitoring = False


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor