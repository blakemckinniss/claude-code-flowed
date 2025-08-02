#!/usr/bin/env python3
"""Performance Monitor Hook - Claude Flow Integration Phase 2

Real-time performance monitoring and bottleneck detection with
neural pattern integration and optimization recommendations.
"""

import sys
import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

# Neural pattern integration with fallback support - avoid namespace collision
neural_module = None
try:
    import modules.pre_tool.analyzers.neural_pattern_validator as neural_module
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False


def create_neural_pattern(**kwargs) -> Any:
    """Create a neural pattern using the available implementation."""
    if _NEURAL_AVAILABLE and neural_module:
        return neural_module.NeuralPattern(**kwargs)
    else:
        # Return a simple dict that mimics the interface
        return {
            'pattern_id': kwargs.get('pattern_id', ''),
            'tool_name': kwargs.get('tool_name', ''),
            'context_hash': kwargs.get('context_hash', ''),
            'success_count': kwargs.get('success_count', 0),
            'failure_count': kwargs.get('failure_count', 0),
            'confidence_score': kwargs.get('confidence_score', 0.0),
            'learned_optimization': kwargs.get('learned_optimization', ''),
            'created_timestamp': kwargs.get('created_timestamp', 0.0),
            'last_used_timestamp': kwargs.get('last_used_timestamp', 0.0),
            'performance_metrics': kwargs.get('performance_metrics', {})
        }


def create_neural_storage() -> Any:
    """Create a neural pattern storage using the available implementation."""
    if _NEURAL_AVAILABLE and neural_module:
        return neural_module.NeuralPatternStorage()
    else:
        # Return a simple fallback storage
        class FallbackStorage:
            def store_pattern(self, pattern: Any) -> bool:
                return True  # Fallback does nothing but maintains interface
            def get_patterns_for_tool(self, tool_name: str, min_confidence: float = 0.5) -> List[Any]:
                return []
        return FallbackStorage()


class PerformanceMonitor:
    """Monitors performance and identifies bottlenecks in real-time."""
    
    def __init__(self):
        self.hooks_dir = Path(__file__).parent
        self.metrics_db_path = self.hooks_dir / ".session" / "performance_metrics.db"
        self.neural_storage = create_neural_storage()
        self.session_start_time = time.time()
        
        # Performance thresholds
        self.thresholds = {
            "slow_operation": 5.0,  # seconds
            "memory_warning": 100,   # MB
            "error_rate_warning": 0.1,  # 10%
            "low_efficiency": 0.6    # 60%
        }
        
        self._initialize_metrics_db()
    
    def _initialize_metrics_db(self) -> None:
        """Initialize performance metrics database."""
        self.metrics_db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.metrics_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    optimization_score REAL DEFAULT 0.5,
                    bottleneck_detected BOOLEAN DEFAULT FALSE,
                    session_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bottleneck_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    bottleneck_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    recommendations TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    session_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    suggestion_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    description TEXT NOT NULL,
                    applied BOOLEAN DEFAULT FALSE,
                    effectiveness_score REAL,
                    session_id TEXT
                )
            """)
    
    def record_operation(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record operation performance metrics."""
        timestamp = datetime.now(timezone.utc).isoformat()
        session_id = operation_data.get("session_id", f"session_{int(time.time())}")
        
        # Extract performance metrics
        duration = operation_data.get("duration", 0.0)
        operation_type = operation_data.get("operation_type", "unknown")
        success = operation_data.get("success", True)
        error_message = operation_data.get("error_message", "")
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(operation_data)
        
        # Detect bottlenecks
        bottleneck_detected = self._detect_bottleneck(operation_data)
        
        # Store in database
        with sqlite3.connect(self.metrics_db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics (
                    timestamp, operation_type, duration_seconds, memory_usage_mb,
                    cpu_usage_percent, success, error_message, optimization_score,
                    bottleneck_detected, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, operation_type, duration,
                operation_data.get("memory_usage_mb", 0),
                operation_data.get("cpu_usage_percent", 0),
                success, error_message, optimization_score,
                bottleneck_detected, session_id
            ))
        
        # Generate recommendations if needed
        recommendations = []
        if bottleneck_detected:
            recommendations = self._generate_bottleneck_recommendations(operation_data)
        
        # Store successful patterns for neural learning
        if success and optimization_score > 0.7:
            self._store_performance_pattern(operation_data)
        
        return {
            "performance_recorded": True,
            "optimization_score": optimization_score,
            "bottleneck_detected": bottleneck_detected,
            "recommendations": recommendations
        }
    
    def _calculate_optimization_score(self, operation_data: Dict[str, Any]) -> float:
        """Calculate optimization score for operation."""
        score_factors = {
            "duration": 0.4,
            "memory": 0.2,
            "cpu": 0.2,
            "success": 0.2
        }
        
        scores = {}
        
        # Duration score (inverse relationship)
        duration = operation_data.get("duration", 0.0)
        if duration > 0:
            scores["duration"] = max(0.0, 1.0 - (duration / 10.0))  # 10s = 0 score
        else:
            scores["duration"] = 1.0
        
        # Memory score (inverse relationship)
        memory_mb = operation_data.get("memory_usage_mb", 0)
        scores["memory"] = max(0.0, 1.0 - (memory_mb / 200.0))  # 200MB = 0 score
        
        # CPU score (inverse relationship)
        cpu_percent = operation_data.get("cpu_usage_percent", 0)
        scores["cpu"] = max(0.0, 1.0 - (cpu_percent / 100.0))
        
        # Success score
        scores["success"] = 1.0 if operation_data.get("success", True) else 0.0
        
        # Weighted average
        optimization_score = sum(scores.get(factor, 0.5) * weight 
                               for factor, weight in score_factors.items())
        
        return min(1.0, max(0.0, optimization_score))
    
    def _detect_bottleneck(self, operation_data: Dict[str, Any]) -> bool:
        """Detect performance bottlenecks."""
        # Duration bottleneck
        if operation_data.get("duration", 0) > self.thresholds["slow_operation"]:
            return True
        
        # Memory bottleneck
        if operation_data.get("memory_usage_mb", 0) > self.thresholds["memory_warning"]:
            return True
        
        # Error rate bottleneck
        if not operation_data.get("success", True):
            return True
        
        return False
    
    def _generate_bottleneck_recommendations(self, operation_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for bottleneck resolution."""
        recommendations = []
        
        duration = operation_data.get("duration", 0.0)
        memory_mb = operation_data.get("memory_usage_mb", 0)
        operation_type = operation_data.get("operation_type", "")
        
        # Duration-based recommendations
        if duration > self.thresholds["slow_operation"]:
            recommendations.append("⏱️ Slow operation detected - consider batching or optimization")
            
            if "file" in operation_type.lower():
                recommendations.append("📁 File operation bottleneck - use parallel processing")
            elif "network" in operation_type.lower():
                recommendations.append("🌐 Network bottleneck - implement caching or retry logic")
            elif "database" in operation_type.lower():
                recommendations.append("🗄️ Database bottleneck - optimize queries or add indexes")
        
        # Memory-based recommendations
        if memory_mb > self.thresholds["memory_warning"]:
            recommendations.append("💾 High memory usage - consider streaming or chunking")
            recommendations.append("🔄 Implement garbage collection or resource cleanup")
        
        # Operation-specific recommendations
        if not operation_data.get("success", True):
            error_message = operation_data.get("error_message", "")
            if "timeout" in error_message.lower():
                recommendations.append("⏰ Timeout error - increase timeout or optimize operation")
            elif "memory" in error_message.lower():
                recommendations.append("💾 Memory error - reduce data size or use streaming")
            else:
                recommendations.append("🔧 Operation failed - review error handling and retry logic")
        
        return recommendations
    
    def _store_performance_pattern(self, operation_data: Dict[str, Any]) -> None:
        """Store successful performance patterns for neural learning."""
        try:
            pattern_id = f"perf_{operation_data.get('operation_type', 'unknown')}_{int(time.time())}"
            self._extract_success_factors(operation_data)
            
            # Create neural pattern using factory function (handles both implementations)
            pattern = create_neural_pattern(
                pattern_id=pattern_id,
                tool_name=operation_data.get("operation_type", "unknown"),
                context_hash=f"perf_{hash(str(operation_data))}",
                success_count=1,
                failure_count=0,
                confidence_score=self._calculate_optimization_score(operation_data),
                learned_optimization=f"Optimized {operation_data.get('operation_type', 'operation')} - duration: {operation_data.get('duration', 0.0):.2f}s",
                created_timestamp=time.time(),
                last_used_timestamp=time.time(),
                performance_metrics={
                    "duration": operation_data.get("duration", 0.0),
                    "memory_usage_mb": operation_data.get("memory_usage_mb", 0.0),
                    "cpu_usage_percent": operation_data.get("cpu_usage_percent", 0.0),
                    "optimization_score": self._calculate_optimization_score(operation_data)
                }
            )
            
            self.neural_storage.store_pattern(pattern)
            
        except Exception:
            pass  # Don't fail monitoring due to pattern storage issues
    
    def _extract_success_factors(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract factors that contributed to operation success."""
        return {
            "fast_execution": operation_data.get("duration", 0.0) < 1.0,
            "low_memory": operation_data.get("memory_usage_mb", 0) < 50,
            "low_cpu": operation_data.get("cpu_usage_percent", 0) < 50,
            "no_errors": operation_data.get("success", True),
            "operation_type": operation_data.get("operation_type", "")
        }
    
    def analyze_performance_trends(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over specified timeframe."""
        cutoff_time = (datetime.now(timezone.utc) - 
                      timedelta(hours=timeframe_hours)).isoformat()
        
        trends = {
            "average_duration": 0.0,
            "success_rate": 1.0,
            "optimization_score": 0.5,
            "bottleneck_count": 0,
            "trend_analysis": [],
            "recommendations": []
        }
        
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                # Average metrics
                cursor = conn.execute("""
                    SELECT 
                        AVG(duration_seconds) as avg_duration,
                        AVG(CAST(success AS FLOAT)) as success_rate,
                        AVG(optimization_score) as avg_opt_score,
                        COUNT(*) as total_ops,
                        SUM(CAST(bottleneck_detected AS INT)) as bottleneck_count
                    FROM performance_metrics 
                    WHERE timestamp > ?
                """, (cutoff_time,))
                
                result = cursor.fetchone()
                if result:
                    avg_duration, success_rate, avg_opt_score, total_ops, bottleneck_count = result
                    
                    trends["average_duration"] = avg_duration or 0.0
                    trends["success_rate"] = success_rate or 1.0
                    trends["optimization_score"] = avg_opt_score or 0.5
                    trends["total_operations"] = total_ops or 0
                    trends["bottleneck_count"] = bottleneck_count or 0
                
                # Generate trend analysis
                trends["trend_analysis"] = self._analyze_trends(trends)
                trends["recommendations"] = self._generate_trend_recommendations(trends)
                
        except Exception as e:
            trends["error"] = f"Could not analyze trends: {e}"
        
        return trends
    
    def _analyze_trends(self, trends: Dict[str, Any]) -> List[str]:
        """Analyze performance trends."""
        analysis = []
        
        avg_duration = trends.get("average_duration", 0.0)
        success_rate = trends.get("success_rate", 1.0)
        opt_score = trends.get("optimization_score", 0.5)
        bottleneck_count = trends.get("bottleneck_count", 0)
        
        # Duration analysis
        if avg_duration > 3.0:
            analysis.append(f"⏱️ Average operation duration is high: {avg_duration:.2f}s")
        elif avg_duration < 1.0:
            analysis.append(f"⚡ Operations are performing well: {avg_duration:.2f}s average")
        
        # Success rate analysis
        if success_rate < 0.9:
            analysis.append(f"⚠️ Success rate is below optimal: {success_rate:.1%}")
        else:
            analysis.append(f"✅ Good success rate: {success_rate:.1%}")
        
        # Optimization score analysis
        if opt_score < 0.6:
            analysis.append(f"🎯 Optimization opportunities available: {opt_score:.2f} score")
        else:
            analysis.append(f"🚀 Operations are well optimized: {opt_score:.2f} score")
        
        # Bottleneck analysis
        if bottleneck_count > 0:
            analysis.append(f"🚨 {bottleneck_count} bottlenecks detected")
        
        return analysis
    
    def _generate_trend_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance trends."""
        recommendations = []
        
        avg_duration = trends.get("average_duration", 0.0)
        success_rate = trends.get("success_rate", 1.0)
        opt_score = trends.get("optimization_score", 0.5)
        
        # Duration recommendations
        if avg_duration > 3.0:
            recommendations.append("⚡ Consider implementing parallel processing for better performance")
            recommendations.append("🔄 Review operation batching strategies")
        
        # Success rate recommendations
        if success_rate < 0.9:
            recommendations.append("🛡️ Implement better error handling and retry mechanisms")
            recommendations.append("🔍 Investigate and fix recurring failure patterns")
        
        # Optimization recommendations
        if opt_score < 0.6:
            recommendations.append("🎯 Use Queen ZEN's coordination tools for better efficiency")
            recommendations.append("🧠 Enable neural pattern learning for automatic optimization")
        
        return recommendations


def main():
    """Main performance monitoring execution."""
    try:
        monitor = PerformanceMonitor()
        
        # Read operation data from stdin
        operation_data = {}
        if not sys.stdin.isatty():
            try:
                operation_data = json.loads(sys.stdin.read())
            except json.JSONDecodeError:
                operation_data = {"operation_type": "unknown", "success": False, "error_message": "Invalid JSON input"}
        
        # Record the operation
        result = monitor.record_operation(operation_data)
        
        # Output results
        print(json.dumps(result, indent=2))
        
        # Also analyze recent trends
        trends = monitor.analyze_performance_trends(timeframe_hours=1)
        if trends.get("trend_analysis"):
            print("\n📊 Performance Trends:", file=sys.stderr)
            for analysis in trends["trend_analysis"]:
                print(f"  {analysis}", file=sys.stderr)
        
        if trends.get("recommendations"):
            print("\n💡 Recommendations:", file=sys.stderr)
            for rec in trends["recommendations"]:
                print(f"  {rec}", file=sys.stderr)
        
    except Exception as e:
        error_result = {
            "performance_recorded": False,
            "error": str(e),
            "optimization_score": 0.0,
            "bottleneck_detected": True,
            "recommendations": ["🔧 Performance monitoring encountered an error - check system resources"]
        }
        print(json.dumps(error_result, indent=2))


if __name__ == "__main__":
    main()