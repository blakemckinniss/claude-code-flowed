#!/usr/bin/env python3
"""Performance monitoring dashboard for optimized Claude Code hooks.

Provides real-time metrics and analysis of hook performance, including:
- Hook execution times
- Cache hit rates
- Memory usage
- Pattern detection statistics
- Optimization effectiveness
"""

import json
import sqlite3
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Set up hook paths using centralized path resolver
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

try:
    from modules.optimization import PerformanceMetricsCache
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False


class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""
    
    def __init__(self):
        self.hooks_dir = Path(__file__).parent
        self.cache_dir = self.hooks_dir / "cache"
        self.db_dir = self.hooks_dir / "db"
        self.metrics_cache = None
        
        if OPTIMIZATION_AVAILABLE:
            self.metrics_cache = PerformanceMetricsCache(
                persistence_path=str(self.cache_dir / "dashboard_metrics.json")
            )
    
    def get_hook_execution_stats(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get hook execution statistics for the past time window (seconds)."""
        stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0,
            "max_duration": 0,
            "min_duration": float('inf'),
            "by_hook": {}
        }
        
        # Read from tool history database
        db_path = self.db_dir / "tool_history.db"
        if not db_path.exists():
            return stats
        
        try:
            cutoff_time = datetime.now() - timedelta(seconds=time_window)
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT tool, success, duration
                    FROM tool_history
                    WHERE timestamp > ?
                """, (cutoff_time.isoformat(),))
                
                durations = []
                
                for tool, success, duration in cursor:
                    stats["total_executions"] += 1
                    
                    if success:
                        stats["successful_executions"] += 1
                    else:
                        stats["failed_executions"] += 1
                    
                    if duration:
                        durations.append(duration)
                        stats["max_duration"] = max(stats["max_duration"], duration)
                        stats["min_duration"] = min(stats["min_duration"], duration)
                    
                    # Track by hook type
                    if tool not in stats["by_hook"]:
                        stats["by_hook"][tool] = {
                            "count": 0,
                            "success": 0,
                            "total_duration": 0
                        }
                    
                    stats["by_hook"][tool]["count"] += 1
                    if success:
                        stats["by_hook"][tool]["success"] += 1
                    if duration:
                        stats["by_hook"][tool]["total_duration"] += duration
                
                if durations:
                    stats["average_duration"] = sum(durations) / len(durations)
                    
        except Exception as e:
            print(f"Error reading execution stats: {e}", file=sys.stderr)
        
        if stats["min_duration"] == float('inf'):
            stats["min_duration"] = 0
        
        return stats
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache hit/miss statistics."""
        cache_stats = {
            "validator_cache": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0,
                "size": 0
            },
            "pattern_cache": {
                "patterns_stored": 0,
                "recent_matches": 0
            }
        }
        
        # Read from metrics files
        metrics_files = [
            self.cache_dir / "metrics.json",
            self.cache_dir / "post_metrics.json",
            self.cache_dir / "session_metrics.json"
        ]
        
        for metrics_file in metrics_files:
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        
                        # Extract cache statistics
                        for entry in data:
                            if "cache_hit" in entry:
                                if entry["cache_hit"]:
                                    cache_stats["validator_cache"]["hits"] += 1
                                else:
                                    cache_stats["validator_cache"]["misses"] += 1
                                    
                except Exception as e:
                    print(f"Error reading {metrics_file}: {e}", file=sys.stderr)
        
        # Calculate hit rate
        total_cache_ops = (cache_stats["validator_cache"]["hits"] + 
                          cache_stats["validator_cache"]["misses"])
        if total_cache_ops > 0:
            cache_stats["validator_cache"]["hit_rate"] = (
                cache_stats["validator_cache"]["hits"] / total_cache_ops * 100
            )
        
        return cache_stats
    
    def get_optimization_effectiveness(self) -> Dict[str, Any]:
        """Calculate optimization effectiveness metrics."""
        effectiveness = {
            "speed_improvement": 0,
            "memory_efficiency": 0,
            "parallel_utilization": 0,
            "error_reduction": 0
        }
        
        # Compare optimized vs non-optimized performance
        if self.metrics_cache:
            recent_metrics = self.metrics_cache._metrics[-100:]  # Last 100 operations
            
            if recent_metrics:
                # Calculate average performance improvements
                optimized_durations = []
                baseline_durations = []
                
                for metric in recent_metrics:
                    if metric.get("optimized"):
                        optimized_durations.append(metric.get("duration", 0))
                    else:
                        baseline_durations.append(metric.get("duration", 0))
                
                if optimized_durations and baseline_durations:
                    avg_optimized = sum(optimized_durations) / len(optimized_durations)
                    avg_baseline = sum(baseline_durations) / len(baseline_durations)
                    
                    if avg_baseline > 0:
                        effectiveness["speed_improvement"] = (
                            (avg_baseline - avg_optimized) / avg_baseline * 100
                        )
        
        return effectiveness
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        memory_stats = {
            "current_usage_mb": 0,
            "peak_usage_mb": 0,
            "object_pool_efficiency": 0,
            "pattern_storage_usage": 0
        }
        
        # Read system metrics
        system_metrics_file = Path("/home/devcontainers/flowed/.claude-flow/metrics/system-metrics.json")
        if system_metrics_file.exists():
            try:
                with open(system_metrics_file, 'r') as f:
                    metrics = json.load(f)
                    
                    if metrics:
                        latest = metrics[-1]
                        memory_stats["current_usage_mb"] = latest.get("memoryUsed", 0) / (1024 * 1024)
                        
                        # Find peak usage
                        peak_usage = max(m.get("memoryUsed", 0) for m in metrics)
                        memory_stats["peak_usage_mb"] = peak_usage / (1024 * 1024)
                        
            except Exception as e:
                print(f"Error reading system metrics: {e}", file=sys.stderr)
        
        return memory_stats
    
    def get_pattern_detection_stats(self) -> Dict[str, Any]:
        """Get pattern detection and learning statistics."""
        pattern_stats = {
            "total_patterns": 0,
            "workflow_drifts_detected": 0,
            "successful_corrections": 0,
            "pattern_types": {}
        }
        
        # Read from pattern storage
        db_path = self.db_dir / "hooks.db"
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    # Count patterns by type
                    cursor = conn.execute("""
                        SELECT pattern_type, COUNT(*) as count
                        FROM patterns
                        GROUP BY pattern_type
                    """)
                    
                    for pattern_type, count in cursor:
                        pattern_stats["pattern_types"][pattern_type] = count
                        pattern_stats["total_patterns"] += count
                        
            except Exception:
                pass  # Table might not exist yet
        
        return pattern_stats
    
    def generate_report(self, format: str = "text") -> str:
        """Generate a performance report."""
        # Collect all statistics
        execution_stats = self.get_hook_execution_stats()
        cache_stats = self.get_cache_statistics()
        optimization_stats = self.get_optimization_effectiveness()
        memory_stats = self.get_memory_usage_stats()
        pattern_stats = self.get_pattern_detection_stats()
        
        if format == "json":
            return json.dumps({
                "timestamp": datetime.now().isoformat(),
                "execution": execution_stats,
                "cache": cache_stats,
                "optimization": optimization_stats,
                "memory": memory_stats,
                "patterns": pattern_stats
            }, indent=2)
        
        # Generate text report
        report = []
        report.append("=" * 60)
        report.append("Claude Code Hooks Performance Dashboard")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # Execution Statistics
        report.append("\n📊 EXECUTION STATISTICS (Last Hour)")
        report.append(f"Total Executions: {execution_stats['total_executions']}")
        report.append(f"Success Rate: {execution_stats['successful_executions'] / max(1, execution_stats['total_executions']) * 100:.1f}%")
        report.append(f"Average Duration: {execution_stats['average_duration']:.3f}s")
        report.append(f"Min/Max Duration: {execution_stats['min_duration']:.3f}s / {execution_stats['max_duration']:.3f}s")
        
        if execution_stats['by_hook']:
            report.append("\nBy Hook Type:")
            for hook, stats in execution_stats['by_hook'].items():
                avg_duration = stats['total_duration'] / max(1, stats['count'])
                report.append(f"  {hook}: {stats['count']} calls, {avg_duration:.3f}s avg")
        
        # Cache Performance
        report.append("\n💾 CACHE PERFORMANCE")
        report.append(f"Cache Hit Rate: {cache_stats['validator_cache']['hit_rate']:.1f}%")
        report.append(f"Total Cache Hits: {cache_stats['validator_cache']['hits']}")
        report.append(f"Total Cache Misses: {cache_stats['validator_cache']['misses']}")
        
        # Optimization Effectiveness
        report.append("\n⚡ OPTIMIZATION EFFECTIVENESS")
        report.append(f"Speed Improvement: {optimization_stats['speed_improvement']:.1f}%")
        report.append(f"Memory Efficiency: {memory_stats.get('object_pool_efficiency', 0):.1f}%")
        
        # Memory Usage
        report.append("\n🧠 MEMORY USAGE")
        report.append(f"Current Usage: {memory_stats['current_usage_mb']:.1f} MB")
        report.append(f"Peak Usage: {memory_stats['peak_usage_mb']:.1f} MB")
        
        # Pattern Detection
        report.append("\n🔍 PATTERN DETECTION")
        report.append(f"Total Patterns Learned: {pattern_stats['total_patterns']}")
        report.append(f"Workflow Drifts Detected: {pattern_stats['workflow_drifts_detected']}")
        
        if pattern_stats['pattern_types']:
            report.append("\nPattern Types:")
            for ptype, count in pattern_stats['pattern_types'].items():
                report.append(f"  {ptype}: {count}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def monitor_real_time(self, interval: int = 5):
        """Monitor performance in real-time."""
        print("Starting real-time performance monitoring...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Clear screen (works on Unix-like systems)
                print("\033[2J\033[H")
                
                # Generate and display report
                report = self.generate_report()
                print(report)
                
                # Wait for next update
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


def main():
    """Main entry point for the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude Code Hooks Performance Dashboard")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="Output format")
    parser.add_argument("--monitor", action="store_true",
                       help="Enable real-time monitoring")
    parser.add_argument("--interval", type=int, default=5,
                       help="Monitoring update interval in seconds")
    
    args = parser.parse_args()
    
    dashboard = PerformanceDashboard()
    
    if args.monitor:
        dashboard.monitor_real_time(args.interval)
    else:
        report = dashboard.generate_report(args.format)
        print(report)


if __name__ == "__main__":
    main()