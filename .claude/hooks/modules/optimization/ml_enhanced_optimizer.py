"""ML-Enhanced Hook Optimizer Integration.

Extends the existing IntegratedHookOptimizer with adaptive learning capabilities
while maintaining system constraints and existing performance monitoring.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from .integrated_optimizer import IntegratedHookOptimizer, AdaptiveOptimizer
from .adaptive_learning_engine import AdaptiveLearningEngine, get_adaptive_learning_engine
from .performance_monitor import get_performance_monitor


class MLEnhancedAdaptiveOptimizer(AdaptiveOptimizer):
    """Enhanced adaptive optimizer with ML capabilities."""
    
    def __init__(self, monitor, learning_engine: AdaptiveLearningEngine):
        super().__init__(monitor)
        self.learning_engine = learning_engine
        self.ml_predictions_cache = {}
        self.prediction_accuracy_history = []
        
        # Enhanced profiles with ML-driven parameters
        self.ml_profiles = {
            "neural_latency": {
                "description": "ML-optimized for ultra-low latency",
                "async_workers": "adaptive_high",
                "batch_size": "ml_optimized_small",
                "cache_aggressive": True,
                "circuit_breaker_sensitive": True,
                "ml_prediction_weight": 0.8,
                "neural_scaling": True
            },
            "neural_throughput": {
                "description": "ML-optimized for maximum throughput",
                "async_workers": "adaptive_max",
                "batch_size": "ml_optimized_large",
                "cache_aggressive": False,
                "circuit_breaker_sensitive": False,
                "ml_prediction_weight": 0.7,
                "neural_scaling": True
            },
            "adaptive_learning": {
                "description": "Continuous learning and adaptation",
                "async_workers": "ml_adaptive",
                "batch_size": "dynamic_learning",
                "cache_aggressive": True,
                "circuit_breaker_sensitive": True,
                "ml_prediction_weight": 0.9,
                "neural_scaling": True,
                "continuous_learning": True
            }
        }
        
        # Merge with existing profiles
        self.profiles.update(self.ml_profiles)
    
    def analyze_performance_with_ml(self) -> Dict[str, Any]:
        """Enhanced performance analysis with ML predictions."""
        # Get base analysis
        base_analysis = self.analyze_performance()
        
        # Get ML learning status
        learning_status = self.learning_engine.get_learning_status()
        
        # Extract ML-specific metrics
        ml_metrics = {
            'pattern_confidence': self._calculate_pattern_confidence(learning_status),
            'prediction_accuracy': self._calculate_prediction_accuracy(),
            'learning_efficiency': self._calculate_learning_efficiency(learning_status),
            'system_adaptation_score': self._calculate_adaptation_score(base_analysis, learning_status)
        }
        
        # Enhanced recommendation with ML input
        enhanced_recommendation = self._ml_enhanced_recommendation(base_analysis, ml_metrics)
        
        return {
            **base_analysis,
            'ml_metrics': ml_metrics,
            'enhanced_recommendation': enhanced_recommendation,
            'learning_status': learning_status
        }
    
    def _calculate_pattern_confidence(self, learning_status: Dict[str, Any]) -> float:
        """Calculate overall confidence in learned patterns."""
        total_patterns = learning_status.get('total_patterns', 0)
        high_confidence_patterns = learning_status.get('high_confidence_patterns', 0)
        
        if total_patterns == 0:
            return 0.0
        
        return high_confidence_patterns / total_patterns
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate ML prediction accuracy."""
        if not self.prediction_accuracy_history:
            return 0.5  # Default neutral accuracy
        
        recent_accuracy = self.prediction_accuracy_history[-10:]  # Last 10 predictions
        return sum(recent_accuracy) / len(recent_accuracy)
    
    def _calculate_learning_efficiency(self, learning_status: Dict[str, Any]) -> float:
        """Calculate how efficiently the system is learning."""
        predictor_metrics = learning_status.get('performance_predictor', {})
        training_loss = predictor_metrics.get('training_loss', 1.0)
        
        # Lower loss = higher efficiency
        return max(0.0, 1.0 - min(1.0, training_loss))
    
    def _calculate_adaptation_score(self, base_analysis: Dict[str, Any], learning_status: Dict[str, Any]) -> float:
        """Calculate how well the system is adapting to workload."""
        # Combine multiple factors
        pattern_usage = len(learning_status.get('optimization_history_size', 0)) / 1000.0
        system_stability = 1.0 - base_analysis.get('error_rate', 0)
        resource_efficiency = 1.0 - max(
            base_analysis.get('cpu_usage', 0) / 100.0,
            base_analysis.get('memory_usage', 0) / 100.0
        )
        
        return (pattern_usage * 0.3 + system_stability * 0.4 + resource_efficiency * 0.3)
    
    def _ml_enhanced_recommendation(self, base_analysis: Dict[str, Any], ml_metrics: Dict[str, Any]) -> str:
        """Enhanced profile recommendation using ML insights."""
        # Start with base recommendation
        base_recommendation = base_analysis.get('recommended_profile', 'balanced')
        
        # ML enhancement factors
        pattern_confidence = ml_metrics['pattern_confidence']
        learning_efficiency = ml_metrics['learning_efficiency']
        adaptation_score = ml_metrics['system_adaptation_score']
        
        # High confidence in patterns + good learning = use adaptive learning
        if pattern_confidence > 0.7 and learning_efficiency > 0.6:
            return 'adaptive_learning'
        
        # Good adaptation but need speed = neural latency
        if adaptation_score > 0.8 and base_analysis.get('avg_latency_ms', 0) > 100:
            return 'neural_latency'
        
        # High resource efficiency = can use neural throughput
        if base_analysis.get('cpu_usage', 0) < 40 and base_analysis.get('memory_usage', 0) < 50:
            return 'neural_throughput'
        
        # Fallback to base recommendation with ML prefix if applicable
        if base_recommendation in ['latency', 'throughput'] and pattern_confidence > 0.5:
            return f'neural_{base_recommendation}'
        
        return base_recommendation


class MLEnhancedHookOptimizer(IntegratedHookOptimizer):
    """Enhanced hook optimizer with adaptive learning capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize base optimizer
        super().__init__(config_path)
        
        # Initialize ML components
        self.learning_engine = get_adaptive_learning_engine()
        
        # Replace adaptive optimizer with ML-enhanced version
        self.adaptive_optimizer = MLEnhancedAdaptiveOptimizer(
            self.monitor, self.learning_engine
        )
        
        # ML-specific configuration
        self.ml_config = {
            'enable_neural_prediction': True,
            'prediction_threshold': 0.7,
            'learning_rate_adjustment': True,
            'adaptive_batch_sizing': True,
            'real_time_optimization': True
        }
        
        # Performance tracking
        self.ml_performance_history = []
        self.optimization_effectiveness = []
    
    async def execute_hook_ml_optimized(self,
                                      hook_path: str,
                                      hook_data: Dict[str, Any],
                                      priority: Optional[str] = None) -> Dict[str, Any]:
        """Execute hook with ML-enhanced optimizations."""
        
        start_time = time.perf_counter()
        
        # Create enhanced task context
        task_context = {
            **hook_data,
            'hook_path': hook_path,
            'priority': priority or 'normal',
            'estimated_cpu_time': self._estimate_cpu_time(hook_path, hook_data),
            'estimated_io_time': self._estimate_io_time(hook_path, hook_data),
            'memory_requirement_mb': self._estimate_memory_requirement(hook_path, hook_data),
            'complexity_score': self._calculate_complexity_score(hook_data)
        }
        
        # Get ML performance prediction
        if self.ml_config['enable_neural_prediction']:
            prediction = await self._get_ml_performance_prediction(task_context)
            task_context.update(prediction)
        
        # Execute with adaptive learning optimization
        async def optimized_hook_task():
            return await super().execute_hook_optimized(hook_path, hook_data)
        
        result = await self.learning_engine.optimize_task_execution(
            optimized_hook_task, task_context
        )
        
        # Record performance metrics
        execution_time = time.perf_counter() - start_time
        self._record_ml_performance(task_context, result, execution_time)
        
        return result
    
    def _estimate_cpu_time(self, hook_path: str, hook_data: Dict[str, Any]) -> float:
        """Estimate CPU time requirement for hook."""
        # Basic heuristics - could be enhanced with historical data
        base_time = 10.0  # Base 10ms
        
        # Adjust based on hook type
        if 'validation' in hook_path.lower():
            base_time *= 2
        if 'complex' in str(hook_data):
            base_time *= 3
        
        return base_time
    
    def _estimate_io_time(self, hook_path: str, hook_data: Dict[str, Any]) -> float:
        """Estimate I/O time requirement for hook."""
        base_time = 5.0  # Base 5ms
        
        # Check for I/O operations
        if any(key in str(hook_data).lower() for key in ['file', 'read', 'write', 'network']):
            base_time *= 10
        
        return base_time
    
    def _estimate_memory_requirement(self, hook_path: str, hook_data: Dict[str, Any]) -> float:
        """Estimate memory requirement for hook."""
        base_memory = 10.0  # Base 10MB
        
        # Adjust based on data size
        data_size = len(str(hook_data))
        if data_size > 1000:
            base_memory *= (data_size / 1000)
        
        return min(base_memory, 500.0)  # Cap at 500MB
    
    def _calculate_complexity_score(self, hook_data: Dict[str, Any]) -> int:
        """Calculate complexity score for hook data."""
        score = 1
        
        # Check nesting depth
        def get_depth(obj, depth=0):
            if isinstance(obj, dict):
                return max(get_depth(v, depth + 1) for v in obj.values()) if obj else depth
            elif isinstance(obj, list):
                return max(get_depth(item, depth + 1) for item in obj) if obj else depth
            return depth
        
        depth = get_depth(hook_data)
        score += min(depth, 10)
        
        # Check data volume
        data_volume = len(str(hook_data))
        if data_volume > 1000:
            score += min(data_volume // 1000, 5)
        
        return min(score, 10)  # Cap at 10
    
    async def _get_ml_performance_prediction(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML-based performance predictions."""
        # Extract features for prediction
        features = self.learning_engine.pattern_learner.extract_features(task_context)
        
        # Get prediction from performance predictor
        prediction = self.learning_engine.performance_predictor.predict(
            features.reshape(1, -1)
        )[0]
        
        return {
            'predicted_latency_ms': prediction[0] * 1000,
            'predicted_throughput': prediction[1] * 100,
            'predicted_cpu_usage': prediction[2] * 100,
            'predicted_memory_usage': prediction[3] * 100,
            'ml_confidence': self._calculate_prediction_confidence(features)
        }
    
    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in ML prediction."""
        # Simple confidence based on feature similarity to training data
        # In production, this would use more sophisticated uncertainty estimation
        return 0.8  # Placeholder - 80% confidence
    
    def _record_ml_performance(self, task_context: Dict[str, Any], 
                              result: Dict[str, Any], execution_time: float):
        """Record ML performance for learning."""
        performance_record = {
            'timestamp': time.time(),
            'execution_time': execution_time,
            'success': result.get('success', False),
            'predicted_latency': task_context.get('predicted_latency_ms', 0),
            'actual_latency': execution_time * 1000,
            'prediction_error': abs(
                task_context.get('predicted_latency_ms', 0) - execution_time * 1000
            ),
            'task_complexity': task_context.get('complexity_score', 1)
        }
        
        self.ml_performance_history.append(performance_record)
        
        # Calculate prediction accuracy
        if task_context.get('predicted_latency_ms', 0) > 0:
            accuracy = 1.0 - min(1.0, performance_record['prediction_error'] / 
                               task_context['predicted_latency_ms'])
            self.adaptive_optimizer.prediction_accuracy_history.append(accuracy)
    
    def get_ml_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive ML optimization status."""
        base_status = super().get_optimization_status()
        
        # Add ML-specific metrics
        ml_status = {
            'learning_engine_status': self.learning_engine.get_learning_status(),
            'ml_performance_history_size': len(self.ml_performance_history),
            'recent_prediction_accuracy': self._get_recent_prediction_accuracy(),
            'optimization_effectiveness': self._get_optimization_effectiveness(),
            'ml_config': self.ml_config,
            'system_constraints_status': self._check_system_constraints()
        }
        
        return {
            **base_status,
            'ml_enhanced': ml_status
        }
    
    def _get_recent_prediction_accuracy(self) -> Dict[str, float]:
        """Get recent prediction accuracy statistics."""
        if not self.adaptive_optimizer.prediction_accuracy_history:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
        
        recent = self.adaptive_optimizer.prediction_accuracy_history[-20:]  # Last 20
        
        return {
            'mean': sum(recent) / len(recent),
            'min': min(recent),
            'max': max(recent),
            'count': len(recent)
        }
    
    def _get_optimization_effectiveness(self) -> Dict[str, float]:
        """Calculate optimization effectiveness."""
        if len(self.ml_performance_history) < 2:
            return {'improvement': 0.0, 'consistency': 0.0}
        
        recent_performance = self.ml_performance_history[-10:]
        baseline_performance = self.ml_performance_history[:10]
        
        if not baseline_performance:
            return {'improvement': 0.0, 'consistency': 0.0}
        
        recent_avg = sum(p['execution_time'] for p in recent_performance) / len(recent_performance)
        baseline_avg = sum(p['execution_time'] for p in baseline_performance) / len(baseline_performance)
        
        improvement = max(0.0, (baseline_avg - recent_avg) / baseline_avg)
        
        # Calculate consistency (lower variance = higher consistency)
        recent_variance = np.var([p['execution_time'] for p in recent_performance])
        consistency = max(0.0, 1.0 - min(1.0, recent_variance))
        
        return {
            'improvement': improvement,
            'consistency': consistency
        }
    
    def _check_system_constraints(self) -> Dict[str, Any]:
        """Check if system is operating within ML optimization constraints."""
        current_status = self.learning_engine.get_learning_status()
        current_perf = current_status['current_system_performance']
        constraints = self.learning_engine.system_constraints
        
        cpu_usage = current_perf.get('cpu_percent', 0) / 100.0
        memory_usage = current_perf.get('memory_percent', 0) / 100.0
        memory_efficiency = (100 - current_perf.get('memory_percent', 0)) / 100.0
        
        return {
            'cpu_within_limits': cpu_usage <= constraints['max_cpu_utilization'],
            'memory_within_limits': memory_usage <= constraints['max_memory_utilization'],
            'memory_efficiency_target': memory_efficiency >= constraints['target_memory_efficiency'],
            'current_cpu_usage': cpu_usage,
            'current_memory_usage': memory_usage,
            'current_memory_efficiency': memory_efficiency,
            'constraints': constraints
        }
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        print("🚀 Starting ML-Enhanced Optimization Benchmark...")
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Light Validation',
                'hook_path': 'validation/light.py',
                'hook_data': {'simple': True, 'data_size': 'small'}
            },
            {
                'name': 'Complex Validation',
                'hook_path': 'validation/complex.py',
                'hook_data': {'complex': True, 'nested': {'deep': {'structure': True}}}
            },
            {
                'name': 'Heavy Processing',
                'hook_path': 'processing/heavy.py',
                'hook_data': {'process_intensive': True, 'data_volume': 'large'}
            }
        ]
        
        benchmark_results = []
        
        for scenario in test_scenarios:
            print(f"   Testing {scenario['name']}...")
            
            # Run multiple iterations
            iteration_times = []
            for i in range(5):
                start_time = time.perf_counter()
                
                try:
                    result = await self.execute_hook_ml_optimized(
                        scenario['hook_path'],
                        scenario['hook_data']
                    )
                    result.get('success', True)  # Assume success if not specified
                except Exception as e:
                    print(f"      Error in iteration {i+1}: {e}")
                
                execution_time = time.perf_counter() - start_time
                iteration_times.append(execution_time)
            
            # Calculate statistics
            avg_time = sum(iteration_times) / len(iteration_times)
            min_time = min(iteration_times)
            max_time = max(iteration_times)
            
            benchmark_results.append({
                'scenario': scenario['name'],
                'avg_time_ms': avg_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'consistency': 1.0 - (max_time - min_time) / avg_time if avg_time > 0 else 0,
                'iterations': len(iteration_times)
            })
            
            print(f"      ✅ Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, Max: {max_time*1000:.2f}ms")
        
        # Get comprehensive status
        status = self.get_ml_optimization_status()
        
        return {
            'benchmark_results': benchmark_results,
            'system_status': status,
            'timestamp': time.time()
        }
    
    async def shutdown(self):
        """Shutdown ML-enhanced optimizer."""
        # Shutdown learning engine
        self.learning_engine.shutdown()
        
        # Shutdown base optimizer
        await super().shutdown()


# Global ML-enhanced optimizer instance
_global_ml_optimizer: Optional[MLEnhancedHookOptimizer] = None


async def get_ml_enhanced_optimizer() -> MLEnhancedHookOptimizer:
    """Get or create global ML-enhanced optimizer."""
    global _global_ml_optimizer
    
    if _global_ml_optimizer is None:
        _global_ml_optimizer = MLEnhancedHookOptimizer()
        await _global_ml_optimizer.initialize_async_components()
    
    return _global_ml_optimizer