import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

class ProactiveBottleneckPredictor:
    """
    Predictive intelligence module for anticipating resource bottlenecks.
    
    Key Capabilities:
    - 5-minute ahead bottleneck forecasting
    - Resource allocation recommendations
    - Performance constraint optimization
    """
    
    def __init__(self, performance_monitor, ml_optimizer):
        """
        Initialize the predictor with performance monitoring and ML optimization capabilities.
        
        Args:
            performance_monitor: Existing performance monitoring infrastructure
            ml_optimizer: MLOptimizedExecutor for workload classification
        """
        self.performance_monitor = performance_monitor
        self.ml_optimizer = ml_optimizer
        
        # Forecasting parameters
        self.forecast_window = 5  # minutes
        self.resource_thresholds = {
            'cpu': 0.80,  # 80% CPU max
            'memory': 0.85,  # 85% memory max
        }
    
    def collect_historical_metrics(self, window_minutes: int = 30) -> pd.DataFrame:
        """
        Collect historical performance metrics for predictive analysis.
        
        Args:
            window_minutes: Historical data window (default 30 minutes)
        
        Returns:
            DataFrame with historical performance metrics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=window_minutes)
        
        metrics = self.performance_monitor.get_metrics(
            start_time=start_time, 
            end_time=end_time
        )
        
        return pd.DataFrame(metrics)
    
    def predict_bottlenecks(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Predict potential bottlenecks using time-series forecasting.
        
        Args:
            historical_data: DataFrame with historical performance metrics
        
        Returns:
            Dictionary of predicted resource utilization
        """
        # Time-series forecasting using exponential weighted moving average
        predictions = {}
        
        for resource in ['cpu', 'memory']:
            # Predict next 5-minute utilization
            ewm = historical_data[resource].ewm(span=5).mean()
            prediction = ewm.iloc[-1]
            predictions[resource] = prediction
        
        return predictions
    
    def recommend_resource_allocation(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate proactive resource allocation recommendations.
        
        Args:
            predictions: Predicted resource utilization
        
        Returns:
            Recommended resource allocation strategies
        """
        recommendations = {}
        
        for resource, utilization in predictions.items():
            if utilization > self.resource_thresholds[resource]:
                # Recommend scaling or load redistribution
                recommendations[resource] = {
                    'action': 'scale',
                    'severity': 'high' if utilization > 0.95 else 'medium',
                    'suggested_allocation': min(utilization * 1.2, 1.0)  # 20% buffer
                }
            else:
                recommendations[resource] = {
                    'action': 'maintain',
                    'current_utilization': utilization
                }
        
        return recommendations
    
    def run_predictive_analysis(self) -> Dict[str, Any]:
        """
        Execute complete predictive bottleneck analysis.
        
        Returns:
            Comprehensive bottleneck prediction report
        """
        # Collect historical metrics
        historical_data = self.collect_historical_metrics()
        
        # Predict bottlenecks
        predictions = self.predict_bottlenecks(historical_data)
        
        # Generate recommendations
        recommendations = self.recommend_resource_allocation(predictions)
        
        return {
            'timestamp': datetime.now(),
            'predictions': predictions,
            'recommendations': recommendations,
            'workload_classification': self.ml_optimizer.classify_current_workload()
        }

def initialize_proactive_bottleneck_predictor(performance_monitor, ml_optimizer):
    """
    Initialize and configure the ProactiveBottleneckPredictor.
    
    Args:
        performance_monitor: Performance monitoring system
        ml_optimizer: Workload classification system
    
    Returns:
        Configured ProactiveBottleneckPredictor instance
    """
    return ProactiveBottleneckPredictor(performance_monitor, ml_optimizer)