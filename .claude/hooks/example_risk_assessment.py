#!/usr/bin/env python3
"""Example usage of the Risk Assessment Engine.

This script demonstrates how to integrate and use the Risk Assessment Engine
for failure prediction and proactive risk management.
"""

import sys
import os
import time
import logging
from datetime import datetime, timezone

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from modules.optimization.circuit_breaker import CircuitBreakerManager
from modules.optimization.performance_monitor import PerformanceMonitor
from modules.predictive.risk_assessment_engine import create_risk_assessment_engine, RiskLevel


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_alert_system(risk_engine):
    """Create alert system for risk notifications."""
    
    def risk_alert_handler(assessment):
        """Handle risk alerts based on severity."""
        timestamp = assessment.timestamp.strftime('%H:%M:%S')
        risk_score = assessment.overall_risk_score
        failure_prob = assessment.failure_probability
        
        if assessment.overall_risk_level == RiskLevel.CRITICAL:
            print(f"\n🚨 CRITICAL ALERT [{timestamp}]")
            print(f"   Risk Score: {risk_score:.3f}")
            print(f"   Failure Probability: {failure_prob:.3f}")
            print("   Immediate action required!")
            
            # Show top mitigation strategies
            for strategy in assessment.mitigation_strategies[:2]:
                print(f"   📋 {strategy.get('action', 'unknown')}: {strategy.get('description', 'N/A')}")
        
        elif assessment.overall_risk_level == RiskLevel.HIGH:
            print(f"\n⚠️  HIGH RISK [{timestamp}] - Score: {risk_score:.3f}, "
                  f"Failure Prob: {failure_prob:.3f}")
            
        elif assessment.overall_risk_level == RiskLevel.MEDIUM:
            print(f"\n💛 MEDIUM RISK [{timestamp}] - Score: {risk_score:.3f}")
    
    risk_engine.add_alert_callback(risk_alert_handler)


def simulate_system_stress(circuit_manager):
    """Simulate system stress to trigger risk assessment."""
    print("\n🔧 Simulating system stress to demonstrate risk detection...")
    
    # Get or create a test circuit breaker
    test_breaker = circuit_manager.get_breaker("demo_service")
    
    # Simulate increasing failure rates
    for i in range(5):
        # Simulate failed operations
        test_breaker.stats.failed_calls += 1
        test_breaker.stats.total_calls += 1
        test_breaker.stats.consecutive_failures += 1
        
        print(f"   Injected failure {i+1}/5 (Total failures: {test_breaker.stats.failed_calls})")
        time.sleep(2)
    
    print("   System stress simulation complete")


def demonstrate_risk_context(risk_engine):
    """Demonstrate risk-aware operation context."""
    print("\n🎯 Demonstrating risk-aware operations...")
    
    # Example 1: Successful operation
    try:
        with risk_engine.risk_context("demo_successful_operation") as initial_risk:
            print(f"   Starting operation with {initial_risk.overall_risk_level.value} risk")
            time.sleep(1)  # Simulate work
            print("   ✅ Operation completed successfully")
    except Exception as e:
        print(f"   ❌ Operation failed: {e}")
    
    # Example 2: Failed operation (for training)
    try:
        with risk_engine.risk_context("demo_failed_operation") as initial_risk:
            print(f"   Starting risky operation with {initial_risk.overall_risk_level.value} risk")
            time.sleep(0.5)
            raise Exception("Simulated operation failure")
    except Exception as e:
        print(f"   ❌ Expected failure for training: {e}")


def show_detailed_assessment(risk_engine):
    """Show detailed risk assessment information."""
    print("\n📊 Detailed Risk Assessment:")
    
    assessment = risk_engine.assess_current_risk()
    
    print(f"   Overall Risk: {assessment.overall_risk_level.value} "
          f"(Score: {assessment.overall_risk_score:.3f})")
    print(f"   Failure Probability: {assessment.failure_probability:.3f}")
    print(f"   Model Confidence: {assessment.confidence:.3f}")
    
    print("\n   Risk Factors:")
    for factor in assessment.factors:
        trend_indicator = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(factor.trend, "❓")
        print(f"     {factor.name}: {factor.current_value:.3f} "
              f"({factor.risk_level.value}) {trend_indicator}")
    
    if assessment.mitigation_strategies:
        print("\n   Recommended Mitigations:")
        for i, strategy in enumerate(assessment.mitigation_strategies[:3], 1):
            priority_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                strategy.get("priority", "low"), "⚪"
            )
            print(f"     {i}. {priority_icon} {strategy.get('action', 'unknown')}")
            print(f"        {strategy.get('description', 'N/A')}")
    
    print("\n   Failure Predictions:")
    for window in [5, 15, 60]:
        prediction = risk_engine.predict_failure_in_window(window)
        prob = prediction['failure_probability']
        conf = prediction['confidence']
        print(f"     {window:2d} minutes: {prob:.3f} (confidence: {conf:.3f})")


def monitor_trends(risk_engine, duration_seconds=60):
    """Monitor risk trends over time."""
    print(f"\n📈 Monitoring risk trends for {duration_seconds} seconds...")
    
    start_time = time.time()
    assessments = []
    
    while time.time() - start_time < duration_seconds:
        assessment = risk_engine.assess_current_risk()
        assessments.append(assessment)
        
        # Show brief status every 10 seconds
        if len(assessments) % 5 == 0:  # Roughly every 10 seconds with 2-second intervals
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"   [{current_time}] Risk: {assessment.overall_risk_level.value} "
                  f"(Score: {assessment.overall_risk_score:.3f})")
        
        time.sleep(2)
    
    # Show trend summary
    if len(assessments) >= 2:
        initial_score = assessments[0].overall_risk_score
        final_score = assessments[-1].overall_risk_score
        trend = "increasing" if final_score > initial_score else "decreasing" if final_score < initial_score else "stable"
        
        print("\n   Trend Summary:")
        print(f"     Initial Score: {initial_score:.3f}")
        print(f"     Final Score: {final_score:.3f}")
        print(f"     Trend: {trend}")


def main():
    """Main demonstration function."""
    print("🚀 Risk Assessment Engine - Live Demonstration")
    print("=" * 60)
    
    setup_logging()
    
    # Create infrastructure components
    print("🔧 Setting up infrastructure...")
    circuit_manager = CircuitBreakerManager()
    perf_monitor = PerformanceMonitor()
    
    # Create risk assessment engine
    risk_engine = create_risk_assessment_engine(circuit_manager, perf_monitor)
    
    # Setup alert system
    create_alert_system(risk_engine)
    
    try:
        # Start continuous monitoring
        print("🔄 Starting continuous risk monitoring...")
        risk_engine.start_monitoring()
        
        # Initial assessment
        show_detailed_assessment(risk_engine)
        
        # Demonstrate risk-aware operations
        demonstrate_risk_context(risk_engine)
        
        # Simulate system stress
        simulate_system_stress(circuit_manager)
        
        # Wait for risk engine to detect the increased risk
        print("\n⏳ Waiting for risk engine to detect changes...")
        time.sleep(35)  # Wait for assessment cycle
        
        # Show updated assessment
        show_detailed_assessment(risk_engine)
        
        # Monitor trends briefly
        monitor_trends(risk_engine, duration_seconds=30)
        
        # Show final status
        print("\n📊 Final Engine Status:")
        status = risk_engine.get_status()
        print(f"   Assessments Performed: {status['assessment_count']}")
        print(f"   Model Confidence: {status['model_confidence']:.3f}")
        print(f"   Alert Callbacks: {status['alert_callbacks']}")
        
        print("\n🎉 Demonstration Complete!")
        print("✅ Risk Assessment Engine successfully demonstrated:")
        print("   • Real-time risk scoring")
        print("   • Failure probability prediction")
        print("   • Automated mitigation recommendations")
        print("   • Integration with existing infrastructure")
        print("   • Risk-aware operation context")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Demonstration error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🛑 Shutting down...")
        risk_engine.stop_monitoring()
        perf_monitor.shutdown()
        print("   All components stopped successfully")


if __name__ == "__main__":
    main()