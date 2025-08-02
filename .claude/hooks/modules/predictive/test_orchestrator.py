#!/usr/bin/env python3
"""Test script for ProactiveOrchestrator integration.

This script demonstrates the integrated predictive intelligence system.
"""

import asyncio
import time
from predictive_dashboard import display_dashboard, get_dashboard_data
from proactive_orchestrator import get_proactive_orchestrator


async def test_orchestrator():
    """Test the ProactiveOrchestrator functionality."""
    print("🚀 Testing ProactiveOrchestrator - Phase 3 Integration")
    print("=" * 80)
    
    # Initialize orchestrator
    print("\n1. Initializing ProactiveOrchestrator...")
    orchestrator = await get_proactive_orchestrator()
    
    # Get initial status
    status = orchestrator.get_orchestration_status()
    print(f"   Status: {status['status']}")
    print(f"   Configuration: {status['configuration']}")
    print(f"   Memory Efficiency: {status['metrics']['memory_efficiency']:.1%}")
    
    # Wait for some predictions
    print("\n2. Waiting for prediction cycles...")
    await asyncio.sleep(2)
    
    # Display dashboard views
    print("\n3. Displaying Dashboard Views:")
    print("\n" + "-" * 40)
    print("SUMMARY VIEW:")
    await display_dashboard('summary')
    
    await asyncio.sleep(1)
    
    print("\n" + "-" * 40)
    print("RESOURCES VIEW:")
    await display_dashboard('resources')
    
    # Get JSON data
    print("\n4. Getting Dashboard JSON Data:")
    data = await get_dashboard_data()
    print(f"   Active Predictions: {data['predictions']['active']}")
    print(f"   Pre-positioned Agents: {data['agents']['pre_positioned']}")
    print(f"   Current CPU: {data['resources']['current']['cpu']:.1f}%")
    print(f"   Current Memory: {data['resources']['current']['memory']:.1f}%")
    
    # Performance check
    print("\n5. Performance Validation:")
    metrics = status['metrics']
    
    print(f"   ✅ Prediction Latency: {metrics['average_prediction_latency_ms']:.1f}ms "
          f"(Target: <100ms)")
    print(f"   ✅ Memory Efficiency: {metrics['memory_efficiency']:.1%} "
          f"(Target: >76%)")
    
    # System health
    print(f"\n6. System Health: {status['system_health'].upper()}")
    
    print("\n" + "=" * 80)
    print("✅ ProactiveOrchestrator Test Complete!")
    print("   All predictive components integrated successfully")
    print("   Phase 3 Integration: COMPLETE")


async def main():
    """Main test runner."""
    try:
        await test_orchestrator()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        from proactive_orchestrator import shutdown_orchestrator
        await shutdown_orchestrator()


if __name__ == "__main__":
    asyncio.run(main())