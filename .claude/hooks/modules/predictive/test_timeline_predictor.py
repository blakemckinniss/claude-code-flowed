#!/usr/bin/env python3
"""Test script for TimelinePredictor functionality."""

import asyncio
import json
from timeline_predictor import get_timeline_predictor, predict_project_timeline


async def test_timeline_prediction():
    """Test timeline prediction with sample project tasks."""
    
    # Sample project tasks with dependencies
    project_tasks = [
        {
            "id": "task-001",
            "type": "analysis",
            "estimated_duration": 300,  # 5 minutes
            "dependencies": [],
            "metadata": {
                "complexity": 3,
                "priority": 8
            }
        },
        {
            "id": "task-002",
            "type": "validation",
            "estimated_duration": 180,  # 3 minutes
            "dependencies": ["task-001"],
            "metadata": {
                "complexity": 2,
                "priority": 7
            }
        },
        {
            "id": "task-003",
            "type": "optimization",
            "estimated_duration": 600,  # 10 minutes
            "dependencies": ["task-001"],
            "metadata": {
                "complexity": 5,
                "priority": 9
            }
        },
        {
            "id": "task-004",
            "type": "hooks",
            "estimated_duration": 120,  # 2 minutes
            "dependencies": ["task-002", "task-003"],
            "metadata": {
                "complexity": 1,
                "priority": 6
            }
        },
        {
            "id": "task-005",
            "type": "memory",
            "estimated_duration": 240,  # 4 minutes
            "dependencies": ["task-004"],
            "metadata": {
                "complexity": 3,
                "priority": 7
            }
        }
    ]
    
    print("=== Timeline Prediction Test ===\n")
    print(f"Testing with {len(project_tasks)} tasks\n")
    
    # Get predictor
    predictor = get_timeline_predictor()
    
    # Predict timeline
    result = await predict_project_timeline(project_tasks)
    
    print("Timeline Prediction Results:")
    print(json.dumps(result, indent=2))
    
    # Test updating with actual durations
    print("\n=== Updating with Actual Durations ===\n")
    
    # Simulate some tasks completing
    predictor.update_with_actual("task-001", 320)  # Took 20s longer
    predictor.update_with_actual("task-002", 150)  # Finished 30s faster
    
    # Get updated status
    status = predictor.get_timeline_status()
    print("\nPredictor Status:")
    print(json.dumps(status, indent=2))
    
    # Test with tasks without explicit durations (will be predicted)
    print("\n=== Testing Duration Prediction ===\n")
    
    auto_tasks = [
        {
            "id": "auto-001",
            "type": "hooks",
            "dependencies": [],
            "metadata": {
                "complexity": 2,
                "size": 100
            }
        },
        {
            "id": "auto-002",
            "type": "memory",
            "dependencies": ["auto-001"],
            "metadata": {
                "complexity": 4,
                "size": 500
            }
        }
    ]
    
    auto_result = await predict_project_timeline(auto_tasks)
    print("Auto-predicted Timeline:")
    print(json.dumps(auto_result, indent=2))


if __name__ == "__main__":
    asyncio.run(test_timeline_prediction())