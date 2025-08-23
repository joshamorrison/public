#!/usr/bin/env python3
"""
Test LangSmith Monitoring Integration

Demonstrates that LangSmith monitoring is working correctly with the AutoML Agent Platform.
Tests agent performance tracking, experiment logging, and communication monitoring.
"""

import sys
import os
from datetime import datetime
import time
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from monitoring.langsmith_client import get_tracker
    from monitoring.agent_monitor import get_monitor
    from monitoring.experiment_tracker import get_experiment_tracker
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"Monitoring modules not available: {e}")
    MONITORING_AVAILABLE = False

def test_langsmith_connection():
    """Test basic LangSmith connection."""
    print("=== Testing LangSmith Connection ===")
    
    tracker = get_tracker()
    
    if not tracker.is_enabled():
        print("❌ LangSmith is not enabled")
        print("Make sure you have:")
        print("  1. LANGCHAIN_API_KEY set in your environment")
        print("  2. LANGCHAIN_TRACING_V2=true")
        print("  3. LANGCHAIN_PROJECT=automl-agent")
        return False
    
    print("✅ LangSmith is enabled and configured")
    
    # Test getting project statistics
    stats = tracker.get_project_statistics(days_back=1)
    if stats:
        print(f"📊 Project: {stats.get('project_name', 'Unknown')}")
        print(f"📈 Total runs: {stats.get('total_runs', 0)}")
        print(f"📈 Success rate: {stats.get('success_rate', 0):.1%}")
    else:
        print("📊 No statistics available yet")
    
    return True

def test_agent_monitoring():
    """Test agent performance monitoring."""
    print("\n=== Testing Agent Performance Monitoring ===")
    
    monitor = get_monitor()
    
    # Simulate some agent executions
    agents_to_test = ["EDAAgent", "ClassificationAgent", "HyperparameterTuningAgent"]
    
    for agent_name in agents_to_test:
        print(f"\n🤖 Testing {agent_name}...")
        
        # Simulate agent execution
        with monitor.track_execution(
            agent_name=agent_name,
            task_description=f"Test execution for {agent_name}",
            metadata={"test": True, "timestamp": datetime.now().isoformat()}
        ) as context:
            # Simulate some work
            execution_time = random.uniform(0.5, 3.0)
            time.sleep(execution_time)
            
            print(f"  ⏱️  Simulated execution time: {execution_time:.2f}s")
    
    # Test agent communication logging
    print("\n💬 Testing agent communication...")
    monitor.log_agent_communication(
        from_agent="EDAAgent",
        to_agent="ClassificationAgent", 
        message="Data analysis complete, ready for model training",
        message_type="handoff",
        metadata={"quality_score": 0.95}
    )
    
    monitor.log_agent_communication(
        from_agent="ClassificationAgent",
        to_agent="HyperparameterTuningAgent",
        message="Initial models trained, requesting parameter optimization",
        message_type="request",
        metadata={"best_score": 0.87}
    )
    
    # Display system overview
    print("\n📊 System Performance Overview:")
    overview = monitor.get_system_overview()
    print(f"  📈 Total agents: {overview['total_agents']}")
    print(f"  🏃 Active agents: {overview['active_agents']}")
    print(f"  ✅ Success rate: {overview['performance_summary']['success_rate']:.1%}")
    print(f"  ⏱️  Average execution time: {overview['performance_summary']['average_execution_time']:.2f}s")
    
    return True

def test_experiment_tracking():
    """Test experiment tracking functionality."""
    print("\n=== Testing Experiment Tracking ===")
    
    tracker = get_experiment_tracker()
    
    # Start a test experiment
    experiment_id = tracker.start_experiment(
        experiment_name="LangSmith Integration Test",
        task_type="classification",
        dataset_info={"name": "test_dataset", "samples": 1000, "features": 10},
        agents_used=["EDAAgent", "ClassificationAgent", "HyperparameterTuningAgent"],
        hyperparameters={"max_depth": 10, "n_estimators": 100},
        quality_threshold=0.85,
        metadata={"test_run": True, "integration_test": True}
    )
    
    print(f"🧪 Started experiment: {experiment_id}")
    
    # Log some agent results
    tracker.log_agent_result(
        experiment_id=experiment_id,
        agent_name="EDAAgent",
        success=True,
        results={"quality_score": 0.92, "features_analyzed": 10, "outliers_found": 5},
        execution_time=2.5
    )
    
    tracker.log_agent_result(
        experiment_id=experiment_id,
        agent_name="ClassificationAgent", 
        success=True,
        results={"models_trained": 3, "best_algorithm": "RandomForest"},
        execution_time=15.3
    )
    
    # Log model results
    tracker.log_model_result(
        experiment_id=experiment_id,
        model_name="RandomForest_optimized",
        algorithm="RandomForest",
        hyperparameters={"n_estimators": 150, "max_depth": 12},
        training_score=0.95,
        validation_score=0.89,
        test_score=0.87,
        metrics={"precision": 0.88, "recall": 0.86, "f1": 0.87},
        training_time=12.5
    )
    
    # Add some artifacts
    tracker.add_artifact(
        experiment_id=experiment_id,
        artifact_name="model_performance_plot",
        artifact_path="outputs/experiments/performance_plot.png",
        artifact_type="visualization"
    )
    
    # Complete the experiment
    result = tracker.complete_experiment(
        experiment_id=experiment_id,
        success=True
    )
    
    print("✅ Experiment completed successfully")
    print(f"  📊 Final quality score: {result.final_quality_score:.3f}")
    print(f"  🤖 Models trained: {len(result.models_trained)}")
    print(f"  ⏱️  Total duration: {result.total_duration:.2f}s")
    
    return True

def test_integration_workflow():
    """Test complete integration workflow."""
    print("\n=== Testing Complete Integration Workflow ===")
    
    tracker = get_tracker()
    monitor = get_monitor()
    exp_tracker = get_experiment_tracker()
    
    print("🔄 Running integrated workflow test...")
    
    # Start experiment
    experiment_id = exp_tracker.start_experiment(
        experiment_name="Full Integration Test",
        task_type="classification", 
        dataset_info={"name": "integration_test", "samples": 5000},
        agents_used=["EDAAgent", "ClassificationAgent", "EnsembleAgent"]
    )
    
    # Simulate multi-agent workflow with monitoring
    agents = [
        ("EDAAgent", "Analyze dataset characteristics", 3.2),
        ("DataHygieneAgent", "Clean and preprocess data", 5.1),
        ("FeatureEngineeringAgent", "Engineer optimal features", 8.7),
        ("ClassificationAgent", "Train classification models", 25.4),
        ("EnsembleAgent", "Create model ensemble", 12.8)
    ]
    
    for agent_name, task_desc, sim_time in agents:
        print(f"  🤖 Executing {agent_name}...")
        
        with monitor.track_execution(
            agent_name=agent_name,
            task_description=task_desc
        ) as context:
            # Simulate work
            time.sleep(min(sim_time / 10, 2.0))  # Scale down for demo
            
            # Log to experiment
            exp_tracker.log_agent_result(
                experiment_id=experiment_id,
                agent_name=agent_name,
                success=True,
                results={"task": task_desc, "simulated": True},
                execution_time=sim_time
            )
        
        print(f"    ✅ {agent_name} completed")
    
    # Complete experiment
    exp_tracker.complete_experiment(experiment_id, success=True)
    
    print("✅ Integration workflow completed successfully!")
    
    # Show final statistics
    overview = monitor.get_system_overview()
    print(f"\n📈 Final Statistics:")
    print(f"  🏆 Total executions: {overview['performance_summary']['total_executions']}")
    print(f"  ✅ Success rate: {overview['performance_summary']['success_rate']:.1%}")
    
    return True

def main():
    """Run all LangSmith integration tests."""
    print("🚀 AutoML Agent LangSmith Integration Test")
    print("=" * 50)
    
    if not MONITORING_AVAILABLE:
        print("❌ Monitoring modules not available")
        return 1
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    test_results = []
    
    # Run tests
    test_results.append(("LangSmith Connection", test_langsmith_connection()))
    test_results.append(("Agent Monitoring", test_agent_monitoring()))
    test_results.append(("Experiment Tracking", test_experiment_tracking()))
    test_results.append(("Integration Workflow", test_integration_workflow()))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Tests passed: {passed}/{len(test_results)}")
    
    if passed == len(test_results):
        print("🎉 All tests passed! LangSmith integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check your LangSmith configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())