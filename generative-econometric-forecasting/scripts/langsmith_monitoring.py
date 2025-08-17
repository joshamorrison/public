#!/usr/bin/env python3
"""
LangSmith Monitoring Dashboard for Generative Econometric Forecasting
Provides insights into AI model performance and usage across the platform.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("LangSmith not available. Install with: pip install langsmith")

def initialize_langsmith_client():
    """Initialize LangSmith client for monitoring."""
    if not LANGSMITH_AVAILABLE:
        return None
    
    api_key = os.getenv('LANGCHAIN_API_KEY')
    if not api_key:
        print("LANGCHAIN_API_KEY not found in environment")
        return None
    
    try:
        client = Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"Failed to initialize LangSmith client: {e}")
        return None

def get_project_statistics(client, project_name: str, days_back: int = 7) -> Dict[str, Any]:
    """Get statistics for a specific project."""
    if not client:
        return {}
    
    try:
        # Get runs from the last week
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            end_time=end_time
        ))
        
        if not runs:
            return {
                'project': project_name,
                'total_runs': 0,
                'success_rate': 0,
                'avg_latency': 0,
                'error_count': 0,
                'time_period': f"Last {days_back} days"
            }
        
        # Calculate statistics
        total_runs = len(runs)
        successful_runs = sum(1 for run in runs if run.status == 'success')
        error_runs = sum(1 for run in runs if run.status == 'error')
        
        # Calculate average latency (in seconds)
        latencies = []
        for run in runs:
            if run.end_time and run.start_time:
                latency = (run.end_time - run.start_time).total_seconds()
                latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        return {
            'project': project_name,
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'error_runs': error_runs,
            'success_rate': round(success_rate, 2),
            'avg_latency_seconds': round(avg_latency, 3),
            'time_period': f"Last {days_back} days"
        }
        
    except Exception as e:
        print(f"Error getting statistics for {project_name}: {e}")
        return {}

def display_monitoring_dashboard():
    """Display comprehensive monitoring dashboard."""
    print("=" * 80)
    print("LANGSMITH AI MONITORING DASHBOARD")
    print("Generative Econometric Forecasting Platform")
    print("=" * 80)
    
    client = initialize_langsmith_client()
    
    if not client:
        print("[X] LangSmith monitoring not available")
        print("   - Check LANGCHAIN_API_KEY environment variable")
        print("   - Install langsmith: pip install langsmith")
        return
    
    print("[OK] LangSmith client connected successfully")
    print()
    
    # Monitor key projects
    projects = [
        'generative-econometric-forecasting',
        'economic-sentiment-analysis', 
        'economic-ai-analysis',
        'huggingface-local-analysis',
        'economic-news-client'
    ]
    
    print("[METRICS] PROJECT PERFORMANCE METRICS")
    print("-" * 50)
    
    all_stats = []
    for project in projects:
        stats = get_project_statistics(client, project, days_back=7)
        if stats:
            all_stats.append(stats)
            
            print(f"[MONITOR] {project}")
            print(f"   Total Runs: {stats.get('total_runs', 0)}")
            print(f"   Success Rate: {stats.get('success_rate', 0)}%")
            print(f"   Avg Latency: {stats.get('avg_latency_seconds', 0)}s")
            print(f"   Errors: {stats.get('error_runs', 0)}")
            print()
    
    # Summary statistics
    if all_stats:
        total_runs = sum(s.get('total_runs', 0) for s in all_stats)
        total_errors = sum(s.get('error_runs', 0) for s in all_stats)
        avg_success_rate = sum(s.get('success_rate', 0) for s in all_stats) / len(all_stats)
        
        print("[SUMMARY] PLATFORM SUMMARY")
        print("-" * 30)
        print(f"   Total AI Operations: {total_runs}")
        print(f"   Platform Success Rate: {avg_success_rate:.1f}%")
        print(f"   Total Errors: {total_errors}")
        print(f"   Active Projects: {len([s for s in all_stats if s.get('total_runs', 0) > 0])}")
        print()
    
    # Usage recommendations
    print("[OPTIMIZE] OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    
    high_error_projects = [s for s in all_stats if s.get('error_runs', 0) > 0]
    slow_projects = [s for s in all_stats if s.get('avg_latency_seconds', 0) > 5.0]
    
    if high_error_projects:
        print("   [WARNING] High Error Rate Projects:")
        for project in high_error_projects:
            print(f"      - {project['project']}: {project['error_runs']} errors")
    
    if slow_projects:
        print("   [SLOW] Slow Response Projects:")
        for project in slow_projects:
            print(f"      - {project['project']}: {project['avg_latency_seconds']}s avg")
    
    if not high_error_projects and not slow_projects:
        print("   [OK] All systems performing optimally!")
    
    print("\n" + "=" * 80)

def monitor_live_runs(duration_minutes: int = 5):
    """Monitor live runs for a specified duration."""
    client = initialize_langsmith_client()
    if not client:
        return
    
    print(f"🔴 LIVE MONITORING (Next {duration_minutes} minutes)")
    print("-" * 50)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    print(f"Monitoring from {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
    print("Run the forecasting system in another terminal to see live traces...")
    print()
    
    last_check = start_time
    
    while datetime.now() < end_time:
        try:
            # Check for new runs since last check
            current_time = datetime.now()
            
            new_runs = list(client.list_runs(
                start_time=last_check,
                end_time=current_time
            ))
            
            for run in new_runs:
                status_emoji = "✅" if run.status == "success" else "❌" if run.status == "error" else "⏳"
                print(f"{status_emoji} {run.name} - {run.run_type} - {run.status}")
            
            last_check = current_time
            
            # Wait before next check
            import time
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"Error during monitoring: {e}")
            break

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LangSmith Monitoring Dashboard")
    parser.add_argument("--live", action="store_true", help="Enable live monitoring mode")
    parser.add_argument("--duration", type=int, default=5, help="Live monitoring duration in minutes")
    
    args = parser.parse_args()
    
    if args.live:
        monitor_live_runs(args.duration)
    else:
        display_monitoring_dashboard()