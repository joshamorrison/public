#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK START DEMO - Multi-Agent Orchestration Platform
Interactive demonstration of all four architecture patterns

Run this after: pip install -r requirements.txt
Goal: ≤5 minutes from clone to working multi-agent system
"""

import sys
import os
import asyncio
import time
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # First try importing as a package
    import src.multi_agent_platform as platform_module
    MultiAgentPlatform = platform_module.MultiAgentPlatform
    print("[SUCCESS] Platform imports successful")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Please ensure you're running from the project root directory with virtual environment activated")
    sys.exit(1)


def print_header():
    """Print demo header."""
    print("\n" + "=" * 70)
    print("[ROBOT] MULTI-AGENT ORCHESTRATION PLATFORM - QUICK START DEMO")
    print("=" * 70)
    print("Revolutionary Four-Pattern Architecture for Agentic AI Systems")
    print("From SIMPLE coordination to SOPHISTICATED self-improvement")
    print("")
    print("[FAST] PATTERNS DEMONSTRATED:")
    print("  [CYCLE] PIPELINE    - Sequential workflow with quality gates")
    print("  [PEOPLE] SUPERVISOR  - Hierarchical coordination & delegation")
    print("  [FAST] PARALLEL    - Concurrent execution with result fusion")
    print("  [SEARCH] REFLECTIVE  - Self-improving feedback loops")
    print("")


def print_separator(title: str):
    """Print section separator."""
    print("\n" + "-" * 60)
    print(f"[LIST] {title.upper()}")
    print("-" * 60)


def print_pattern_intro(pattern_name: str, emoji: str, description: str):
    """Print pattern introduction."""
    print(f"\n{emoji} {pattern_name.upper()} PATTERN")
    print(f"{'=' * (len(pattern_name) + 10)}")
    print(f"[IDEA] {description}")
    print("")


async def demo_pipeline_pattern(platform: MultiAgentPlatform) -> Dict[str, Any]:
    """Demo the Pipeline Pattern."""
    print_pattern_intro(
        "Pipeline", 
        "[CYCLE]", 
        "Sequential workflow: Research → Analysis → Synthesis → Quality Review"
    )
    
    print("[BUILD]  Building research pipeline with 4 stages...")
    
    # Create pre-configured research pipeline
    pipeline = platform.create_research_pipeline("demo_pipeline")
    
    print(f"[OK] Pipeline created with {len(pipeline.pipeline_stages)} stages")
    print("[CHART] Stages configured:")
    for i, stage in enumerate(pipeline.pipeline_stages, 1):
        print(f"   {i}. {stage['stage_name']} ({stage['agent'].name})")
    
    # Execute pipeline
    print("\n[ROCKET] Executing pipeline workflow...")
    
    task = {
        "type": "research",
        "description": "Multi-agent AI systems in enterprise environments",
        "task_id": "pipeline_demo_001",
        "requirements": {
            "depth": "comprehensive",
            "focus": "business_impact"
        }
    }
    
    start_time = time.time()
    result = await platform.execute_pattern(pipeline.pattern_id, task)
    execution_time = time.time() - start_time
    
    print(f"[TIME]  Pipeline execution completed in {execution_time:.2f} seconds")
    print(f"[OK] Success: {result.get('success', False)}")
    
    if result.get("success"):
        pattern_result = result.get("result", {})
        stages_completed = len(pattern_result.get("stages_executed", []))
        print(f"[TARGET] Stages completed: {stages_completed}")
        
        if pattern_result.get("final_result"):
            final_content = pattern_result["final_result"][:200] + "..." if len(pattern_result["final_result"]) > 200 else pattern_result["final_result"]
            print(f"[DOC] Final result preview: {final_content}")
    else:
        print(f"[ERROR] Error: {result.get('error', 'Unknown error')}")
    
    return result


async def demo_supervisor_pattern(platform: MultiAgentPlatform) -> Dict[str, Any]:
    """Demo the Supervisor Pattern."""
    print_pattern_intro(
        "Supervisor", 
        "[PEOPLE]", 
        "Hierarchical coordination: Supervisor delegates to specialist agents"
    )
    
    print("[BUILD]  Building analysis supervisor with specialist agents...")
    
    # Create pre-configured supervisor
    supervisor = platform.create_analysis_supervisor("demo_supervisor")
    
    specialist_count = len(supervisor.supervisor.specialist_agents)
    print(f"[OK] Supervisor created with {specialist_count} specialist agents")
    print("[PEOPLE] Specialists available:")
    for agent_id, agent in supervisor.supervisor.specialist_agents.items():
        capabilities = ", ".join(agent.get_capabilities()[:3])  # First 3 capabilities
        print(f"   • {agent.name}: {capabilities}")
    
    # Execute supervisor coordination
    print("\n[ROCKET] Executing supervisor coordination...")
    
    task = {
        "type": "strategic_analysis",
        "description": "Competitive landscape analysis for AI orchestration platforms",
        "task_id": "supervisor_demo_001",
        "requirements": {
            "scope": "comprehensive",
            "stakeholders": "technical_leaders",
            "deliverable": "executive_summary"
        }
    }
    
    start_time = time.time()
    result = await platform.execute_pattern(supervisor.pattern_id, task)
    execution_time = time.time() - start_time
    
    print(f"[TIME]  Supervisor coordination completed in {execution_time:.2f} seconds")
    print(f"[OK] Success: {result.get('success', False)}")
    
    if result.get("success"):
        pattern_result = result.get("result", {})
        coordination_result = pattern_result.get("coordination_result")
        if coordination_result:
            print(f"[TARGET] Coordination confidence: {coordination_result.confidence:.1%}")
            print(f"[PEOPLE] Specialists utilized: {pattern_result.get('specialists_available', 0)}")
            
            content_preview = coordination_result.content[:200] + "..." if len(coordination_result.content) > 200 else coordination_result.content
            print(f"[DOC] Coordination result preview: {content_preview}")
    else:
        print(f"[ERROR] Error: {result.get('error', 'Unknown error')}")
    
    return result


async def demo_parallel_pattern(platform: MultiAgentPlatform) -> Dict[str, Any]:
    """Demo the Parallel Pattern."""
    print_pattern_intro(
        "Parallel", 
        "[FAST]", 
        "Concurrent execution: Multiple agents work simultaneously, results fused"
    )
    
    print("[BUILD]  Building competitive analysis with parallel agents...")
    
    # Create pre-configured parallel pattern
    parallel = platform.create_competitive_analysis_parallel("demo_parallel")
    
    agent_count = len(parallel.parallel_agents)
    print(f"[OK] Parallel pattern created with {agent_count} concurrent agents")
    print("[FAST] Parallel agents configured:")
    for i, agent_config in enumerate(parallel.parallel_agents, 1):
        agent_name = agent_config['agent'].name
        weight = agent_config['weight']
        variant = agent_config['task_variant']
        print(f"   {i}. {agent_name} (weight: {weight}, variant: {variant})")
    
    print(f"[MERGE] Fusion strategy: {parallel.result_fusion_strategy}")
    
    # Execute parallel processing
    print("\n[ROCKET] Executing parallel processing...")
    
    task = {
        "type": "competitive_analysis",
        "description": "Multi-dimensional analysis of AI agent platforms",
        "task_id": "parallel_demo_001",
        "requirements": {
            "dimensions": ["technology", "market", "pricing"],
            "concurrency": "maximum",
            "fusion": "comprehensive"
        }
    }
    
    start_time = time.time()
    result = await platform.execute_pattern(parallel.pattern_id, task, {"max_concurrent": 3})
    execution_time = time.time() - start_time
    
    print(f"[TIME]  Parallel execution completed in {execution_time:.2f} seconds")
    print(f"[OK] Success: {result.get('success', False)}")
    
    if result.get("success"):
        pattern_result = result.get("result", {})
        agents_executed = pattern_result.get("agents_executed", [])
        fused_result = pattern_result.get("fused_result")
        
        print(f"[FAST] Agents executed: {len(agents_executed)}")
        if pattern_result.get("partial_failure"):
            failed_agents = pattern_result.get("failed_agents", [])
            print(f"[WARNING]  Partial failure: {len(failed_agents)} agents failed")
        
        if fused_result:
            print(f"[MERGE] Fusion confidence: {fused_result.get('confidence', 0):.1%}")
            print(f"[CHART] Contributing agents: {fused_result.get('contributing_agents', 0)}")
            
            content_preview = fused_result.get("content", "")[:200] + "..." if len(fused_result.get("content", "")) > 200 else fused_result.get("content", "")
            print(f"[DOC] Fused result preview: {content_preview}")
    else:
        print(f"[ERROR] Error: {result.get('error', 'Unknown error')}")
    
    return result


async def demo_reflective_pattern(platform: MultiAgentPlatform) -> Dict[str, Any]:
    """Demo the Reflective Pattern."""
    print_pattern_intro(
        "Reflective", 
        "[SEARCH]", 
        "Self-improving: Iterative refinement with critic feedback and meta-reasoning"
    )
    
    print("[BUILD]  Building content optimization with reflective improvement...")
    
    # Create pre-configured reflective pattern
    reflective = platform.create_content_optimization_reflective("demo_reflective")
    
    primary_agent = reflective.primary_agent.name if reflective.primary_agent else "None"
    critics_count = len(reflective.critic_agents)
    
    print(f"[OK] Reflective pattern created")
    print(f"[TARGET] Primary agent: {primary_agent}")
    print(f"[EYE]  Critic agents: {critics_count}")
    print("[TOOL] Reflection configuration:")
    print(f"   • Max iterations: {reflective.max_iterations}")
    print(f"   • Convergence threshold: {reflective.convergence_threshold:.1%}")
    print(f"   • Meta-reasoning: {'Enabled' if reflective.enable_meta_reasoning else 'Disabled'}")
    print(f"   • Peer review: {'Enabled' if reflective.enable_peer_review else 'Disabled'}")
    
    # Execute reflective improvement
    print("\n[ROCKET] Executing reflective improvement process...")
    
    task = {
        "type": "content_optimization", 
        "description": "Strategic whitepaper on multi-agent orchestration best practices",
        "task_id": "reflective_demo_001",
        "requirements": {
            "quality_level": "publication_ready",
            "target_audience": "enterprise_architects",
            "iteration_focus": "clarity_and_impact"
        }
    }
    
    start_time = time.time()
    result = await platform.execute_pattern(reflective.pattern_id, task)
    execution_time = time.time() - start_time
    
    print(f"[TIME]  Reflective process completed in {execution_time:.2f} seconds")
    print(f"[OK] Success: {result.get('success', False)}")
    
    if result.get("success"):
        pattern_result = result.get("result", {})
        iterations = pattern_result.get("iterations", [])
        final_result = pattern_result.get("final_result")
        
        print(f"[CYCLE] Iterations completed: {len(iterations)}")
        print(f"[TARGET] Improvement achieved: {'Yes' if pattern_result.get('improvement_achieved', False) else 'No'}")
        print(f"[STAR] Convergence reached: {'Yes' if pattern_result.get('convergence_reached', False) else 'No'}")
        
        if pattern_result.get("total_improvement"):
            print(f"[UP] Total improvement: {pattern_result['total_improvement']:.3f}")
        
        if final_result:
            print(f"[STAR] Final confidence: {final_result.confidence:.1%}")
            content_preview = final_result.content[:200] + "..." if len(final_result.content) > 200 else final_result.content
            print(f"[DOC] Final result preview: {content_preview}")
    else:
        print(f"[ERROR] Error: {result.get('error', 'Unknown error')}")
    
    return result


async def run_comprehensive_demo():
    """Run comprehensive demonstration of all patterns."""
    print_header()
    
    # Initialize platform
    print_separator("Platform Initialization")
    print("[ROCKET] Initializing Multi-Agent Orchestration Platform...")
    
    platform = MultiAgentPlatform("quickstart_demo")
    
    print("[OK] Platform initialized successfully")
    print(f"[ID] Platform ID: {platform.platform_id}")
    print(f"[BOX] Version: {platform.version}")
    
    # Track demo results
    demo_results = {}
    total_start_time = time.time()
    
    # Demo each pattern
    patterns = [
        ("pipeline", demo_pipeline_pattern),
        ("supervisor", demo_supervisor_pattern),
        ("parallel", demo_parallel_pattern),
        ("reflective", demo_reflective_pattern)
    ]
    
    for pattern_name, demo_func in patterns:
        print_separator(f"{pattern_name} Pattern Demo")
        
        try:
            result = await demo_func(platform)
            demo_results[pattern_name] = {
                "success": result.get("success", False),
                "execution_time": result.get("execution_time", 0),
                "error": result.get("error")
            }
            
            print(f"\n[OK] {pattern_name.title()} pattern demo completed!")
            
        except Exception as e:
            print(f"\n[ERROR] {pattern_name.title()} pattern demo failed: {str(e)}")
            demo_results[pattern_name] = {
                "success": False,
                "error": str(e),
                "execution_time": 0
            }
    
    # Summary
    total_time = time.time() - total_start_time
    successful_patterns = sum(1 for result in demo_results.values() if result["success"])
    
    print_separator("Demo Summary")
    print("[TARGET] QUICK START DEMO RESULTS")
    print(f"[TIME]  Total execution time: {total_time:.2f} seconds")
    print(f"[OK] Successful patterns: {successful_patterns}/{len(patterns)}")
    print(f"[CHART] Success rate: {successful_patterns/len(patterns):.1%}")
    
    print("\n[LIST] Pattern Results:")
    for pattern_name, result in demo_results.items():
        status = "[OK] SUCCESS" if result["success"] else "[ERROR] FAILED"
        time_str = f"{result['execution_time']:.2f}s" if result["success"] else "N/A"
        print(f"   {pattern_name.upper():<12}: {status:<10} ({time_str})")
        if not result["success"] and result.get("error"):
            print(f"      Error: {result['error']}")
    
    # Platform status
    print("\n[SEARCH] Platform Status:")
    platform_status = platform.get_platform_status()
    registry_status = platform_status["registry_status"]
    
    print(f"   • Agents created: {registry_status['registered_agents']}")
    print(f"   • Patterns created: {registry_status['active_patterns']}")
    print(f"   • Workflows executed: {platform_status['platform_metrics']['workflows_executed']}")
    
    print("\n[PARTY] Multi-Agent Orchestration Platform Quick Start Demo Complete!")
    print("[IDEA] Next steps:")
    print("   • Explore individual pattern APIs")
    print("   • Customize agents and workflows")
    print("   • Build production applications")
    print("   • Monitor performance and optimize")
    
    return demo_results


def check_dependencies():
    """Check if basic dependencies are available."""
    print("[SEARCH] Checking dependencies...")
    
    missing_deps = []
    
    try:
        import asyncio
        print("[OK] asyncio available")
    except ImportError:
        missing_deps.append("asyncio")
    
    # Check if our platform modules load
    try:
        from src.multi_agent_platform import MultiAgentPlatform
        print("[OK] Platform module available")
    except ImportError as e:
        print(f"[ERROR] Platform module error: {e}")
        missing_deps.append("platform_modules")
    
    if missing_deps:
        print("[ERROR] Missing dependencies:")
        for dep in missing_deps:
            print(f"   • {dep}")
        return False
    
    print("[OK] All dependencies available")
    return True


def print_instructions():
    """Print usage instructions."""
    print("\n[BOOK] USAGE INSTRUCTIONS")
    print("=" * 50)
    print("This quick start demo showcases all four multi-agent patterns:")
    print("")
    print("[CYCLE] PIPELINE PATTERN")
    print("   Sequential workflow with quality gates")
    print("   Use case: Content creation, data processing")
    print("")
    print("[PEOPLE] SUPERVISOR PATTERN")
    print("   Hierarchical coordination with specialists")
    print("   Use case: Complex analysis, strategic planning")
    print("")
    print("[FAST] PARALLEL PATTERN")
    print("   Concurrent execution with result fusion")
    print("   Use case: Competitive analysis, market research")
    print("")
    print("[SEARCH] REFLECTIVE PATTERN")
    print("   Self-improving with iterative refinement")
    print("   Use case: Content optimization, strategic thinking")
    print("")
    print("[ROCKET] Run the demo to see all patterns in action!")
    print("   Each pattern will execute with synthetic data")
    print("   showing real multi-agent coordination.")


async def main():
    """Main entry point for quick start demo."""
    print("Multi-Agent Orchestration Platform - Quick Start")
    print("=" * 55)
    
    # Check dependencies
    if not check_dependencies():
        print("\n[ERROR] Dependencies check failed. Please install requirements:")
        print("   pip install -r requirements.txt")
        return
    
    print_instructions()
    
    # Ask user if they want to run the demo
    try:
        response = input("\n[ROCKET] Run the comprehensive demo? (y/n): ").lower().strip()
        if response in ['y', 'yes', '']:
            await run_comprehensive_demo()
        else:
            print("[WAVE] Demo skipped. You can run it anytime with:")
            print("   python quick_start.py")
    
    except KeyboardInterrupt:
        print("\n\n[WAVE] Demo interrupted by user. Goodbye!")
    
    except Exception as e:
        print(f"\n[ERROR] Demo failed with error: {str(e)}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    print("[ROBOT] Starting Multi-Agent Orchestration Platform Demo...")
    asyncio.run(main())