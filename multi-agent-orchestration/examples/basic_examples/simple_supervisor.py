"""
Simple Supervisor Example - Research Coordination

Demonstrates hierarchical coordination where a supervisor agent 
coordinates and delegates work to specialist agents.

This example shows intelligent task decomposition, delegation,
and result synthesis through centralized coordination.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List

from src.agents.supervisor_agent import SupervisorAgent
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.summary_agent import SummaryAgent


async def run_research_coordination(research_topic: str = "quantum computing applications") -> Dict[str, Any]:
    """
    Run coordinated research using the Supervisor pattern.
    
    The supervisor intelligently decomposes the research task and
    delegates specialized work to appropriate agents.
    
    Workflow:
    1. Supervisor analyzes the research requirements
    2. Supervisor delegates specific tasks to specialist agents
    3. Specialists complete their assigned work
    4. Supervisor synthesizes all results into final output
    
    Args:
        research_topic: Topic to research comprehensively
        
    Returns:
        Coordinated research results with supervision metadata
    """
    print(f"👥 Starting Supervised Research Coordination for: {research_topic}")
    print("=" * 70)
    
    # Initialize supervisor and specialist agents
    supervisor = SupervisorAgent()
    research_agent = ResearchAgent()
    analysis_agent = AnalysisAgent()
    summary_agent = SummaryAgent()
    
    # Register specialists with supervisor
    specialists = {
        "research_specialist": research_agent,
        "analysis_specialist": analysis_agent,
        "synthesis_specialist": summary_agent
    }
    
    print("🎯 Phase 1: Supervisor Planning & Task Decomposition")
    
    # Supervisor analyzes the research requirements
    planning_task = {
        "type": "research_planning",
        "description": f"Plan comprehensive research approach for {research_topic}",
        "research_scope": research_topic,
        "available_specialists": list(specialists.keys()),
        "specialist_capabilities": {
            "research_specialist": "Information gathering, source analysis, fact finding",
            "analysis_specialist": "Data analysis, pattern recognition, insight extraction",
            "synthesis_specialist": "Content creation, summarization, report generation"
        },
        "research_requirements": [
            "Current state of the field",
            "Key technological developments",
            "Market applications and potential",
            "Future trends and implications",
            "Challenges and limitations"
        ]
    }
    
    supervision_plan = await supervisor.process_task(planning_task)
    print(f"✅ Supervisor planning completed with confidence: {supervision_plan.confidence}")
    print(f"📋 Research plan: {supervision_plan.content[:200]}...")
    print()
    
    # Parse supervisor's delegation decisions
    delegations = parse_supervisor_delegations(supervision_plan.content, research_topic)
    
    print("🎯 Phase 2: Task Delegation & Specialist Execution")
    
    specialist_results = {}
    
    # Execute delegated tasks
    for delegation in delegations:
        specialist_id = delegation["assigned_specialist"]
        task_description = delegation["task_description"]
        
        if specialist_id in specialists:
            agent = specialists[specialist_id]
            
            print(f"📤 Delegating to {specialist_id}: {task_description[:100]}...")
            
            # Create specialized task
            specialist_task = {
                "type": delegation["task_type"],
                "description": task_description,
                "research_topic": research_topic,
                "specific_focus": delegation["focus_areas"],
                "supervisor_guidance": supervision_plan.content,
                "coordination_context": {
                    "other_specialists": [s for s in specialists.keys() if s != specialist_id],
                    "overall_objective": f"Comprehensive research on {research_topic}"
                }
            }
            
            # Execute specialist task
            result = await agent.process_task(specialist_task)
            specialist_results[specialist_id] = {
                "result": result,
                "delegation": delegation,
                "completion_time": datetime.now()
            }
            
            print(f"✅ {specialist_id} completed task with confidence: {result.confidence}")
            print(f"📄 Result preview: {result.content[:150]}...")
            print()
    
    print("🎯 Phase 3: Supervisor Synthesis & Final Coordination")
    
    # Supervisor synthesizes all specialist results
    synthesis_task = {
        "type": "research_synthesis",
        "description": f"Synthesize all specialist research on {research_topic}",
        "research_topic": research_topic,
        "specialist_results": {
            specialist_id: data["result"].content 
            for specialist_id, data in specialist_results.items()
        },
        "original_plan": supervision_plan.content,
        "synthesis_requirements": [
            "Integrate all specialist perspectives",
            "Identify key themes and insights",
            "Resolve any conflicting information",
            "Create coherent narrative",
            "Highlight most significant findings"
        ]
    }
    
    final_synthesis = await supervisor.process_task(synthesis_task)
    print(f"✅ Supervisor synthesis completed with confidence: {final_synthesis.confidence}")
    print(f"📑 Final synthesis preview: {final_synthesis.content[:200]}...")
    print()
    
    # Calculate coordination metrics
    total_specialists = len(specialist_results)
    avg_specialist_confidence = sum(
        data["result"].confidence for data in specialist_results.values()
    ) / max(total_specialists, 1)
    
    overall_confidence = (supervision_plan.confidence + final_synthesis.confidence + avg_specialist_confidence) / 3
    
    # Compile comprehensive results
    coordination_results = {
        "workflow_type": "supervisor",
        "research_topic": research_topic,
        "execution_time": datetime.now().isoformat(),
        "supervision_phases": {
            "planning": {
                "supervisor_plan": supervision_plan.content,
                "planning_confidence": supervision_plan.confidence,
                "delegations_created": len(delegations)
            },
            "delegation_execution": {
                "specialists_used": list(specialist_results.keys()),
                "specialist_results": {
                    specialist_id: {
                        "confidence": data["result"].confidence,
                        "content": data["result"].content,
                        "task_focus": data["delegation"]["focus_areas"],
                        "metadata": data["result"].metadata
                    }
                    for specialist_id, data in specialist_results.items()
                }
            },
            "synthesis": {
                "final_output": final_synthesis.content,
                "synthesis_confidence": final_synthesis.confidence,
                "integration_quality": "High" if final_synthesis.confidence > 0.8 else "Medium"
            }
        },
        "coordination_metrics": {
            "total_specialists": total_specialists,
            "successful_delegations": len(specialist_results),
            "avg_specialist_confidence": avg_specialist_confidence,
            "supervisor_confidence": (supervision_plan.confidence + final_synthesis.confidence) / 2,
            "overall_confidence": overall_confidence,
            "coordination_efficiency": f"{total_specialists} specialists coordinated successfully"
        },
        "final_research_output": final_synthesis.content
    }
    
    print("🎉 Supervised Research Coordination Completed Successfully!")
    print(f"👥 Specialists Coordinated: {total_specialists}")
    print(f"📈 Overall Confidence: {overall_confidence:.2f}")
    print(f"🎯 Coordination Efficiency: 100% (all delegations successful)")
    print("=" * 70)
    
    return coordination_results


def parse_supervisor_delegations(supervision_plan: str, research_topic: str) -> List[Dict[str, Any]]:
    """
    Parse supervisor's plan into specific delegations.
    
    In a production system, this would use NLP to parse the supervisor's
    natural language plan. For this demo, we'll create structured delegations.
    """
    # Simulate intelligent delegation parsing
    delegations = [
        {
            "assigned_specialist": "research_specialist",
            "task_type": "research",
            "task_description": f"Conduct comprehensive information gathering on {research_topic}, focusing on current state, key developments, and technological foundations",
            "focus_areas": [
                "Current technological state",
                "Key research developments",
                "Leading researchers and institutions",
                "Foundational concepts and principles"
            ]
        },
        {
            "assigned_specialist": "analysis_specialist", 
            "task_type": "analysis",
            "task_description": f"Analyze the market applications, commercial potential, and competitive landscape for {research_topic}",
            "focus_areas": [
                "Market applications and use cases",
                "Commercial viability and potential",
                "Competitive landscape analysis",
                "Investment and funding trends"
            ]
        },
        {
            "assigned_specialist": "synthesis_specialist",
            "task_type": "synthesis",
            "task_description": f"Synthesize future outlook, challenges, and strategic implications for {research_topic}",
            "focus_areas": [
                "Future trends and predictions",
                "Technical and practical challenges",
                "Strategic implications for businesses",
                "Policy and regulatory considerations"
            ]
        }
    ]
    
    return delegations


async def run_simple_supervisor_demo():
    """Run a simple demo of the supervisor pattern."""
    print("🚀 Multi-Agent Supervisor Pattern Demo")
    print("Demonstrating hierarchical coordination and intelligent delegation")
    print()
    
    # Run supervised research coordination
    results = await run_research_coordination("blockchain technology applications")
    
    # Display summary
    print("\n📋 Supervision Summary:")
    print(f"Research Topic: {results['research_topic']}")
    print(f"Specialists Coordinated: {results['coordination_metrics']['total_specialists']}")
    print(f"Successful Delegations: {results['coordination_metrics']['successful_delegations']}")
    print(f"Overall Confidence: {results['coordination_metrics']['overall_confidence']:.2f}")
    print(f"Supervisor Confidence: {results['coordination_metrics']['supervisor_confidence']:.2f}")
    print(f"Average Specialist Confidence: {results['coordination_metrics']['avg_specialist_confidence']:.2f}")
    print(f"Final Output Length: {len(results['final_research_output'])} characters")
    
    return results


async def demonstrate_coordination_benefits():
    """Demonstrate the benefits of supervisor coordination."""
    print("\n🔬 Coordination Benefits Demonstration")
    print("=" * 50)
    
    topic = "artificial intelligence ethics"
    
    # Run supervised coordination
    supervised_results = await run_research_coordination(topic)
    
    # Analyze coordination benefits
    benefits_analysis = {
        "task_decomposition": "Supervisor intelligently broke down complex research into specialized tasks",
        "resource_optimization": f"Efficiently utilized {supervised_results['coordination_metrics']['total_specialists']} specialists",
        "quality_assurance": f"Supervisor synthesis achieved {supervised_results['coordination_metrics']['supervisor_confidence']:.2f} confidence",
        "comprehensive_coverage": "Multiple specialist perspectives ensure thorough analysis",
        "centralized_coordination": "Single point of coordination prevents conflicts and gaps"
    }
    
    print("🎯 Coordination Benefits:")
    for benefit, description in benefits_analysis.items():
        print(f"• {benefit.replace('_', ' ').title()}: {description}")
    
    return benefits_analysis


if __name__ == "__main__":
    # Run the demos
    print("=" * 80)
    results = asyncio.run(run_simple_supervisor_demo())
    
    # Demonstrate coordination benefits
    benefits = asyncio.run(demonstrate_coordination_benefits())
    
    print("\n🎯 Key Takeaways:")
    print("1. Supervisor pattern enables intelligent task decomposition")
    print("2. Centralized coordination ensures efficient resource utilization")
    print("3. Specialist agents focus on their areas of expertise")
    print("4. Supervisor synthesis integrates diverse perspectives")
    print("5. Hierarchical structure scales to complex, multi-faceted problems")
    print("6. Quality assurance through supervisor oversight and final synthesis")
    print("7. Ideal for complex projects requiring multiple types of expertise")