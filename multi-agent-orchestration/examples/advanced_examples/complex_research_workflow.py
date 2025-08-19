"""
Complex Research Workflow - Multi-Pattern Integration

Demonstrates sophisticated research workflow combining multiple orchestration patterns:
- Pipeline for sequential research phases
- Parallel for concurrent analysis streams
- Supervisor for coordination and synthesis
- Reflective for quality improvement

This example shows enterprise-grade research capabilities with
comprehensive analysis, validation, and reporting.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.orchestration.workflow_engine import WorkflowEngine


async def run_comprehensive_research(
    research_query: str, 
    depth_level: str = "comprehensive",
    validation_required: bool = True
) -> Dict[str, Any]:
    """
    Execute a complex research workflow using multiple orchestration patterns.
    
    Workflow Structure:
    1. Research Planning (Supervisor)
    2. Information Gathering (Pipeline + Parallel)
    3. Analysis & Synthesis (Parallel + Supervisor)  
    4. Quality Validation (Reflective)
    5. Final Report Generation (Pipeline)
    
    Args:
        research_query: Complex research question or topic
        depth_level: "basic", "comprehensive", or "exhaustive"
        validation_required: Whether to include quality validation phase
        
    Returns:
        Comprehensive research results with multi-pattern orchestration
    """
    print(f"🔬 Starting Complex Research Workflow")
    print(f"📋 Query: {research_query}")
    print(f"🎯 Depth: {depth_level}")
    print("=" * 80)
    
    # Initialize specialized agent teams
    supervisor = SupervisorAgent()
    research_team = [ResearchAgent() for _ in range(3)]  # Multiple research specialists
    analysis_team = [AnalysisAgent() for _ in range(2)]  # Analysis specialists
    synthesis_agent = SummaryAgent()
    
    # Workflow engine for orchestration
    workflow_engine = WorkflowEngine()
    
    # Phase 1: Research Planning & Coordination (Supervisor Pattern)
    print("📋 Phase 1: Research Planning & Strategic Coordination")
    print("-" * 60)
    
    planning_task = {
        "type": "research_planning",
        "description": f"Plan comprehensive research strategy for: {research_query}",
        "complexity_level": depth_level,
        "research_scope": {
            "primary_questions": extract_research_questions(research_query),
            "methodology_requirements": determine_methodology(depth_level),
            "resource_allocation": plan_resource_allocation(len(research_team), len(analysis_team)),
            "timeline_expectations": "multi-phase execution",
            "quality_standards": "enterprise-grade accuracy and completeness"
        },
        "team_composition": {
            "research_specialists": len(research_team),
            "analysis_specialists": len(analysis_team),
            "synthesis_capabilities": ["content_creation", "report_generation", "executive_summary"]
        }
    }
    
    research_plan = await supervisor.process_task(planning_task)
    print(f"✅ Research planning completed with confidence: {research_plan.confidence:.2f}")
    print(f"📋 Strategic plan: {research_plan.content[:200]}...")
    print()
    
    # Phase 2: Multi-Stream Information Gathering (Pipeline + Parallel)
    print("📚 Phase 2: Multi-Stream Information Gathering")
    print("-" * 60)
    
    # Parallel information gathering streams
    research_streams = create_research_streams(research_query, research_plan.content, depth_level)
    
    print(f"🔀 Executing {len(research_streams)} parallel research streams...")
    
    # Execute research streams in parallel
    research_tasks = []
    for i, (agent, stream_task) in enumerate(zip(research_team, research_streams)):
        print(f"📤 Stream {i+1}: {stream_task['focus_area']}")
        research_tasks.append(agent.process_task(stream_task))
    
    # Wait for all research streams to complete
    research_results = await asyncio.gather(*research_tasks, return_exceptions=True)
    
    print(f"✅ All research streams completed")
    for i, result in enumerate(research_results):
        if not isinstance(result, Exception):
            print(f"   Stream {i+1}: {result.confidence:.2f} confidence")
    print()
    
    # Phase 3: Parallel Analysis & Synthesis Coordination
    print("🔍 Phase 3: Parallel Analysis & Synthesis")
    print("-" * 60)
    
    # Create analysis tasks based on research results
    analysis_tasks = create_analysis_tasks(research_results, research_query)
    
    # Execute analysis in parallel
    print(f"🔀 Executing {len(analysis_tasks)} parallel analysis streams...")
    
    analysis_coroutines = []
    for i, (agent, task) in enumerate(zip(analysis_team, analysis_tasks)):
        print(f"📊 Analysis {i+1}: {task['analysis_type']}")
        analysis_coroutines.append(agent.process_task(task))
    
    analysis_results = await asyncio.gather(*analysis_coroutines, return_exceptions=True)
    
    print(f"✅ All analysis streams completed")
    for i, result in enumerate(analysis_results):
        if not isinstance(result, Exception):
            print(f"   Analysis {i+1}: {result.confidence:.2f} confidence")
    print()
    
    # Supervisor synthesis of all results
    print("🎯 Supervisor-Coordinated Synthesis...")
    
    synthesis_task = {
        "type": "comprehensive_synthesis",
        "description": f"Synthesize all research and analysis for: {research_query}",
        "research_inputs": [r.content for r in research_results if not isinstance(r, Exception)],
        "analysis_inputs": [a.content for a in analysis_results if not isinstance(a, Exception)],
        "original_plan": research_plan.content,
        "synthesis_requirements": [
            "Integrate all research perspectives",
            "Resolve conflicting information",
            "Identify key insights and patterns",
            "Create coherent narrative",
            "Highlight critical findings",
            "Assess reliability and confidence levels"
        ]
    }
    
    synthesis_result = await synthesis_agent.process_task(synthesis_task)
    print(f"✅ Synthesis completed with confidence: {synthesis_result.confidence:.2f}")
    print()
    
    # Phase 4: Quality Validation (Reflective Pattern)
    if validation_required:
        print("🔬 Phase 4: Quality Validation & Improvement")
        print("-" * 60)
        
        validation_cycles = []
        current_synthesis = synthesis_result.content
        
        for cycle in range(2):  # Multiple validation cycles
            print(f"🔄 Validation Cycle {cycle + 1}")
            
            # Critical evaluation
            validation_task = {
                "type": "research_validation",
                "description": f"Validate research quality and completeness for: {research_query}",
                "content_to_validate": current_synthesis,
                "validation_criteria": [
                    "Factual accuracy and source reliability",
                    "Comprehensiveness and coverage",
                    "Logical consistency and coherence",
                    "Methodological rigor",
                    "Bias detection and mitigation",
                    "Gap identification and resolution"
                ],
                "original_research_data": [r.content for r in research_results if not isinstance(r, Exception)]
            }
            
            validation_result = await analysis_team[0].process_task(validation_task)
            
            # Improvement implementation if needed
            if validation_result.confidence < 0.9 and cycle < 1:
                improvement_task = {
                    "type": "research_improvement", 
                    "description": f"Improve research synthesis based on validation feedback",
                    "current_content": current_synthesis,
                    "validation_feedback": validation_result.content,
                    "improvement_targets": extract_improvement_areas(validation_result.content)
                }
                
                improved_synthesis = await synthesis_agent.process_task(improvement_task)
                current_synthesis = improved_synthesis.content
                
                print(f"   Improved synthesis confidence: {improved_synthesis.confidence:.2f}")
            
            validation_cycles.append({
                "cycle": cycle + 1,
                "validation_confidence": validation_result.confidence,
                "validation_feedback": validation_result.content
            })
            
            print(f"   Validation confidence: {validation_result.confidence:.2f}")
        
        print("✅ Quality validation completed")
        print()
    
    # Phase 5: Final Report Generation (Pipeline Pattern)
    print("📄 Phase 5: Final Report Generation")
    print("-" * 60)
    
    # Sequential report generation pipeline
    report_sections = ["executive_summary", "methodology", "findings", "analysis", "conclusions", "recommendations"]
    
    final_report_sections = {}
    
    for section in report_sections:
        section_task = {
            "type": "report_section_generation",
            "description": f"Generate {section} section for research report",
            "section_type": section,
            "research_content": current_synthesis if validation_required else synthesis_result.content,
            "previous_sections": final_report_sections,
            "formatting_requirements": get_section_requirements(section),
            "target_audience": "executive leadership and technical stakeholders"
        }
        
        section_result = await synthesis_agent.process_task(section_task)
        final_report_sections[section] = section_result.content
        
        print(f"✅ {section.replace('_', ' ').title()} section completed")
    
    print()
    
    # Compile comprehensive results
    workflow_results = {
        "workflow_type": "complex_research",
        "research_query": research_query,
        "depth_level": depth_level,
        "validation_enabled": validation_required,
        "execution_time": datetime.now().isoformat(),
        "orchestration_phases": {
            "planning": {
                "supervisor_plan": research_plan.content,
                "planning_confidence": research_plan.confidence,
                "resource_allocation": f"{len(research_team)} research + {len(analysis_team)} analysis agents"
            },
            "information_gathering": {
                "research_streams": len(research_streams),
                "successful_streams": len([r for r in research_results if not isinstance(r, Exception)]),
                "research_results": [
                    {"confidence": r.confidence, "content_length": len(r.content)}
                    for r in research_results if not isinstance(r, Exception)
                ]
            },
            "analysis_synthesis": {
                "analysis_streams": len(analysis_tasks),
                "successful_analyses": len([a for a in analysis_results if not isinstance(a, Exception)]),
                "synthesis_confidence": synthesis_result.confidence,
                "synthesis_length": len(synthesis_result.content)
            },
            "quality_validation": validation_cycles if validation_required else "skipped",
            "report_generation": {
                "sections_generated": list(final_report_sections.keys()),
                "total_report_length": sum(len(content) for content in final_report_sections.values())
            }
        },
        "final_outputs": {
            "synthesis": current_synthesis if validation_required else synthesis_result.content,
            "report_sections": final_report_sections,
            "full_report": compile_full_report(final_report_sections)
        },
        "quality_metrics": {
            "overall_confidence": calculate_overall_confidence(
                research_plan, research_results, analysis_results, synthesis_result
            ),
            "validation_confidence": validation_cycles[-1]["validation_confidence"] if validation_required else None,
            "comprehensiveness_score": calculate_comprehensiveness(final_report_sections),
            "multi_pattern_coordination": "Successful"
        }
    }
    
    print("🎉 Complex Research Workflow Completed Successfully!")
    print(f"📊 Overall Confidence: {workflow_results['quality_metrics']['overall_confidence']:.2f}")
    print(f"📄 Final Report Length: {len(workflow_results['final_outputs']['full_report'])} characters")
    print(f"🔬 Research Streams: {len(research_streams)} executed successfully")
    print(f"🔍 Analysis Streams: {len(analysis_tasks)} completed")
    print("=" * 80)
    
    return workflow_results


def extract_research_questions(query: str) -> List[str]:
    """Extract primary research questions from the main query."""
    return [
        f"What is the current state of {query}?",
        f"What are the key challenges and opportunities in {query}?",
        f"What are the emerging trends and future outlook for {query}?",
        f"Who are the key stakeholders and decision makers in {query}?"
    ]


def determine_methodology(depth_level: str) -> List[str]:
    """Determine research methodology based on depth level."""
    methodologies = {
        "basic": ["literature_review", "source_verification"],
        "comprehensive": ["literature_review", "comparative_analysis", "trend_analysis", "stakeholder_mapping"],
        "exhaustive": ["literature_review", "comparative_analysis", "trend_analysis", "stakeholder_mapping", 
                      "quantitative_analysis", "scenario_modeling", "expert_synthesis"]
    }
    return methodologies.get(depth_level, methodologies["comprehensive"])


def plan_resource_allocation(research_agents: int, analysis_agents: int) -> Dict[str, Any]:
    """Plan resource allocation across the workflow."""
    return {
        "research_capacity": f"{research_agents} parallel research streams",
        "analysis_capacity": f"{analysis_agents} parallel analysis streams", 
        "coordination_overhead": "Supervisor-managed with quality gates",
        "estimated_completion": "Multi-phase execution with validation cycles"
    }


def create_research_streams(query: str, plan: str, depth: str) -> List[Dict[str, Any]]:
    """Create specialized research streams based on the plan."""
    streams = [
        {
            "type": "foundational_research",
            "description": f"Conduct foundational research on {query}",
            "focus_area": "Core concepts and fundamentals",
            "research_scope": ["definitions", "history", "key_principles", "foundational_literature"],
            "depth_level": depth
        },
        {
            "type": "current_state_research", 
            "description": f"Research current state and recent developments in {query}",
            "focus_area": "Current developments and trends",
            "research_scope": ["recent_news", "industry_reports", "market_data", "expert_opinions"],
            "depth_level": depth
        },
        {
            "type": "comparative_research",
            "description": f"Conduct comparative analysis related to {query}",
            "focus_area": "Comparative and competitive landscape",
            "research_scope": ["alternatives", "competitors", "best_practices", "benchmarking"],
            "depth_level": depth
        }
    ]
    return streams


def create_analysis_tasks(research_results: List[Any], query: str) -> List[Dict[str, Any]]:
    """Create analysis tasks based on research results."""
    return [
        {
            "type": "trend_analysis",
            "description": f"Analyze trends and patterns from research on {query}",
            "analysis_type": "Trend and Pattern Analysis",
            "research_data": [r.content for r in research_results if not isinstance(r, Exception)],
            "analysis_focus": ["emerging_trends", "pattern_identification", "trend_correlation", "future_projections"]
        },
        {
            "type": "stakeholder_analysis",
            "description": f"Analyze stakeholders and market dynamics for {query}",
            "analysis_type": "Stakeholder and Market Analysis", 
            "research_data": [r.content for r in research_results if not isinstance(r, Exception)],
            "analysis_focus": ["key_players", "market_dynamics", "influence_mapping", "decision_factors"]
        }
    ]


def extract_improvement_areas(validation_feedback: str) -> List[str]:
    """Extract specific areas for improvement from validation feedback."""
    return [
        "Enhance source diversity and reliability",
        "Strengthen quantitative analysis components",
        "Improve clarity in complex technical sections",
        "Add more recent developments and updates"
    ]


def get_section_requirements(section: str) -> Dict[str, Any]:
    """Get formatting requirements for report sections."""
    requirements = {
        "executive_summary": {"length": "2-3 paragraphs", "focus": "key_findings_and_recommendations"},
        "methodology": {"length": "1-2 paragraphs", "focus": "research_approach_and_sources"},
        "findings": {"length": "detailed", "focus": "comprehensive_research_results"},
        "analysis": {"length": "detailed", "focus": "insights_and_interpretations"},
        "conclusions": {"length": "1-2 paragraphs", "focus": "key_takeaways"},
        "recommendations": {"length": "bulleted_list", "focus": "actionable_next_steps"}
    }
    return requirements.get(section, {"length": "standard", "focus": "comprehensive"})


def compile_full_report(sections: Dict[str, str]) -> str:
    """Compile all sections into a full report."""
    report = "COMPREHENSIVE RESEARCH REPORT\n"
    report += "=" * 50 + "\n\n"
    
    for section_name, content in sections.items():
        report += f"{section_name.replace('_', ' ').upper()}\n"
        report += "-" * 30 + "\n"
        report += content + "\n\n"
    
    return report


def calculate_overall_confidence(plan, research_results, analysis_results, synthesis) -> float:
    """Calculate overall workflow confidence."""
    research_conf = sum(r.confidence for r in research_results if not isinstance(r, Exception)) / max(len(research_results), 1)
    analysis_conf = sum(a.confidence for a in analysis_results if not isinstance(a, Exception)) / max(len(analysis_results), 1)
    
    return (plan.confidence + research_conf + analysis_conf + synthesis.confidence) / 4


def calculate_comprehensiveness(report_sections: Dict[str, str]) -> float:
    """Calculate comprehensiveness score based on report completeness."""
    expected_sections = ["executive_summary", "methodology", "findings", "analysis", "conclusions", "recommendations"]
    completion_rate = len(report_sections) / len(expected_sections)
    
    avg_section_length = sum(len(content) for content in report_sections.values()) / len(report_sections)
    length_score = min(avg_section_length / 500, 1.0)  # Normalize to 500 chars per section
    
    return (completion_rate + length_score) / 2


async def run_complex_research_demo():
    """Run a demonstration of the complex research workflow."""
    print("🚀 Complex Research Workflow Demo")
    print("Demonstrating multi-pattern orchestration for enterprise research")
    print()
    
    # Run comprehensive research
    results = await run_comprehensive_research(
        research_query="artificial intelligence impact on healthcare delivery",
        depth_level="comprehensive",
        validation_required=True
    )
    
    # Display summary
    print("\n📋 Complex Research Summary:")
    print(f"Research Query: {results['research_query']}")
    print(f"Depth Level: {results['depth_level']}")
    print(f"Research Streams: {results['orchestration_phases']['information_gathering']['research_streams']}")
    print(f"Analysis Streams: {results['orchestration_phases']['analysis_synthesis']['analysis_streams']}")
    print(f"Overall Confidence: {results['quality_metrics']['overall_confidence']:.2f}")
    print(f"Report Sections: {len(results['final_outputs']['report_sections'])}")
    print(f"Full Report Length: {len(results['final_outputs']['full_report'])} characters")
    
    return results


if __name__ == "__main__":
    # Run the complex research demo
    print("=" * 90)
    results = asyncio.run(run_complex_research_demo())
    
    print("\n🎯 Key Benefits of Multi-Pattern Orchestration:")
    print("1. Supervisor coordination ensures strategic alignment")
    print("2. Parallel execution maximizes research efficiency")
    print("3. Pipeline processing maintains quality progression")
    print("4. Reflective validation ensures accuracy and completeness")
    print("5. Enterprise-grade outputs suitable for decision making")
    print("6. Scalable architecture adapts to complexity requirements")
    print("7. Quality metrics provide confidence assessment")