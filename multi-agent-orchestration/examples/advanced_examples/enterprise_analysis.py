"""
Enterprise Analysis Workflow - Business Intelligence Integration

Demonstrates enterprise-grade analysis workflow for strategic business decisions.
Combines multiple data sources, analytical perspectives, and validation processes
to deliver executive-ready insights.

This example showcases how multi-agent orchestration can support
complex business analysis and strategic planning scenarios.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.supervisor_agent import SupervisorAgent


async def run_enterprise_analysis(
    business_context: str,
    analysis_scope: List[str],
    stakeholder_requirements: Dict[str, Any],
    timeline: str = "strategic"
) -> Dict[str, Any]:
    """
    Execute comprehensive enterprise analysis workflow.
    
    Workflow Architecture:
    1. Strategic Planning & Scoping (Supervisor)
    2. Multi-Source Data Collection (Parallel)
    3. Specialized Analysis Streams (Parallel + Pipeline)
    4. Cross-Functional Integration (Supervisor)
    5. Executive Synthesis & Recommendations (Pipeline)
    6. Stakeholder Validation (Reflective)
    
    Args:
        business_context: Business domain or market context
        analysis_scope: List of analysis dimensions (financial, market, operational, etc.)
        stakeholder_requirements: Requirements from different stakeholder groups
        timeline: "tactical", "strategic", or "transformational"
        
    Returns:
        Enterprise-grade analysis results with executive summary
    """
    print(f"🏢 Starting Enterprise Analysis Workflow")
    print(f"📊 Context: {business_context}")
    print(f"🎯 Scope: {', '.join(analysis_scope)}")
    print(f"⏱️  Timeline: {timeline}")
    print("=" * 80)
    
    # Initialize enterprise agent teams
    strategic_supervisor = SupervisorAgent()
    market_research_team = [ResearchAgent() for _ in range(2)]
    analytical_team = [AnalysisAgent() for _ in range(3)]
    executive_synthesizer = SummaryAgent()
    
    # Phase 1: Strategic Planning & Analysis Scoping
    print("🎯 Phase 1: Strategic Planning & Analysis Architecture")
    print("-" * 70)
    
    strategic_planning_task = {
        "type": "enterprise_analysis_planning",
        "description": f"Plan comprehensive enterprise analysis for {business_context}",
        "business_context": business_context,
        "analysis_dimensions": analysis_scope,
        "stakeholder_matrix": stakeholder_requirements,
        "timeline_horizon": timeline,
        "deliverable_requirements": {
            "executive_summary": "C-level ready insights",
            "strategic_recommendations": "Actionable business decisions",
            "risk_assessment": "Comprehensive risk analysis",
            "financial_implications": "ROI and investment analysis",
            "implementation_roadmap": "Phased execution plan"
        },
        "analytical_framework": determine_analytical_framework(analysis_scope, timeline)
    }
    
    strategic_plan = await strategic_supervisor.process_task(strategic_planning_task)
    print(f"✅ Strategic planning completed with confidence: {strategic_plan.confidence:.2f}")
    print(f"📋 Analysis architecture: {strategic_plan.content[:250]}...")
    print()
    
    # Phase 2: Multi-Source Intelligence Gathering
    print("📊 Phase 2: Multi-Source Business Intelligence Collection")
    print("-" * 70)
    
    # Create specialized intelligence gathering tasks
    intelligence_streams = create_intelligence_streams(
        business_context, analysis_scope, strategic_plan.content, timeline
    )
    
    print(f"🔀 Executing {len(intelligence_streams)} intelligence gathering streams...")
    
    # Execute intelligence gathering in parallel
    intelligence_tasks = []
    for i, (agent, stream) in enumerate(zip(market_research_team, intelligence_streams[:len(market_research_team)])):
        print(f"📡 Stream {i+1}: {stream['intelligence_type']}")
        intelligence_tasks.append(agent.process_task(stream))
    
    intelligence_results = await asyncio.gather(*intelligence_tasks, return_exceptions=True)
    
    print(f"✅ Intelligence gathering completed")
    successful_streams = [r for r in intelligence_results if not isinstance(r, Exception)]
    print(f"📈 {len(successful_streams)} successful intelligence streams")
    for i, result in enumerate(successful_streams):
        print(f"   Stream {i+1}: {result.confidence:.2f} confidence")
    print()
    
    # Phase 3: Specialized Analysis Execution
    print("🔍 Phase 3: Multi-Dimensional Analysis Execution")
    print("-" * 70)
    
    # Create specialized analysis tasks for each dimension
    analysis_dimensions = create_analysis_dimensions(
        analysis_scope, intelligence_results, business_context, stakeholder_requirements
    )
    
    print(f"🔀 Executing {len(analysis_dimensions)} specialized analyses...")
    
    # Execute analyses in parallel
    analysis_tasks = []
    for i, (agent, dimension) in enumerate(zip(analytical_team, analysis_dimensions)):
        print(f"📊 Analysis {i+1}: {dimension['dimension_name']}")
        analysis_tasks.append(agent.process_task(dimension))
    
    analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
    
    print(f"✅ Specialized analyses completed")
    successful_analyses = [a for a in analysis_results if not isinstance(a, Exception)]
    print(f"📊 {len(successful_analyses)} successful analyses")
    for i, result in enumerate(successful_analyses):
        print(f"   Analysis {i+1}: {result.confidence:.2f} confidence")
    print()
    
    # Phase 4: Cross-Functional Integration & Synthesis
    print("🎯 Phase 4: Strategic Integration & Cross-Functional Synthesis")
    print("-" * 70)
    
    integration_task = {
        "type": "enterprise_integration",
        "description": f"Integrate all analytical perspectives for {business_context}",
        "strategic_framework": strategic_plan.content,
        "intelligence_inputs": [r.content for r in successful_streams],
        "analytical_inputs": [a.content for a in successful_analyses],
        "business_context": business_context,
        "stakeholder_priorities": stakeholder_requirements,
        "integration_objectives": [
            "Synthesize cross-functional insights",
            "Identify strategic opportunities and threats",
            "Assess resource requirements and constraints",
            "Evaluate implementation feasibility",
            "Quantify business impact and ROI",
            "Develop strategic recommendations"
        ]
    }
    
    integrated_analysis = await strategic_supervisor.process_task(integration_task)
    print(f"✅ Strategic integration completed with confidence: {integrated_analysis.confidence:.2f}")
    print(f"🎯 Integration length: {len(integrated_analysis.content)} characters")
    print()
    
    # Phase 5: Executive Synthesis & Strategic Recommendations
    print("👔 Phase 5: Executive Synthesis & Strategic Recommendations")
    print("-" * 70)
    
    # Sequential synthesis pipeline for executive deliverables
    executive_deliverables = {}
    
    deliverable_types = [
        "executive_summary",
        "strategic_recommendations", 
        "risk_assessment",
        "financial_analysis",
        "implementation_roadmap",
        "success_metrics"
    ]
    
    for deliverable in deliverable_types:
        synthesis_task = {
            "type": "executive_deliverable",
            "description": f"Create executive {deliverable} for {business_context}",
            "deliverable_type": deliverable,
            "integrated_analysis": integrated_analysis.content,
            "stakeholder_audience": get_stakeholder_audience(deliverable, stakeholder_requirements),
            "formatting_requirements": get_executive_formatting(deliverable),
            "business_context": business_context,
            "previous_deliverables": executive_deliverables
        }
        
        deliverable_result = await executive_synthesizer.process_task(synthesis_task)
        executive_deliverables[deliverable] = {
            "content": deliverable_result.content,
            "confidence": deliverable_result.confidence,
            "metadata": deliverable_result.metadata
        }
        
        print(f"✅ {deliverable.replace('_', ' ').title()} completed")
    
    print()
    
    # Phase 6: Stakeholder Validation & Refinement
    print("👥 Phase 6: Stakeholder Validation & Quality Assurance")
    print("-" * 70)
    
    validation_cycles = []
    
    for stakeholder_group, requirements in stakeholder_requirements.items():
        print(f"🔍 Validating for {stakeholder_group}...")
        
        validation_task = {
            "type": "stakeholder_validation",
            "description": f"Validate analysis deliverables for {stakeholder_group}",
            "stakeholder_group": stakeholder_group,
            "stakeholder_requirements": requirements,
            "deliverables_to_validate": executive_deliverables,
            "validation_criteria": [
                "Relevance to stakeholder needs",
                "Actionability of recommendations", 
                "Clarity and comprehensiveness",
                "Risk assessment accuracy",
                "Implementation feasibility",
                "ROI and value proposition"
            ]
        }
        
        validation_result = await analytical_team[0].process_task(validation_task)
        
        validation_cycles.append({
            "stakeholder_group": stakeholder_group,
            "validation_confidence": validation_result.confidence,
            "validation_feedback": validation_result.content,
            "requirements_met": assess_requirements_compliance(validation_result.content)
        })
        
        print(f"   {stakeholder_group}: {validation_result.confidence:.2f} validation confidence")
    
    print("✅ Stakeholder validation completed")
    print()
    
    # Compile comprehensive enterprise results
    enterprise_results = {
        "workflow_type": "enterprise_analysis",
        "business_context": business_context,
        "analysis_scope": analysis_scope,
        "timeline_horizon": timeline,
        "execution_timestamp": datetime.now().isoformat(),
        "orchestration_phases": {
            "strategic_planning": {
                "framework": strategic_plan.content,
                "planning_confidence": strategic_plan.confidence,
                "analysis_architecture": f"{len(analysis_scope)} dimensions planned"
            },
            "intelligence_gathering": {
                "streams_executed": len(intelligence_streams),
                "successful_streams": len(successful_streams),
                "intelligence_confidence": sum(r.confidence for r in successful_streams) / max(len(successful_streams), 1)
            },
            "analytical_execution": {
                "dimensions_analyzed": len(analysis_dimensions),
                "successful_analyses": len(successful_analyses),
                "analytical_confidence": sum(a.confidence for a in successful_analyses) / max(len(successful_analyses), 1)
            },
            "strategic_integration": {
                "integration_confidence": integrated_analysis.confidence,
                "synthesis_complexity": len(integrated_analysis.content)
            },
            "executive_synthesis": {
                "deliverables_created": list(executive_deliverables.keys()),
                "average_deliverable_confidence": sum(
                    d["confidence"] for d in executive_deliverables.values()
                ) / len(executive_deliverables)
            },
            "stakeholder_validation": {
                "stakeholder_groups": list(stakeholder_requirements.keys()),
                "validation_cycles": validation_cycles,
                "overall_validation_confidence": sum(
                    cycle["validation_confidence"] for cycle in validation_cycles
                ) / max(len(validation_cycles), 1)
            }
        },
        "executive_deliverables": executive_deliverables,
        "integrated_analysis": integrated_analysis.content,
        "business_intelligence": {
            "intelligence_sources": len(successful_streams),
            "analytical_dimensions": len(successful_analyses),
            "cross_functional_integration": "Strategic supervisor coordination",
            "stakeholder_alignment": f"{len(validation_cycles)} stakeholder groups validated"
        },
        "enterprise_metrics": {
            "overall_confidence": calculate_enterprise_confidence(
                strategic_plan, successful_streams, successful_analyses, 
                integrated_analysis, executive_deliverables, validation_cycles
            ),
            "analytical_comprehensiveness": len(analysis_scope) / 6.0,  # Normalized to common 6 dimensions
            "stakeholder_satisfaction": sum(
                cycle["validation_confidence"] for cycle in validation_cycles
            ) / max(len(validation_cycles), 1),
            "implementation_readiness": assess_implementation_readiness(executive_deliverables),
            "strategic_value": "High" if len(successful_analyses) >= 3 else "Medium"
        }
    }
    
    print("🎉 Enterprise Analysis Workflow Completed Successfully!")
    print(f"📊 Overall Confidence: {enterprise_results['enterprise_metrics']['overall_confidence']:.2f}")
    print(f"👥 Stakeholder Groups: {len(validation_cycles)} validated")
    print(f"📋 Executive Deliverables: {len(executive_deliverables)} created")
    print(f"🎯 Implementation Readiness: {enterprise_results['enterprise_metrics']['implementation_readiness']}")
    print("=" * 80)
    
    return enterprise_results


def determine_analytical_framework(scope: List[str], timeline: str) -> Dict[str, Any]:
    """Determine the analytical framework based on scope and timeline."""
    frameworks = {
        "tactical": "Operational efficiency and short-term optimization",
        "strategic": "Market positioning and competitive advantage",
        "transformational": "Digital transformation and business model innovation"
    }
    
    return {
        "primary_framework": frameworks.get(timeline, frameworks["strategic"]),
        "analysis_depth": "comprehensive" if timeline == "transformational" else "focused",
        "stakeholder_engagement": "executive" if timeline in ["strategic", "transformational"] else "operational",
        "timeline_considerations": f"{timeline} planning horizon with appropriate depth"
    }


def create_intelligence_streams(context: str, scope: List[str], plan: str, timeline: str) -> List[Dict[str, Any]]:
    """Create specialized business intelligence gathering streams."""
    return [
        {
            "type": "market_intelligence",
            "description": f"Gather market intelligence for {context}",
            "intelligence_type": "Market & Competitive Intelligence",
            "focus_areas": ["market_size", "growth_trends", "competitive_landscape", "customer_segments"],
            "timeline_focus": timeline,
            "data_sources": ["industry_reports", "market_research", "competitive_analysis", "customer_insights"]
        },
        {
            "type": "operational_intelligence",
            "description": f"Collect operational and performance intelligence for {context}",
            "intelligence_type": "Operational & Performance Intelligence", 
            "focus_areas": ["operational_metrics", "cost_structure", "resource_utilization", "process_efficiency"],
            "timeline_focus": timeline,
            "data_sources": ["internal_metrics", "benchmarking", "process_analysis", "resource_assessment"]
        }
    ]


def create_analysis_dimensions(scope: List[str], intelligence: List[Any], context: str, stakeholders: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create specialized analysis dimensions based on scope and intelligence."""
    dimensions = []
    
    if "financial" in scope:
        dimensions.append({
            "type": "financial_analysis",
            "description": f"Comprehensive financial analysis for {context}",
            "dimension_name": "Financial Performance & Projections",
            "intelligence_data": [r.content for r in intelligence if not isinstance(r, Exception)],
            "analysis_focus": ["revenue_analysis", "cost_optimization", "roi_projections", "financial_risks"],
            "stakeholder_priorities": stakeholders.get("financial_team", {})
        })
    
    if "market" in scope:
        dimensions.append({
            "type": "market_analysis", 
            "description": f"Strategic market analysis for {context}",
            "dimension_name": "Market Position & Opportunities",
            "intelligence_data": [r.content for r in intelligence if not isinstance(r, Exception)],
            "analysis_focus": ["market_positioning", "growth_opportunities", "competitive_threats", "customer_value"],
            "stakeholder_priorities": stakeholders.get("marketing_team", {})
        })
    
    if "operational" in scope:
        dimensions.append({
            "type": "operational_analysis",
            "description": f"Operational excellence analysis for {context}",
            "dimension_name": "Operational Efficiency & Optimization",
            "intelligence_data": [r.content for r in intelligence if not isinstance(r, Exception)],
            "analysis_focus": ["process_optimization", "resource_efficiency", "scalability_assessment", "operational_risks"],
            "stakeholder_priorities": stakeholders.get("operations_team", {})
        })
    
    return dimensions


def get_stakeholder_audience(deliverable: str, stakeholders: Dict[str, Any]) -> str:
    """Determine primary stakeholder audience for each deliverable."""
    audiences = {
        "executive_summary": "C-level executives and board members",
        "strategic_recommendations": "Executive leadership and strategy team",
        "risk_assessment": "Risk management and executive team",
        "financial_analysis": "CFO, finance team, and investors",
        "implementation_roadmap": "Project management and operational leads",
        "success_metrics": "Performance management and executive team"
    }
    return audiences.get(deliverable, "Executive stakeholders")


def get_executive_formatting(deliverable: str) -> Dict[str, str]:
    """Get executive formatting requirements for deliverables."""
    formats = {
        "executive_summary": {"length": "1-2 pages", "style": "high-level strategic overview"},
        "strategic_recommendations": {"length": "bulleted priorities", "style": "actionable decisions"},
        "risk_assessment": {"length": "structured analysis", "style": "risk matrix with mitigation"},
        "financial_analysis": {"length": "detailed metrics", "style": "quantitative with projections"},
        "implementation_roadmap": {"length": "phased timeline", "style": "milestone-based execution"},
        "success_metrics": {"length": "KPI dashboard", "style": "measurable outcomes"}
    }
    return formats.get(deliverable, {"length": "comprehensive", "style": "executive-ready"})


def assess_requirements_compliance(validation_feedback: str) -> float:
    """Assess how well deliverables meet stakeholder requirements."""
    # Simulate requirements compliance assessment
    # In production, this would analyze the validation feedback
    return 0.85  # High compliance score


def calculate_enterprise_confidence(plan, intelligence, analyses, integration, deliverables, validations) -> float:
    """Calculate overall enterprise analysis confidence."""
    plan_conf = plan.confidence
    intel_conf = sum(r.confidence for r in intelligence) / max(len(intelligence), 1)
    analysis_conf = sum(a.confidence for a in analyses) / max(len(analyses), 1) 
    integration_conf = integration.confidence
    deliverable_conf = sum(d["confidence"] for d in deliverables.values()) / len(deliverables)
    validation_conf = sum(v["validation_confidence"] for v in validations) / max(len(validations), 1)
    
    return (plan_conf + intel_conf + analysis_conf + integration_conf + deliverable_conf + validation_conf) / 6


def assess_implementation_readiness(deliverables: Dict[str, Any]) -> str:
    """Assess implementation readiness based on deliverable quality."""
    avg_confidence = sum(d["confidence"] for d in deliverables.values()) / len(deliverables)
    
    if avg_confidence >= 0.85:
        return "High - Ready for implementation"
    elif avg_confidence >= 0.75:
        return "Medium - Minor refinements needed"
    else:
        return "Low - Significant improvements required"


async def run_enterprise_analysis_demo():
    """Run a demonstration of the enterprise analysis workflow."""
    print("🚀 Enterprise Analysis Workflow Demo")
    print("Demonstrating comprehensive business intelligence and strategic analysis")
    print()
    
    # Define enterprise analysis scenario
    business_context = "digital transformation initiative for retail operations"
    analysis_scope = ["financial", "market", "operational"]
    stakeholder_requirements = {
        "executive_team": {"focus": "strategic_value", "timeline": "quarterly_updates"},
        "financial_team": {"focus": "roi_analysis", "timeline": "monthly_metrics"},
        "operations_team": {"focus": "implementation_feasibility", "timeline": "weekly_progress"}
    }
    
    # Run enterprise analysis
    results = await run_enterprise_analysis(
        business_context=business_context,
        analysis_scope=analysis_scope,
        stakeholder_requirements=stakeholder_requirements,
        timeline="strategic"
    )
    
    # Display comprehensive summary
    print("\n📋 Enterprise Analysis Summary:")
    print(f"Business Context: {results['business_context']}")
    print(f"Analysis Scope: {', '.join(results['analysis_scope'])}")
    print(f"Timeline Horizon: {results['timeline_horizon']}")
    print(f"Intelligence Streams: {results['orchestration_phases']['intelligence_gathering']['streams_executed']}")
    print(f"Analysis Dimensions: {results['orchestration_phases']['analytical_execution']['dimensions_analyzed']}")
    print(f"Executive Deliverables: {len(results['executive_deliverables'])}")
    print(f"Stakeholder Groups: {len(results['orchestration_phases']['stakeholder_validation']['stakeholder_groups'])}")
    print(f"Overall Confidence: {results['enterprise_metrics']['overall_confidence']:.2f}")
    print(f"Implementation Readiness: {results['enterprise_metrics']['implementation_readiness']}")
    
    return results


if __name__ == "__main__":
    # Run the enterprise analysis demo
    print("=" * 90)
    results = asyncio.run(run_enterprise_analysis_demo())
    
    print("\n🎯 Enterprise Analysis Benefits:")
    print("1. Multi-dimensional business intelligence integration")
    print("2. Strategic planning with executive-grade deliverables")
    print("3. Cross-functional coordination and synthesis")
    print("4. Stakeholder-specific validation and refinement")
    print("5. Implementation-ready strategic recommendations")
    print("6. Comprehensive risk assessment and mitigation")
    print("7. ROI analysis and financial impact quantification")
    print("8. Scalable framework adaptable to various business contexts")