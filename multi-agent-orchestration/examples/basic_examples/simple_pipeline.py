"""
Simple Pipeline Example - Content Creation Workflow

Demonstrates sequential agent collaboration for content creation:
Research → Analysis → Writing → Review

This example shows how agents pass information between stages
and build upon each other's work.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.summary_agent import SummaryAgent
from src.orchestration.workflow_engine import WorkflowEngine


async def run_content_creation_pipeline(topic: str = "artificial intelligence trends") -> Dict[str, Any]:
    """
    Run a content creation pipeline using the Pipeline pattern.
    
    Workflow: Research → Analysis → Summary → Quality Review
    
    Args:
        topic: Topic to create content about
        
    Returns:
        Complete workflow results with all stage outputs
    """
    print(f"🔄 Starting Content Creation Pipeline for: {topic}")
    print("=" * 60)
    
    # Initialize agents
    research_agent = ResearchAgent()
    analysis_agent = AnalysisAgent()
    summary_agent = SummaryAgent()
    
    # Create workflow engine
    workflow_engine = WorkflowEngine()
    
    # Stage 1: Research
    print("📚 Stage 1: Research Phase")
    research_task = {
        "type": "research",
        "description": f"Research comprehensive information about {topic}",
        "requirements": [
            "Current trends and developments",
            "Key industry players and innovations",
            "Market implications and future outlook"
        ],
        "sources": ["academic", "industry", "news"]
    }
    
    research_result = await research_agent.process_task(research_task)
    print(f"✅ Research completed with confidence: {research_result.confidence}")
    print(f"📄 Research findings: {research_result.content[:200]}...")
    print()
    
    # Stage 2: Analysis
    print("🔍 Stage 2: Analysis Phase")
    analysis_task = {
        "type": "analysis",
        "description": f"Analyze research findings about {topic}",
        "input_data": research_result.content,
        "analysis_focus": [
            "Identify key patterns and trends",
            "Assess market opportunities and risks",
            "Determine strategic implications"
        ]
    }
    
    analysis_result = await analysis_agent.process_task(analysis_task)
    print(f"✅ Analysis completed with confidence: {analysis_result.confidence}")
    print(f"📊 Analysis insights: {analysis_result.content[:200]}...")
    print()
    
    # Stage 3: Content Creation
    print("✍️ Stage 3: Content Creation Phase")
    summary_task = {
        "type": "summarization",
        "description": f"Create comprehensive content about {topic}",
        "research_data": research_result.content,
        "analysis_data": analysis_result.content,
        "content_requirements": [
            "Executive summary",
            "Key insights and trends",
            "Strategic recommendations",
            "Future outlook"
        ],
        "target_audience": "business executives and decision makers"
    }
    
    summary_result = await summary_agent.process_task(summary_task)
    print(f"✅ Content creation completed with confidence: {summary_result.confidence}")
    print(f"📝 Content preview: {summary_result.content[:200]}...")
    print()
    
    # Compile final results
    pipeline_results = {
        "workflow_type": "pipeline",
        "topic": topic,
        "execution_time": datetime.now().isoformat(),
        "stages": {
            "research": {
                "agent": "research_agent",
                "confidence": research_result.confidence,
                "content": research_result.content,
                "metadata": research_result.metadata
            },
            "analysis": {
                "agent": "analysis_agent", 
                "confidence": analysis_result.confidence,
                "content": analysis_result.content,
                "metadata": analysis_result.metadata
            },
            "content_creation": {
                "agent": "summary_agent",
                "confidence": summary_result.confidence,
                "content": summary_result.content,
                "metadata": summary_result.metadata
            }
        },
        "final_output": summary_result.content,
        "overall_confidence": (
            research_result.confidence + 
            analysis_result.confidence + 
            summary_result.confidence
        ) / 3,
        "pipeline_metrics": {
            "total_stages": 3,
            "successful_stages": 3,
            "average_confidence": (
                research_result.confidence + 
                analysis_result.confidence + 
                summary_result.confidence
            ) / 3
        }
    }
    
    print("🎉 Pipeline Completed Successfully!")
    print(f"📈 Overall Confidence: {pipeline_results['overall_confidence']:.2f}")
    print("=" * 60)
    
    return pipeline_results


async def run_simple_pipeline_demo():
    """Run a simple demo of the pipeline pattern."""
    print("🚀 Multi-Agent Pipeline Pattern Demo")
    print("Demonstrating sequential agent collaboration")
    print()
    
    # Run content creation pipeline
    results = await run_content_creation_pipeline("sustainable energy technologies")
    
    # Display summary
    print("\n📋 Pipeline Summary:")
    print(f"Topic: {results['topic']}")
    print(f"Stages Completed: {results['pipeline_metrics']['successful_stages']}")
    print(f"Average Confidence: {results['pipeline_metrics']['average_confidence']:.2f}")
    print(f"Final Content Length: {len(results['final_output'])} characters")
    
    return results


if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(run_simple_pipeline_demo())
    
    print("\n🎯 Key Takeaways:")
    print("1. Pipeline pattern enables sequential agent collaboration")
    print("2. Each stage builds upon previous stage outputs")
    print("3. Information flows linearly through the workflow")
    print("4. Quality improves as data passes through specialized agents")
    print("5. Final output represents accumulated intelligence from all stages")