"""
Simple Parallel Example - Market Analysis

Demonstrates concurrent agent execution for market analysis:
Multiple agents analyze different aspects simultaneously, then results are fused.

This example shows how parallel processing can speed up analysis
while capturing diverse perspectives.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List

from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.summary_agent import SummaryAgent


async def run_market_analysis_parallel(company: str = "Tesla") -> Dict[str, Any]:
    """
    Run parallel market analysis using multiple specialized agents.
    
    Agents work simultaneously on different aspects:
    - Research Agent: Company background and industry
    - Analysis Agent: Financial and competitive analysis  
    - Summary Agent: Market sentiment and positioning
    
    Args:
        company: Company to analyze
        
    Returns:
        Fused analysis results from all agents
    """
    print(f"⚡ Starting Parallel Market Analysis for: {company}")
    print("=" * 60)
    
    # Initialize specialized agents
    research_agent = ResearchAgent()
    analysis_agent = AnalysisAgent()  
    summary_agent = SummaryAgent()
    
    # Prepare parallel tasks for each agent
    print("🔀 Preparing Parallel Analysis Tasks...")
    
    # Research Agent Task: Company Background
    research_task = {
        "type": "research",
        "description": f"Research {company} company background and industry context",
        "focus_areas": [
            "Company history and leadership",
            "Core business segments",
            "Industry position and market share",
            "Recent developments and news"
        ],
        "perspective": "company_background"
    }
    
    # Analysis Agent Task: Financial & Competitive Analysis
    analysis_task = {
        "type": "analysis", 
        "description": f"Analyze {company} financial performance and competitive position",
        "focus_areas": [
            "Financial metrics and trends",
            "Competitive advantages and threats",
            "Market opportunities and risks",
            "Strategic positioning"
        ],
        "perspective": "financial_competitive"
    }
    
    # Summary Agent Task: Market Sentiment
    sentiment_task = {
        "type": "market_sentiment",
        "description": f"Analyze market sentiment and positioning for {company}",
        "focus_areas": [
            "Investor sentiment and analyst opinions",
            "Public perception and brand strength",
            "Market trends affecting the company",
            "Future outlook and expectations"
        ],
        "perspective": "market_sentiment"
    }
    
    # Execute all tasks in parallel
    print("🚀 Executing Parallel Analysis...")
    start_time = datetime.now()
    
    # Run all agents concurrently
    research_task_coro = research_agent.process_task(research_task)
    analysis_task_coro = analysis_agent.process_task(analysis_task)
    sentiment_task_coro = summary_agent.process_task(sentiment_task)
    
    # Wait for all to complete
    results = await asyncio.gather(
        research_task_coro,
        analysis_task_coro, 
        sentiment_task_coro,
        return_exceptions=True
    )
    
    end_time = datetime.now()
    execution_duration = (end_time - start_time).total_seconds()
    
    # Process results
    research_result, analysis_result, sentiment_result = results
    
    print(f"✅ All parallel analyses completed in {execution_duration:.2f} seconds")
    print()
    
    # Display individual results
    print("📚 Research Agent Results:")
    print(f"   Confidence: {research_result.confidence:.2f}")
    print(f"   Content: {research_result.content[:150]}...")
    print()
    
    print("🔍 Analysis Agent Results:")
    print(f"   Confidence: {analysis_result.confidence:.2f}")
    print(f"   Content: {analysis_result.content[:150]}...")
    print()
    
    print("📊 Sentiment Agent Results:")
    print(f"   Confidence: {sentiment_result.confidence:.2f}")
    print(f"   Content: {sentiment_result.content[:150]}...")
    print()
    
    # Fuse results into comprehensive analysis
    print("🔗 Fusing Parallel Results...")
    
    fused_analysis = f\"\"\"\
COMPREHENSIVE MARKET ANALYSIS: {company}

=== COMPANY BACKGROUND & INDUSTRY CONTEXT ===
{research_result.content}

=== FINANCIAL & COMPETITIVE ANALYSIS ===
{analysis_result.content}

=== MARKET SENTIMENT & POSITIONING ===
{sentiment_result.content}

=== INTEGRATED INSIGHTS ===
This analysis combines three parallel perspectives to provide a comprehensive view of {company}'s market position. 
The research perspective provides foundational context, the analytical perspective offers quantitative insights, 
and the sentiment perspective captures market dynamics and perceptions.
\"\"\"
    
    # Calculate fusion metrics
    individual_confidences = [
        research_result.confidence,
        analysis_result.confidence,
        sentiment_result.confidence
    ]
    
    avg_confidence = sum(individual_confidences) / len(individual_confidences)
    min_confidence = min(individual_confidences)
    max_confidence = max(individual_confidences)
    
    # Compile comprehensive results
    parallel_results = {
        "workflow_type": "parallel",
        "company": company,
        "execution_time": start_time.isoformat(),
        "execution_duration_seconds": execution_duration,
        "individual_results": {
            "research_background": {
                "agent": "research_agent",
                "perspective": "company_background",
                "confidence": research_result.confidence,
                "content": research_result.content,
                "metadata": research_result.metadata
            },
            "financial_competitive": {
                "agent": "analysis_agent",
                "perspective": "financial_competitive", 
                "confidence": analysis_result.confidence,
                "content": analysis_result.content,
                "metadata": analysis_result.metadata
            },
            "market_sentiment": {
                "agent": "summary_agent",
                "perspective": "market_sentiment",
                "confidence": sentiment_result.confidence,
                "content": sentiment_result.content,
                "metadata": sentiment_result.metadata
            }
        },
        "fused_analysis": fused_analysis,
        "fusion_metrics": {
            "agents_count": 3,
            "successful_agents": 3,
            "average_confidence": avg_confidence,
            "confidence_range": {
                "min": min_confidence,
                "max": max_confidence,
                "spread": max_confidence - min_confidence
            },
            "parallel_efficiency": f"3 agents completed in {execution_duration:.2f}s"
        }
    }
    
    print("🎉 Parallel Analysis Completed Successfully!")
    print(f"📈 Average Confidence: {avg_confidence:.2f}")
    print(f"⚡ Parallel Efficiency: 3 agents in {execution_duration:.2f} seconds")
    print("=" * 60)
    
    return parallel_results


async def run_simple_parallel_demo():
    """Run a simple demo of the parallel pattern."""
    print("🚀 Multi-Agent Parallel Pattern Demo")
    print("Demonstrating concurrent agent execution and result fusion")
    print()
    
    # Run parallel market analysis
    results = await run_market_analysis_parallel("Apple Inc.")
    
    # Display summary
    print("\n📋 Parallel Analysis Summary:")
    print(f"Company: {results['company']}")
    print(f"Agents Executed: {results['fusion_metrics']['agents_count']}")
    print(f"Execution Time: {results['execution_duration_seconds']:.2f} seconds")
    print(f"Average Confidence: {results['fusion_metrics']['average_confidence']:.2f}")
    print(f"Confidence Range: {results['fusion_metrics']['confidence_range']['min']:.2f} - {results['fusion_metrics']['confidence_range']['max']:.2f}")
    print(f"Final Analysis Length: {len(results['fused_analysis'])} characters")
    
    return results


async def compare_parallel_vs_sequential():
    """Compare parallel vs sequential execution performance."""
    print("\n🔬 Performance Comparison: Parallel vs Sequential")
    print("=" * 50)
    
    company = "Microsoft"
    
    # Test parallel execution
    print("⚡ Testing Parallel Execution...")
    parallel_start = datetime.now()
    parallel_results = await run_market_analysis_parallel(company)
    parallel_duration = parallel_results['execution_duration_seconds']
    
    # Simulate sequential execution timing
    print("\n🔄 Simulating Sequential Execution...")
    sequential_duration = sum([
        result['metadata'].get('processing_time', 0.5) 
        for result in parallel_results['individual_results'].values()
    ])
    
    # Calculate efficiency gains
    time_saved = sequential_duration - parallel_duration
    efficiency_gain = (time_saved / sequential_duration) * 100 if sequential_duration > 0 else 0
    
    print(f"\n📊 Performance Results:")
    print(f"Parallel Execution:   {parallel_duration:.2f} seconds")
    print(f"Sequential (Est.):    {sequential_duration:.2f} seconds")
    print(f"Time Saved:           {time_saved:.2f} seconds")
    print(f"Efficiency Gain:      {efficiency_gain:.1f}%")
    
    return {
        "parallel_duration": parallel_duration,
        "sequential_duration": sequential_duration,
        "time_saved": time_saved,
        "efficiency_gain": efficiency_gain
    }


if __name__ == "__main__":
    # Run the demos
    print("=" * 70)
    results = asyncio.run(run_simple_parallel_demo())
    
    # Run performance comparison
    performance = asyncio.run(compare_parallel_vs_sequential())
    
    print("\n🎯 Key Takeaways:")
    print("1. Parallel pattern enables simultaneous agent execution")
    print("2. Multiple perspectives provide comprehensive analysis")
    print("3. Result fusion combines diverse viewpoints")
    print("4. Significant performance improvements through concurrency")
    print(f"5. {performance['efficiency_gain']:.1f}% faster than sequential execution")
    print("6. Ideal for independent analysis tasks that can be parallelized")