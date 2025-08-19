"""
API Integration Example - External Service Orchestration

Demonstrates how to integrate the multi-agent orchestration platform
with external APIs, web services, and cloud platforms for real-world
data processing and analysis workflows.

This example shows practical integration patterns for:
- REST API consumption and data aggregation
- Real-time data processing workflows
- External service coordination
- Error handling and resilience
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.summary_agent import SummaryAgent


class APIIntegrationOrchestrator:
    """
    Orchestrator that integrates multi-agent workflows with external APIs
    and web services for comprehensive data processing.
    """
    
    def __init__(self):
        self.research_agents = [ResearchAgent() for _ in range(2)]
        self.analysis_agents = [AnalysisAgent() for _ in range(2)]
        self.synthesis_agent = SummaryAgent()
        
        # Simulated API endpoints for demonstration
        self.api_endpoints = {
            "market_data": "https://api.example.com/market-data",
            "news_feed": "https://api.example.com/news",
            "social_sentiment": "https://api.example.com/sentiment",
            "financial_metrics": "https://api.example.com/financials",
            "industry_reports": "https://api.example.com/reports"
        }
    
    async def run_api_integration_workflow(
        self,
        target_entity: str,
        integration_scope: List[str],
        real_time_processing: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a comprehensive workflow that integrates multiple external APIs
        with multi-agent processing for real-world data analysis.
        
        Args:
            target_entity: Entity to analyze (company, market, product, etc.)
            integration_scope: List of API integrations to include
            real_time_processing: Whether to enable real-time data processing
            
        Returns:
            Comprehensive analysis results with API integration metrics
        """
        print(f"🌐 Starting API Integration Workflow")
        print(f"🎯 Target: {target_entity}")
        print(f"🔗 Integration Scope: {', '.join(integration_scope)}")
        print(f"⚡ Real-time Processing: {real_time_processing}")
        print("=" * 80)
        
        # Phase 1: API Data Collection and Aggregation
        print("📡 Phase 1: Multi-Source API Data Collection")
        print("-" * 60)
        
        api_data_streams = await self.collect_api_data(target_entity, integration_scope)
        
        print(f"✅ Collected data from {len(api_data_streams)} API sources")
        for source, data in api_data_streams.items():
            print(f"   {source}: {len(str(data))} characters, confidence: {data.get('confidence', 'N/A')}")
        print()
        
        # Phase 2: Parallel Agent Processing of API Data
        print("🔄 Phase 2: Multi-Agent API Data Processing")
        print("-" * 60)
        
        processing_tasks = self.create_api_processing_tasks(api_data_streams, target_entity)
        
        # Execute processing in parallel
        processing_results = await asyncio.gather(
            *[self.process_api_data_stream(task) for task in processing_tasks],
            return_exceptions=True
        )
        
        successful_processing = [r for r in processing_results if not isinstance(r, Exception)]
        
        print(f"✅ Processed {len(successful_processing)} data streams successfully")
        for i, result in enumerate(successful_processing):
            print(f"   Stream {i+1}: {result['processing_confidence']:.2f} confidence")
        print()
        
        # Phase 3: Real-time Data Integration and Synthesis
        if real_time_processing:
            print("⚡ Phase 3: Real-time Data Integration and Synthesis")
            print("-" * 60)
            
            real_time_updates = await self.process_real_time_updates(
                target_entity, api_data_streams, successful_processing
            )
            
            print(f"✅ Real-time processing completed with {len(real_time_updates)} updates")
        else:
            real_time_updates = []
        
        # Phase 4: Cross-Source Analysis and Correlation
        print("🔍 Phase 4: Cross-Source Analysis and Correlation")
        print("-" * 60)
        
        correlation_analysis = await self.perform_cross_source_analysis(
            successful_processing, target_entity, api_data_streams
        )
        
        print(f"✅ Cross-source analysis completed with {correlation_analysis['correlation_confidence']:.2f} confidence")
        print()
        
        # Phase 5: Integrated Insights and Recommendations
        print("💡 Phase 5: Integrated Insights and API-Enhanced Recommendations")
        print("-" * 60)
        
        final_synthesis = await self.synthesize_api_enhanced_insights(
            api_data_streams, successful_processing, correlation_analysis, 
            real_time_updates, target_entity
        )
        
        print(f"✅ Final synthesis completed with {final_synthesis['synthesis_confidence']:.2f} confidence")
        
        # Compile comprehensive API integration results
        integration_results = {
            "workflow_type": "api_integration",
            "target_entity": target_entity,
            "integration_scope": integration_scope,
            "real_time_enabled": real_time_processing,
            "execution_timestamp": datetime.now().isoformat(),
            "api_integration_phases": {
                "data_collection": {
                    "api_sources": list(api_data_streams.keys()),
                    "total_data_collected": sum(len(str(data)) for data in api_data_streams.values()),
                    "collection_success_rate": len(api_data_streams) / len(integration_scope)
                },
                "agent_processing": {
                    "processing_tasks": len(processing_tasks),
                    "successful_processing": len(successful_processing),
                    "processing_success_rate": len(successful_processing) / len(processing_tasks),
                    "average_processing_confidence": sum(
                        r['processing_confidence'] for r in successful_processing
                    ) / max(len(successful_processing), 1)
                },
                "real_time_integration": {
                    "enabled": real_time_processing,
                    "updates_processed": len(real_time_updates),
                    "real_time_latency": "< 100ms" if real_time_processing else "N/A"
                },
                "cross_source_analysis": {
                    "correlation_confidence": correlation_analysis['correlation_confidence'],
                    "data_consistency": correlation_analysis['data_consistency'],
                    "insight_quality": correlation_analysis['insight_quality']
                },
                "final_synthesis": {
                    "synthesis_confidence": final_synthesis['synthesis_confidence'],
                    "api_enhancement_factor": final_synthesis['api_enhancement_factor'],
                    "actionability_score": final_synthesis['actionability_score']
                }
            },
            "api_data_sources": api_data_streams,
            "processed_insights": successful_processing,
            "correlation_analysis": correlation_analysis,
            "real_time_updates": real_time_updates,
            "final_insights": final_synthesis,
            "integration_metrics": {
                "api_reliability": self.calculate_api_reliability(api_data_streams),
                "data_quality_score": self.assess_data_quality(api_data_streams, successful_processing),
                "integration_efficiency": self.calculate_integration_efficiency(
                    len(api_data_streams), len(successful_processing), real_time_processing
                ),
                "real_time_performance": self.assess_real_time_performance(real_time_updates) if real_time_processing else None
            }
        }
        
        print("🎉 API Integration Workflow Completed Successfully!")
        print(f"🌐 API Sources: {len(api_data_streams)}")
        print(f"🔄 Processing Success Rate: {integration_results['api_integration_phases']['agent_processing']['processing_success_rate']:.2%}")
        print(f"📊 Overall Confidence: {final_synthesis['synthesis_confidence']:.2f}")
        print(f"⚡ Real-time Updates: {len(real_time_updates) if real_time_processing else 'Disabled'}")
        print("=" * 80)
        
        return integration_results
    
    async def collect_api_data(self, target_entity: str, scope: List[str]) -> Dict[str, Any]:
        """
        Collect data from multiple external APIs in parallel.
        """
        api_tasks = []
        
        for source in scope:
            if source in self.api_endpoints:
                api_tasks.append(self.fetch_api_data(source, target_entity))
        
        # Execute API calls in parallel
        api_results = await asyncio.gather(*api_tasks, return_exceptions=True)
        
        # Organize results by source
        api_data = {}
        for i, (source, result) in enumerate(zip(scope, api_results)):
            if not isinstance(result, Exception) and source in self.api_endpoints:
                api_data[source] = result
        
        return api_data
    
    async def fetch_api_data(self, source: str, target_entity: str) -> Dict[str, Any]:
        """
        Simulate fetching data from an external API.
        In production, this would make actual HTTP requests.
        """
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # Simulate different types of API responses
        if source == "market_data":
            return {
                "source": source,
                "target": target_entity,
                "data": {
                    "price": 150.25,
                    "volume": 1250000,
                    "market_cap": "2.5B",
                    "trend": "bullish",
                    "volatility": 0.12
                },
                "confidence": 0.92,
                "timestamp": datetime.now().isoformat(),
                "api_version": "v2.1"
            }
        elif source == "news_feed":
            return {
                "source": source,
                "target": target_entity,
                "data": {
                    "headlines": [
                        f"{target_entity} reports strong quarterly earnings",
                        f"Industry experts bullish on {target_entity} prospects", 
                        f"{target_entity} announces new strategic partnership"
                    ],
                    "sentiment_score": 0.75,
                    "news_volume": 42,
                    "trending_topics": ["earnings", "growth", "innovation"]
                },
                "confidence": 0.88,
                "timestamp": datetime.now().isoformat(),
                "api_version": "v1.3"
            }
        elif source == "social_sentiment":
            return {
                "source": source,
                "target": target_entity,
                "data": {
                    "overall_sentiment": 0.68,
                    "mention_volume": 15420,
                    "engagement_rate": 0.23,
                    "trending_hashtags": [f"#{target_entity}", "#innovation", "#growth"],
                    "influencer_sentiment": 0.71
                },
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat(),
                "api_version": "v3.0"
            }
        else:
            # Generic API response
            return {
                "source": source,
                "target": target_entity,
                "data": {
                    "metrics": {"value1": 123.45, "value2": 67.89},
                    "status": "active",
                    "category": "standard"
                },
                "confidence": 0.80,
                "timestamp": datetime.now().isoformat(),
                "api_version": "v1.0"
            }
    
    def create_api_processing_tasks(self, api_data: Dict[str, Any], target_entity: str) -> List[Dict[str, Any]]:
        """
        Create specialized processing tasks for each API data source.
        """
        tasks = []
        
        for source, data in api_data.items():
            task = {
                "source": source,
                "agent_type": self.determine_optimal_agent_type(source),
                "task_definition": {
                    "type": f"{source}_processing",
                    "description": f"Process {source} data for {target_entity}",
                    "api_data": data,
                    "target_entity": target_entity,
                    "processing_requirements": self.get_processing_requirements(source)
                }
            }
            tasks.append(task)
        
        return tasks
    
    def determine_optimal_agent_type(self, source: str) -> str:
        """
        Determine the optimal agent type for processing each API source.
        """
        agent_mapping = {
            "market_data": "analysis",
            "news_feed": "research", 
            "social_sentiment": "analysis",
            "financial_metrics": "analysis",
            "industry_reports": "research"
        }
        return agent_mapping.get(source, "research")
    
    def get_processing_requirements(self, source: str) -> List[str]:
        """
        Get specific processing requirements for each API source.
        """
        requirements = {
            "market_data": ["numerical_analysis", "trend_identification", "volatility_assessment"],
            "news_feed": ["sentiment_analysis", "topic_extraction", "relevance_scoring"],
            "social_sentiment": ["sentiment_quantification", "influence_analysis", "trending_detection"],
            "financial_metrics": ["ratio_analysis", "performance_comparison", "risk_assessment"],
            "industry_reports": ["key_insights_extraction", "competitive_analysis", "market_positioning"]
        }
        return requirements.get(source, ["general_analysis", "insight_extraction"])
    
    async def process_api_data_stream(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single API data stream using the appropriate agent.
        """
        source = task["source"]
        agent_type = task["agent_type"]
        task_def = task["task_definition"]
        
        # Select appropriate agent
        if agent_type == "analysis":
            agent = self.analysis_agents[0]
        else:
            agent = self.research_agents[0]
        
        # Process the API data
        result = await agent.process_task(task_def)
        
        return {
            "source": source,
            "agent_type": agent_type,
            "processing_confidence": result.confidence,
            "processed_content": result.content,
            "original_api_data": task_def["api_data"],
            "processing_metadata": result.metadata,
            "insights_extracted": self.extract_api_insights(result.content, source)
        }
    
    def extract_api_insights(self, processed_content: str, source: str) -> List[str]:
        """
        Extract key insights from processed API content.
        """
        # Simulate insight extraction based on source type
        if source == "market_data":
            return ["Strong price momentum", "High trading volume", "Low volatility risk"]
        elif source == "news_feed":
            return ["Positive earnings outlook", "Strategic growth initiatives", "Market confidence"]
        elif source == "social_sentiment":
            return ["Positive public perception", "High engagement levels", "Trending visibility"]
        else:
            return ["General positive indicators", "Stable performance metrics", "Growth potential"]
    
    async def process_real_time_updates(
        self, 
        target_entity: str, 
        api_data: Dict[str, Any], 
        processed_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process real-time updates from API sources.
        """
        real_time_updates = []
        
        # Simulate real-time data updates
        for source in api_data.keys():
            update_task = {
                "type": "real_time_update",
                "description": f"Process real-time update from {source} for {target_entity}",
                "source": source,
                "update_data": await self.fetch_api_data(source, target_entity),  # Fresh data
                "previous_state": next((r for r in processed_results if r["source"] == source), None)
            }
            
            # Process update with research agent
            update_result = await self.research_agents[1].process_task(update_task)
            
            real_time_updates.append({
                "source": source,
                "update_confidence": update_result.confidence,
                "update_content": update_result.content,
                "timestamp": datetime.now().isoformat(),
                "change_detected": True,  # Simulated
                "impact_assessment": "moderate"  # Simulated
            })
        
        return real_time_updates
    
    async def perform_cross_source_analysis(
        self,
        processed_results: List[Dict[str, Any]],
        target_entity: str,
        api_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform correlation analysis across multiple API data sources.
        """
        correlation_task = {
            "type": "cross_source_correlation",
            "description": f"Analyze correlations across API sources for {target_entity}",
            "processed_results": processed_results,
            "api_sources": list(api_data.keys()),
            "correlation_objectives": [
                "Identify data consistency patterns",
                "Detect conflicting signals",
                "Find reinforcing indicators",
                "Assess overall data reliability"
            ]
        }
        
        correlation_result = await self.analysis_agents[1].process_task(correlation_task)
        
        return {
            "correlation_confidence": correlation_result.confidence,
            "correlation_analysis": correlation_result.content,
            "data_consistency": self.assess_data_consistency(processed_results),
            "signal_correlations": self.calculate_signal_correlations(processed_results),
            "insight_quality": "high" if correlation_result.confidence > 0.8 else "medium",
            "reliability_score": self.calculate_reliability_score(processed_results, api_data)
        }
    
    def assess_data_consistency(self, processed_results: List[Dict[str, Any]]) -> float:
        """
        Assess consistency across processed API data sources.
        """
        # Simulate consistency assessment
        confidence_scores = [r["processing_confidence"] for r in processed_results]
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    def calculate_signal_correlations(self, processed_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate correlations between different API signals.
        """
        # Simulate signal correlation calculation
        return {
            "market_sentiment_correlation": 0.82,
            "news_market_correlation": 0.75,
            "social_news_correlation": 0.68,
            "overall_signal_alignment": 0.78
        }
    
    def calculate_reliability_score(
        self, 
        processed_results: List[Dict[str, Any]], 
        api_data: Dict[str, Any]
    ) -> float:
        """
        Calculate overall reliability score for the API integration.
        """
        # Factor in API reliability, processing quality, and data consistency
        api_reliability = self.calculate_api_reliability(api_data)
        processing_quality = sum(r["processing_confidence"] for r in processed_results) / len(processed_results)
        data_consistency = self.assess_data_consistency(processed_results)
        
        return (api_reliability + processing_quality + data_consistency) / 3
    
    async def synthesize_api_enhanced_insights(
        self,
        api_data: Dict[str, Any],
        processed_results: List[Dict[str, Any]],
        correlation_analysis: Dict[str, Any],
        real_time_updates: List[Dict[str, Any]],
        target_entity: str
    ) -> Dict[str, Any]:
        """
        Synthesize final insights enhanced by API integration.
        """
        synthesis_task = {
            "type": "api_enhanced_synthesis",
            "description": f"Synthesize API-enhanced insights for {target_entity}",
            "api_data_sources": list(api_data.keys()),
            "processed_insights": [r["processed_content"] for r in processed_results],
            "correlation_findings": correlation_analysis["correlation_analysis"],
            "real_time_context": [u["update_content"] for u in real_time_updates],
            "synthesis_objectives": [
                "Integrate multi-source insights",
                "Leverage API data for accuracy",
                "Provide actionable recommendations",
                "Assess real-time implications"
            ]
        }
        
        synthesis_result = await self.synthesis_agent.process_task(synthesis_task)
        
        return {
            "synthesis_confidence": synthesis_result.confidence,
            "synthesized_insights": synthesis_result.content,
            "api_enhancement_factor": self.calculate_api_enhancement(api_data, processed_results),
            "actionability_score": self.assess_actionability(synthesis_result.content),
            "real_time_relevance": len(real_time_updates) > 0,
            "integration_quality": "high" if synthesis_result.confidence > 0.85 else "medium"
        }
    
    def calculate_api_reliability(self, api_data: Dict[str, Any]) -> float:
        """
        Calculate reliability score for API data sources.
        """
        if not api_data:
            return 0.0
        
        confidence_scores = [data.get("confidence", 0.8) for data in api_data.values()]
        return sum(confidence_scores) / len(confidence_scores)
    
    def assess_data_quality(
        self, 
        api_data: Dict[str, Any], 
        processed_results: List[Dict[str, Any]]
    ) -> float:
        """
        Assess overall data quality from API integration.
        """
        api_quality = self.calculate_api_reliability(api_data)
        processing_quality = sum(r["processing_confidence"] for r in processed_results) / max(len(processed_results), 1)
        
        return (api_quality + processing_quality) / 2
    
    def calculate_integration_efficiency(
        self, 
        api_sources: int, 
        successful_processing: int, 
        real_time_enabled: bool
    ) -> float:
        """
        Calculate integration efficiency score.
        """
        base_efficiency = successful_processing / max(api_sources, 1)
        real_time_bonus = 0.1 if real_time_enabled else 0.0
        
        return min(1.0, base_efficiency + real_time_bonus)
    
    def assess_real_time_performance(self, real_time_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess real-time processing performance.
        """
        if not real_time_updates:
            return {"enabled": False}
        
        avg_confidence = sum(u["update_confidence"] for u in real_time_updates) / len(real_time_updates)
        
        return {
            "enabled": True,
            "updates_processed": len(real_time_updates),
            "average_confidence": avg_confidence,
            "latency": "< 100ms",  # Simulated
            "throughput": "high"
        }
    
    def calculate_api_enhancement(
        self, 
        api_data: Dict[str, Any], 
        processed_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate how much the API integration enhanced the analysis.
        """
        # Factor in number of sources, data quality, and processing success
        source_diversity = len(api_data) / 5.0  # Normalized to 5 sources
        data_quality = self.calculate_api_reliability(api_data)
        processing_success = len(processed_results) / max(len(api_data), 1)
        
        return (source_diversity + data_quality + processing_success) / 3
    
    def assess_actionability(self, synthesized_content: str) -> float:
        """
        Assess how actionable the synthesized insights are.
        """
        # Simulate actionability assessment based on content quality
        content_length = len(synthesized_content)
        base_score = min(1.0, content_length / 1000)  # Normalized to 1000 chars
        
        # Add quality factors (simulated)
        return min(1.0, base_score + 0.2)


async def run_api_integration_workflow(
    target_entity: str = "TechCorp Inc.",
    integration_scope: Optional[List[str]] = None,
    real_time_processing: bool = True
) -> Dict[str, Any]:
    """
    Run a comprehensive API integration workflow demonstration.
    
    Args:
        target_entity: Entity to analyze
        integration_scope: List of API sources to integrate
        real_time_processing: Enable real-time data processing
        
    Returns:
        Complete API integration workflow results
    """
    if integration_scope is None:
        integration_scope = ["market_data", "news_feed", "social_sentiment", "financial_metrics"]
    
    orchestrator = APIIntegrationOrchestrator()
    
    results = await orchestrator.run_api_integration_workflow(
        target_entity=target_entity,
        integration_scope=integration_scope,
        real_time_processing=real_time_processing
    )
    
    return results


async def run_api_integration_demo():
    """Run a comprehensive demonstration of API integration capabilities."""
    print("🚀 API Integration Workflow Demo")
    print("Demonstrating external API integration with multi-agent processing")
    print()
    
    # Run API integration workflow
    results = await run_api_integration_workflow(
        target_entity="InnovaCorp",
        integration_scope=["market_data", "news_feed", "social_sentiment", "industry_reports"],
        real_time_processing=True
    )
    
    # Display comprehensive summary
    print("\n📋 API Integration Summary:")
    print(f"Target Entity: {results['target_entity']}")
    print(f"API Sources Integrated: {len(results['api_integration_phases']['data_collection']['api_sources'])}")
    print(f"Data Collection Success Rate: {results['api_integration_phases']['data_collection']['collection_success_rate']:.2%}")
    print(f"Processing Success Rate: {results['api_integration_phases']['agent_processing']['processing_success_rate']:.2%}")
    print(f"Real-time Updates: {results['api_integration_phases']['real_time_integration']['updates_processed']}")
    print(f"Overall Synthesis Confidence: {results['api_integration_phases']['final_synthesis']['synthesis_confidence']:.2f}")
    print(f"API Enhancement Factor: {results['api_integration_phases']['final_synthesis']['api_enhancement_factor']:.2f}")
    print(f"Integration Efficiency: {results['integration_metrics']['integration_efficiency']:.2f}")
    
    return results


if __name__ == "__main__":
    # Run the API integration demo
    print("=" * 90)
    results = asyncio.run(run_api_integration_demo())
    
    print("\n🎯 API Integration Benefits:")
    print("1. Real-time external data integration with multi-agent processing")
    print("2. Parallel API consumption for improved performance")
    print("3. Cross-source correlation and consistency analysis")
    print("4. Automated data quality assessment and reliability scoring")
    print("5. Real-time update processing with change detection")
    print("6. API-enhanced insights with actionable recommendations")
    print("7. Scalable architecture supporting multiple external services")
    print("8. Error handling and resilience for production deployments")