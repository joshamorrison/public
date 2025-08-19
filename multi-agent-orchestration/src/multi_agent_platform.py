"""
Multi-Agent Platform

Main platform interface that brings together all components of the 
multi-agent orchestration system.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

try:
    from .agents import (
        BaseAgent, SupervisorAgent, ResearcherAgent, AnalystAgent, 
        CriticAgent, SynthesizerAgent
    )
    from .patterns import (
        PipelinePattern, SupervisorPattern, ParallelPattern, ReflectivePattern,
        PatternBuilder
    )
    from .orchestration import (
        WorkflowEngine, ResultAggregator, StateManager, FeedbackLoop
    )
except ImportError:
    # Handle direct execution case
    from agents import (
        BaseAgent, SupervisorAgent, ResearcherAgent, AnalystAgent, 
        CriticAgent, SynthesizerAgent
    )
    from patterns import (
        PipelinePattern, SupervisorPattern, ParallelPattern, ReflectivePattern,
        PatternBuilder
    )
    from orchestration import (
        WorkflowEngine, ResultAggregator, StateManager, FeedbackLoop
    )


class MultiAgentPlatform:
    """
    Main platform interface for multi-agent orchestration.
    
    The platform provides:
    - Unified interface for all orchestration patterns
    - Agent management and lifecycle
    - Workflow execution and monitoring
    - Result aggregation and feedback loops
    - Performance analytics and optimization
    """

    def __init__(self, platform_id: str = "multi-agent-platform-001"):
        """
        Initialize the multi-agent platform.
        
        Args:
            platform_id: Unique identifier for this platform instance
        """
        self.platform_id = platform_id
        self.name = "Multi-Agent Orchestration Platform"
        self.version = "1.0.0"
        self.created_at = datetime.now()
        
        # Core components
        self.workflow_engine = WorkflowEngine(f"{platform_id}_engine")
        self.result_aggregator = ResultAggregator()
        self.state_manager = StateManager()
        self.feedback_loop = FeedbackLoop(f"{platform_id}_feedback")
        self.pattern_builder = PatternBuilder()
        
        # Registries
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.active_patterns: Dict[str, Union[PipelinePattern, SupervisorPattern, ParallelPattern, ReflectivePattern]] = {}
        
        # Platform metrics
        self.platform_metrics = {
            "agents_created": 0,
            "patterns_created": 0,
            "workflows_executed": 0,
            "uptime_start": datetime.now()
        }

    # Agent Management
    def create_agent(self, agent_type: str, agent_id: str = None, **kwargs) -> BaseAgent:
        """Create and register a new agent."""
        agent = self.pattern_builder.create_agent(agent_type, agent_id, **kwargs)
        self.registered_agents[agent.agent_id] = agent
        self.platform_metrics["agents_created"] += 1
        
        print(f"[PLATFORM] Created and registered {agent_type} agent: {agent.agent_id}")
        return agent

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get a registered agent by ID."""
        return self.registered_agents.get(agent_id)

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all registered agents."""
        return {
            agent_id: {
                "name": agent.name,
                "type": type(agent).__name__,
                "capabilities": agent.get_capabilities(),
                "performance": agent.performance_metrics
            }
            for agent_id, agent in self.registered_agents.items()
        }

    # Pattern Management
    def create_pattern(self, pattern_type: str, pattern_id: str = None, 
                      **kwargs) -> Union[PipelinePattern, SupervisorPattern, ParallelPattern, ReflectivePattern]:
        """Create and register a new orchestration pattern."""
        if pattern_type == "pipeline":
            pattern = self.pattern_builder.create_pipeline(pattern_id)
        elif pattern_type == "supervisor":
            supervisor_agent = kwargs.get("supervisor_agent")
            pattern = self.pattern_builder.create_supervisor(pattern_id, supervisor_agent)
        elif pattern_type == "parallel":
            pattern = self.pattern_builder.create_parallel(pattern_id)
        elif pattern_type == "reflective":
            primary_agent = kwargs.get("primary_agent")
            pattern = self.pattern_builder.create_reflective(pattern_id, primary_agent)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        self.active_patterns[pattern.pattern_id] = pattern
        self.platform_metrics["patterns_created"] += 1
        
        print(f"[PLATFORM] Created and registered {pattern_type} pattern: {pattern.pattern_id}")
        return pattern

    def get_pattern(self, pattern_id: str) -> Optional[Union[PipelinePattern, SupervisorPattern, ParallelPattern, ReflectivePattern]]:
        """Get an active pattern by ID."""
        return self.active_patterns.get(pattern_id)

    def list_patterns(self) -> Dict[str, Dict[str, Any]]:
        """List all active patterns."""
        return {
            pattern_id: {
                "name": pattern.name,
                "type": type(pattern).__name__,
                "configuration": pattern.get_pattern_configuration()
            }
            for pattern_id, pattern in self.active_patterns.items()
        }

    # Pre-built Pattern Templates
    def create_research_pipeline(self, pipeline_id: str = "research_pipeline") -> PipelinePattern:
        """Create a pre-configured research pipeline."""
        return self.pattern_builder.build_research_pipeline(pipeline_id)

    def create_analysis_supervisor(self, supervisor_id: str = "analysis_supervisor") -> SupervisorPattern:
        """Create a pre-configured analysis supervisor."""
        return self.pattern_builder.build_analysis_supervisor(supervisor_id)

    def create_competitive_analysis_parallel(self, parallel_id: str = "competitive_parallel") -> ParallelPattern:
        """Create a pre-configured competitive analysis parallel pattern."""
        return self.pattern_builder.build_competitive_analysis_parallel(parallel_id)

    def create_content_optimization_reflective(self, reflective_id: str = "content_reflective") -> ReflectivePattern:
        """Create a pre-configured content optimization reflective pattern."""
        return self.pattern_builder.build_content_optimization_reflective(reflective_id)

    # Workflow Execution
    async def execute_pattern(self, pattern_id: str, task: Dict[str, Any], 
                            execution_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a pattern with the specified task."""
        pattern = self.get_pattern(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern not found: {pattern_id}")
        
        # Execute through workflow engine
        result = await self.workflow_engine.execute_pattern(pattern, task, execution_config=execution_config)
        
        self.platform_metrics["workflows_executed"] += 1
        
        # Submit performance feedback
        await self.feedback_loop.submit_performance_feedback(
            source="platform",
            target=pattern_id,
            execution_time=result.get("execution_time", 0),
            success_rate=1.0 if result.get("success") else 0.0,
            confidence=0.8,  # Default platform confidence
            error_count=0 if result.get("success") else 1
        )
        
        return result

    async def execute_workflow_chain(self, workflow_chain: List[Dict[str, Any]], 
                                   initial_task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chain of patterns in sequence."""
        return await self.workflow_engine.execute_workflow_chain(workflow_chain, initial_task)

    # Analytics and Monitoring
    def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status."""
        uptime = (datetime.now() - self.platform_metrics["uptime_start"]).total_seconds()
        
        return {
            "platform_info": {
                "platform_id": self.platform_id,
                "name": self.name,
                "version": self.version,
                "created_at": self.created_at.isoformat(),
                "uptime_seconds": uptime
            },
            "component_status": {
                "workflow_engine": self.workflow_engine.get_execution_metrics(),
                "result_aggregator": self.result_aggregator.get_aggregation_metrics(),
                "state_manager": self.state_manager.get_state_metrics(),
                "feedback_loop": self.feedback_loop.get_feedback_metrics()
            },
            "registry_status": {
                "registered_agents": len(self.registered_agents),
                "active_patterns": len(self.active_patterns),
                "pattern_builder_status": self.pattern_builder.get_builder_status()
            },
            "platform_metrics": self.platform_metrics.copy()
        }

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics across the platform."""
        # Aggregate metrics from all components
        workflow_metrics = self.workflow_engine.get_execution_metrics()
        # Note: This would need to be await in an async context
        feedback_summary = {"message": "Feedback summary not available in sync context"}
        
        # Calculate platform-wide performance indicators
        overall_success_rate = workflow_metrics.get("success_rate", 0)
        average_execution_time = workflow_metrics.get("average_execution_time", 0)
        
        return {
            "performance_summary": {
                "overall_success_rate": overall_success_rate,
                "average_execution_time": average_execution_time,
                "total_workflows": workflow_metrics.get("total_workflows_executed", 0),
                "active_workflows": workflow_metrics.get("active_workflows", 0)
            },
            "pattern_performance": {
                pattern_id: pattern.get_execution_metrics()
                for pattern_id, pattern in self.active_patterns.items()
                if hasattr(pattern, 'get_execution_metrics')
            },
            "agent_performance": {
                agent_id: agent.performance_metrics
                for agent_id, agent in self.registered_agents.items()
            },
            "feedback_insights": feedback_summary
        }

    # Demonstration and Testing
    async def run_demo_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Run a demonstration scenario."""
        print(f"[PLATFORM] Running demo scenario: {scenario_name}")
        
        if scenario_name == "research_pipeline":
            return await self._demo_research_pipeline()
        elif scenario_name == "analysis_supervisor":
            return await self._demo_analysis_supervisor()
        elif scenario_name == "competitive_parallel":
            return await self._demo_competitive_parallel()
        elif scenario_name == "content_reflective":
            return await self._demo_content_reflective()
        elif scenario_name == "all_patterns":
            return await self._demo_all_patterns()
        else:
            raise ValueError(f"Unknown demo scenario: {scenario_name}")

    async def _demo_research_pipeline(self) -> Dict[str, Any]:
        """Demo research pipeline pattern."""
        pipeline = self.create_research_pipeline("demo_research_pipeline")
        
        task = {
            "type": "research",
            "description": "Research the impact of artificial intelligence on modern business operations",
            "requirements": {
                "depth": "comprehensive",
                "sources": ["academic", "industry", "news"],
                "focus_areas": ["automation", "decision_making", "cost_efficiency"]
            }
        }
        
        return await self.execute_pattern(pipeline.pattern_id, task)

    async def _demo_analysis_supervisor(self) -> Dict[str, Any]:
        """Demo analysis supervisor pattern."""
        supervisor = self.create_analysis_supervisor("demo_analysis_supervisor")
        
        task = {
            "type": "analysis",
            "description": "Analyze market trends and competitive landscape for AI-powered business tools",
            "requirements": {
                "market_size": "global",
                "time_horizon": "2024-2026",
                "competitors": ["established", "emerging"],
                "metrics": ["market_share", "growth_rate", "innovation_index"]
            }
        }
        
        return await self.execute_pattern(supervisor.pattern_id, task)

    async def _demo_competitive_parallel(self) -> Dict[str, Any]:
        """Demo competitive analysis parallel pattern."""
        parallel = self.create_competitive_analysis_parallel("demo_competitive_parallel")
        
        task = {
            "type": "competitive_analysis",
            "description": "Comprehensive competitive analysis of the multi-agent AI market",
            "requirements": {
                "analysis_dimensions": ["technology", "market_position", "pricing", "features"],
                "competitors": ["OpenAI", "Anthropic", "Microsoft", "Google"],
                "analysis_depth": "detailed"
            }
        }
        
        return await self.execute_pattern(parallel.pattern_id, task)

    async def _demo_content_reflective(self) -> Dict[str, Any]:
        """Demo content optimization reflective pattern."""
        reflective = self.create_content_optimization_reflective("demo_content_reflective")
        
        task = {
            "type": "content_optimization",
            "description": "Create and optimize a strategic whitepaper on multi-agent AI systems",
            "requirements": {
                "target_audience": "technical_leaders",
                "content_type": "whitepaper",
                "tone": "authoritative_yet_accessible",
                "length": "3000_words",
                "quality_standards": "publication_ready"
            }
        }
        
        return await self.execute_pattern(reflective.pattern_id, task)

    async def _demo_all_patterns(self) -> Dict[str, Any]:
        """Demo all four patterns in sequence."""
        results = {}
        
        # Run each pattern demo
        scenarios = ["research_pipeline", "analysis_supervisor", "competitive_parallel", "content_reflective"]
        
        for scenario in scenarios:
            print(f"\n[PLATFORM] Running {scenario} demo...")
            try:
                result = await self.run_demo_scenario(scenario)
                results[scenario] = {
                    "success": result.get("success", False),
                    "execution_time": result.get("execution_time", 0),
                    "pattern_type": result.get("pattern_type", "unknown")
                }
            except Exception as e:
                results[scenario] = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "demo_type": "all_patterns",
            "patterns_tested": len(scenarios),
            "successful_patterns": len([r for r in results.values() if r.get("success", False)]),
            "results": results,
            "platform_status": self.get_platform_status()
        }

    # Cleanup and Maintenance
    def cleanup_resources(self):
        """Clean up platform resources."""
        self.workflow_engine.clear_history()
        self.result_aggregator.clear_history()
        self.state_manager.clear_all_state()
        self.feedback_loop.clear_feedback_history()
        self.pattern_builder.clear_builder()
        
        print("[PLATFORM] Platform resources cleaned up")

    def reset_platform(self):
        """Reset platform to initial state."""
        self.cleanup_resources()
        self.registered_agents.clear()
        self.active_patterns.clear()
        
        self.platform_metrics = {
            "agents_created": 0,
            "patterns_created": 0,
            "workflows_executed": 0,
            "uptime_start": datetime.now()
        }
        
        print("[PLATFORM] Platform reset to initial state")


# Convenience function for creating platform instance
def create_platform(platform_id: str = None) -> MultiAgentPlatform:
    """Create a new multi-agent platform instance."""
    return MultiAgentPlatform(platform_id or f"platform_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


# Main entry point for CLI usage
def main():
    """Main entry point for the platform."""
    print("Multi-Agent Orchestration Platform")
    print("=" * 50)
    print("Use this platform to orchestrate multi-agent workflows")
    print("with Pipeline, Supervisor, Parallel, and Reflective patterns.")
    print("")
    print("For interactive demo, run: python quick_start.py")


if __name__ == "__main__":
    main()