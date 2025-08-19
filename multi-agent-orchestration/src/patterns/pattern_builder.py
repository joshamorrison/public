"""
Pattern Builder

Utility for dynamically constructing and combining orchestration patterns.
Supports pattern composition, validation, and optimization.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from ..agents.base_agent import BaseAgent
from ..agents.supervisor_agent import SupervisorAgent
from ..agents.researcher_agent import ResearcherAgent
from ..agents.analyst_agent import AnalystAgent
from ..agents.critic_agent import CriticAgent
from ..agents.synthesizer_agent import SynthesizerAgent

from .pipeline_pattern import PipelinePattern
from .supervisor_pattern import SupervisorPattern
from .parallel_pattern import ParallelPattern
from .reflective_pattern import ReflectivePattern


class PatternBuilder:
    """
    Builder for constructing and combining multi-agent orchestration patterns.
    
    The pattern builder:
    - Provides fluent interface for pattern construction
    - Validates pattern configurations
    - Supports pattern composition and chaining
    - Offers optimization recommendations
    - Creates common pattern templates
    """

    def __init__(self):
        """Initialize the pattern builder."""
        self.patterns: Dict[str, Any] = {}
        self.agents: Dict[str, BaseAgent] = {}
        self.composition_history: List[Dict[str, Any]] = []

    def create_pipeline(self, pattern_id: str = None) -> PipelinePattern:
        """
        Create a new pipeline pattern.
        
        Args:
            pattern_id: Optional pattern identifier
            
        Returns:
            Configured pipeline pattern
        """
        if not pattern_id:
            pattern_id = f"pipeline_{len(self.patterns) + 1}"
        
        pipeline = PipelinePattern(pattern_id)
        self.patterns[pattern_id] = pipeline
        
        print(f"[PATTERN_BUILDER] Created pipeline pattern: {pattern_id}")
        return pipeline

    def create_supervisor(self, pattern_id: str = None, 
                         supervisor_agent: SupervisorAgent = None) -> SupervisorPattern:
        """
        Create a new supervisor pattern.
        
        Args:
            pattern_id: Optional pattern identifier
            supervisor_agent: Optional supervisor agent (creates default if None)
            
        Returns:
            Configured supervisor pattern
        """
        if not pattern_id:
            pattern_id = f"supervisor_{len(self.patterns) + 1}"
        
        if not supervisor_agent:
            supervisor_agent = SupervisorAgent(f"supervisor_{pattern_id}")
            self.agents[supervisor_agent.agent_id] = supervisor_agent
        
        supervisor_pattern = SupervisorPattern(supervisor_agent, pattern_id)
        self.patterns[pattern_id] = supervisor_pattern
        
        print(f"[PATTERN_BUILDER] Created supervisor pattern: {pattern_id}")
        return supervisor_pattern

    def create_parallel(self, pattern_id: str = None) -> ParallelPattern:
        """
        Create a new parallel pattern.
        
        Args:
            pattern_id: Optional pattern identifier
            
        Returns:
            Configured parallel pattern
        """
        if not pattern_id:
            pattern_id = f"parallel_{len(self.patterns) + 1}"
        
        parallel = ParallelPattern(pattern_id)
        self.patterns[pattern_id] = parallel
        
        print(f"[PATTERN_BUILDER] Created parallel pattern: {pattern_id}")
        return parallel

    def create_reflective(self, pattern_id: str = None, 
                         primary_agent: BaseAgent = None) -> ReflectivePattern:
        """
        Create a new reflective pattern.
        
        Args:
            pattern_id: Optional pattern identifier
            primary_agent: Optional primary agent for reflection
            
        Returns:
            Configured reflective pattern
        """
        if not pattern_id:
            pattern_id = f"reflective_{len(self.patterns) + 1}"
        
        reflective = ReflectivePattern(pattern_id)
        
        if primary_agent:
            reflective.set_primary_agent(primary_agent)
        
        self.patterns[pattern_id] = reflective
        
        print(f"[PATTERN_BUILDER] Created reflective pattern: {pattern_id}")
        return reflective

    def create_agent(self, agent_type: str, agent_id: str = None, **kwargs) -> BaseAgent:
        """
        Create and register a new agent.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Optional agent identifier
            **kwargs: Additional agent configuration
            
        Returns:
            Created agent
        """
        if not agent_id:
            agent_id = f"{agent_type}_{len(self.agents) + 1}"
        
        agent = None
        
        if agent_type == "supervisor":
            agent = SupervisorAgent(agent_id)
        elif agent_type == "researcher":
            agent = ResearcherAgent(agent_id)
        elif agent_type == "analyst":
            agent = AnalystAgent(agent_id)
        elif agent_type == "critic":
            agent = CriticAgent(agent_id)
        elif agent_type == "synthesizer":
            agent = SynthesizerAgent(agent_id)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.agents[agent_id] = agent
        print(f"[PATTERN_BUILDER] Created {agent_type} agent: {agent_id}")
        
        return agent

    def build_research_pipeline(self, pipeline_id: str = "research_pipeline") -> PipelinePattern:
        """
        Build a pre-configured research pipeline pattern.
        
        Args:
            pipeline_id: Identifier for the pipeline
            
        Returns:
            Configured research pipeline
        """
        # Create agents
        researcher = self.create_agent("researcher", f"{pipeline_id}_researcher")
        analyst = self.create_agent("analyst", f"{pipeline_id}_analyst")
        synthesizer = self.create_agent("synthesizer", f"{pipeline_id}_synthesizer")
        critic = self.create_agent("critic", f"{pipeline_id}_critic")
        
        # Create pipeline
        pipeline = self.create_pipeline(pipeline_id)
        
        # Configure stages with quality gates
        pipeline.add_stage(
            researcher, 
            "Information Gathering",
            quality_gate={"min_confidence": 0.7, "min_content_length": 500}
        )
        
        pipeline.add_stage(
            analyst,
            "Analysis & Insights", 
            quality_gate={"min_confidence": 0.75, "required_keywords": ["analysis", "insights"]}
        )
        
        pipeline.add_stage(
            synthesizer,
            "Result Synthesis",
            quality_gate={"min_confidence": 0.8}
        )
        
        pipeline.add_stage(
            critic,
            "Quality Review",
            quality_gate={"min_confidence": 0.85}
        )
        
        print(f"[PATTERN_BUILDER] Built research pipeline with {len(pipeline.pipeline_stages)} stages")
        return pipeline

    def build_analysis_supervisor(self, supervisor_id: str = "analysis_supervisor") -> SupervisorPattern:
        """
        Build a pre-configured analysis supervisor pattern.
        
        Args:
            supervisor_id: Identifier for the supervisor pattern
            
        Returns:
            Configured analysis supervisor
        """
        # Create supervisor
        supervisor_pattern = self.create_supervisor(supervisor_id)
        
        # Create and register specialist agents
        researcher = self.create_agent("researcher", f"{supervisor_id}_researcher")
        analyst = self.create_agent("analyst", f"{supervisor_id}_analyst")
        synthesizer = self.create_agent("synthesizer", f"{supervisor_id}_synthesizer")
        
        supervisor_pattern.register_specialist(researcher)
        supervisor_pattern.register_specialist(analyst)
        supervisor_pattern.register_specialist(synthesizer)
        
        print(f"[PATTERN_BUILDER] Built analysis supervisor with {len(supervisor_pattern.supervisor.specialist_agents)} specialists")
        return supervisor_pattern

    def build_competitive_analysis_parallel(self, parallel_id: str = "competitive_parallel") -> ParallelPattern:
        """
        Build a pre-configured competitive analysis parallel pattern.
        
        Args:
            parallel_id: Identifier for the parallel pattern
            
        Returns:
            Configured competitive analysis parallel pattern
        """
        # Create parallel pattern
        parallel = self.create_parallel(parallel_id)
        
        # Create specialized analysts
        market_analyst = self.create_agent("analyst", f"{parallel_id}_market")
        competitor_analyst = self.create_agent("analyst", f"{parallel_id}_competitor")
        trend_analyst = self.create_agent("analyst", f"{parallel_id}_trends")
        
        # Add parallel agents with different weights and variants
        parallel.add_parallel_agent(market_analyst, weight=1.2, task_variant="market_analysis")
        parallel.add_parallel_agent(competitor_analyst, weight=1.0, task_variant="competitor_analysis")
        parallel.add_parallel_agent(trend_analyst, weight=0.8, task_variant="trend_analysis")
        
        # Configure fusion strategy
        parallel.set_fusion_strategy("weighted_consensus")
        
        print(f"[PATTERN_BUILDER] Built competitive analysis parallel with {len(parallel.parallel_agents)} agents")
        return parallel

    def build_content_optimization_reflective(self, reflective_id: str = "content_reflective") -> ReflectivePattern:
        """
        Build a pre-configured content optimization reflective pattern.
        
        Args:
            reflective_id: Identifier for the reflective pattern
            
        Returns:
            Configured content optimization reflective pattern
        """
        # Create primary agent and critics
        synthesizer = self.create_agent("synthesizer", f"{reflective_id}_primary")
        content_critic = self.create_agent("critic", f"{reflective_id}_content_critic")
        quality_critic = self.create_agent("critic", f"{reflective_id}_quality_critic")
        
        # Create reflective pattern
        reflective = self.create_reflective(reflective_id, synthesizer)
        
        # Add critics
        reflective.add_critic_agent(content_critic, "content_specialist")
        reflective.add_critic_agent(quality_critic, "quality_specialist")
        
        # Configure reflection parameters
        reflective.configure_reflection(
            max_iterations=4,
            convergence_threshold=0.9,
            improvement_threshold=0.05,
            enable_meta_reasoning=True,
            enable_peer_review=True
        )
        
        print(f"[PATTERN_BUILDER] Built content optimization reflective with {len(reflective.critic_agents)} critics")
        return reflective

    def create_hybrid_pattern(self, name: str, pattern_combination: List[str]) -> Dict[str, Any]:
        """
        Create a hybrid pattern combining multiple orchestration patterns.
        
        Args:
            name: Name for the hybrid pattern
            pattern_combination: List of pattern types to combine
            
        Returns:
            Hybrid pattern configuration
        """
        hybrid_config = {
            "name": name,
            "type": "hybrid",
            "patterns": [],
            "composition_strategy": "sequential",  # or "parallel", "conditional"
            "created_at": datetime.now()
        }
        
        # Create individual patterns
        for i, pattern_type in enumerate(pattern_combination):
            pattern_id = f"{name}_{pattern_type}_{i}"
            
            if pattern_type == "pipeline":
                pattern = self.create_pipeline(pattern_id)
            elif pattern_type == "supervisor":
                pattern = self.create_supervisor(pattern_id)
            elif pattern_type == "parallel":
                pattern = self.create_parallel(pattern_id)
            elif pattern_type == "reflective":
                pattern = self.create_reflective(pattern_id)
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
            
            hybrid_config["patterns"].append({
                "pattern_id": pattern_id,
                "pattern_type": pattern_type,
                "pattern_instance": pattern,
                "execution_order": i
            })
        
        self.composition_history.append(hybrid_config)
        
        print(f"[PATTERN_BUILDER] Created hybrid pattern '{name}' with {len(pattern_combination)} components")
        return hybrid_config

    def validate_pattern_configuration(self, pattern_id: str) -> Dict[str, Any]:
        """
        Validate a pattern configuration for completeness and correctness.
        
        Args:
            pattern_id: Pattern to validate
            
        Returns:
            Validation results and recommendations
        """
        if pattern_id not in self.patterns:
            return {"valid": False, "error": "Pattern not found"}
        
        pattern = self.patterns[pattern_id]
        validation = {
            "pattern_id": pattern_id,
            "pattern_type": pattern.name,
            "valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Pattern-specific validation
        if isinstance(pattern, PipelinePattern):
            if len(pattern.pipeline_stages) == 0:
                validation["warnings"].append("Pipeline has no stages")
            elif len(pattern.pipeline_stages) == 1:
                validation["warnings"].append("Pipeline has only one stage - consider adding more")
            
            if not pattern.quality_gates:
                validation["recommendations"].append("Consider adding quality gates for better control")
        
        elif isinstance(pattern, SupervisorPattern):
            if len(pattern.supervisor.specialist_agents) == 0:
                validation["warnings"].append("Supervisor has no registered specialists")
            elif len(pattern.supervisor.specialist_agents) < 2:
                validation["recommendations"].append("Consider adding more specialists for better task distribution")
        
        elif isinstance(pattern, ParallelPattern):
            if len(pattern.parallel_agents) == 0:
                validation["warnings"].append("Parallel pattern has no agents")
            elif len(pattern.parallel_agents) == 1:
                validation["warnings"].append("Parallel pattern has only one agent - not utilizing parallelism")
        
        elif isinstance(pattern, ReflectivePattern):
            if not pattern.primary_agent:
                validation["warnings"].append("Reflective pattern has no primary agent")
            if len(pattern.critic_agents) == 0:
                validation["recommendations"].append("Add critic agents for better reflection quality")
        
        return validation

    def optimize_pattern_performance(self, pattern_id: str) -> Dict[str, Any]:
        """
        Provide optimization recommendations for a pattern.
        
        Args:
            pattern_id: Pattern to optimize
            
        Returns:
            Optimization recommendations
        """
        if pattern_id not in self.patterns:
            return {"error": "Pattern not found"}
        
        pattern = self.patterns[pattern_id]
        recommendations = {
            "pattern_id": pattern_id,
            "current_configuration": pattern.get_pattern_configuration(),
            "optimizations": []
        }
        
        # Get execution metrics if available
        metrics = pattern.get_execution_metrics()
        
        # Pattern-specific optimizations
        if isinstance(pattern, PipelinePattern) and not metrics.get("no_executions"):
            success_rate = metrics.get("success_rate", 0)
            if success_rate < 0.8:
                recommendations["optimizations"].append({
                    "type": "reliability",
                    "description": "Low success rate detected",
                    "suggestion": "Review quality gates and add error handling"
                })
        
        elif isinstance(pattern, ParallelPattern) and not metrics.get("no_executions"):
            partial_failure_rate = metrics.get("partial_failure_rate", 0)
            if partial_failure_rate > 0.3:
                recommendations["optimizations"].append({
                    "type": "resilience", 
                    "description": "High partial failure rate",
                    "suggestion": "Add timeout controls and fallback agents"
                })
        
        # General recommendations
        recommendations["optimizations"].append({
            "type": "monitoring",
            "description": "Enhanced observability",
            "suggestion": "Implement detailed logging and metrics collection"
        })
        
        return recommendations

    def get_builder_status(self) -> Dict[str, Any]:
        """
        Get current status of the pattern builder.
        
        Returns:
            Builder status and statistics
        """
        pattern_types = {}
        for pattern in self.patterns.values():
            pattern_type = type(pattern).__name__
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        agent_types = {}
        for agent in self.agents.values():
            agent_type = type(agent).__name__
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        return {
            "total_patterns": len(self.patterns),
            "pattern_types": pattern_types,
            "total_agents": len(self.agents),
            "agent_types": agent_types,
            "hybrid_patterns_created": len(self.composition_history),
            "patterns": list(self.patterns.keys()),
            "agents": list(self.agents.keys())
        }

    def clear_builder(self):
        """Clear all patterns and agents from the builder."""
        self.patterns.clear()
        self.agents.clear()
        self.composition_history.clear()
        print("[PATTERN_BUILDER] Builder cleared")