"""
Multi-Agent Orchestration Platform

Framework for building and deploying scalable agentic AI systems with 
four core orchestration patterns:
- Pipeline: Sequential workflow with handoffs
- Supervisor: Hierarchical task delegation  
- Parallel: Concurrent execution with fusion
- Reflective: Self-improving feedback loops
"""

__version__ = "0.1.0"
__author__ = "Joshua Morrison"

# Core components
from .multi_agent_platform import MultiAgentPlatform
from .agents.base_agent import BaseAgent
from .patterns.pipeline_pattern import PipelinePattern
from .patterns.supervisor_pattern import SupervisorPattern
from .patterns.parallel_pattern import ParallelPattern
from .patterns.reflective_pattern import ReflectivePattern

__all__ = [
    "MultiAgentPlatform",
    "BaseAgent", 
    "PipelinePattern",
    "SupervisorPattern",
    "ParallelPattern",
    "ReflectivePattern",
]