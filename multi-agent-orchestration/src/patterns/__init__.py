"""
Multi-Agent Orchestration Patterns

Four core architectural patterns for multi-agent systems:
- Pipeline: Sequential workflow with handoffs and quality gates
- Supervisor: Hierarchical coordination with task delegation
- Parallel: Concurrent execution with result aggregation
- Reflective: Self-improving feedback loops with meta-cognition
"""

from .pipeline_pattern import PipelinePattern
from .supervisor_pattern import SupervisorPattern
from .parallel_pattern import ParallelPattern
from .reflective_pattern import ReflectivePattern
from .pattern_builder import PatternBuilder

__all__ = [
    "PipelinePattern",
    "SupervisorPattern", 
    "ParallelPattern",
    "ReflectivePattern",
    "PatternBuilder"
]