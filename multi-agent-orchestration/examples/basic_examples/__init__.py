"""
Basic Multi-Agent Examples

Simple demonstrations of the four core orchestration patterns.
Perfect for understanding the fundamentals and getting started.
"""

from .simple_pipeline import run_content_creation_pipeline
from .simple_parallel import run_market_analysis_parallel
from .simple_supervisor import run_research_coordination
from .simple_reflective import run_iterative_improvement

__all__ = [
    "run_content_creation_pipeline",
    "run_market_analysis_parallel",
    "run_research_coordination", 
    "run_iterative_improvement"
]