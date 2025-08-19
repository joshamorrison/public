"""
Multi-Agent Orchestration Examples

Working examples demonstrating the four core orchestration patterns
and their real-world applications across basic, advanced, and integration scenarios.
"""

# Basic Examples - Fundamental pattern demonstrations
from .basic_examples.simple_pipeline import run_content_creation_pipeline
from .basic_examples.simple_parallel import run_market_analysis_parallel
from .basic_examples.simple_supervisor import run_research_coordination
from .basic_examples.simple_reflective import run_iterative_improvement

# Advanced Examples - Complex workflow demonstrations
from .advanced_examples.complex_research_workflow import run_comprehensive_research
from .advanced_examples.enterprise_analysis import run_enterprise_analysis
from .advanced_examples.adaptive_workflow import run_adaptive_workflow

# Integration Examples - Real-world integration scenarios
from .integration_examples.api_integration_example import run_api_integration_workflow

__all__ = [
    # Basic Examples
    "run_content_creation_pipeline",
    "run_market_analysis_parallel", 
    "run_research_coordination",
    "run_iterative_improvement",
    
    # Advanced Examples
    "run_comprehensive_research",
    "run_enterprise_analysis",
    "run_adaptive_workflow",
    
    # Integration Examples
    "run_api_integration_workflow"
]