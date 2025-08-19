"""
Agent implementations for multi-agent orchestration.

Includes specialized agents for different roles in multi-agent workflows:
- BaseAgent: Abstract interface for all agents
- SupervisorAgent: Hierarchical coordination and task delegation  
- ResearcherAgent: Information gathering and analysis
- AnalystAgent: Data analysis and insight generation
- CriticAgent: Quality assessment and feedback
- SynthesizerAgent: Result aggregation and fusion
"""

from .base_agent import BaseAgent
from .supervisor_agent import SupervisorAgent
from .researcher_agent import ResearcherAgent
from .analyst_agent import AnalystAgent
from .critic_agent import CriticAgent
from .synthesizer_agent import SynthesizerAgent

__all__ = [
    "BaseAgent",
    "SupervisorAgent", 
    "ResearcherAgent",
    "AnalystAgent",
    "CriticAgent",
    "SynthesizerAgent",
]