"""
CrewAI Integration for AutoML Platform

This package contains CrewAI crew configurations and orchestration logic
for coordinating multiple agents in complex AutoML workflows.
"""

from .automl_crew import AutoMLCrew
from .crew_configs import CrewConfigurations

__all__ = [
    "AutoMLCrew",
    "CrewConfigurations",
]