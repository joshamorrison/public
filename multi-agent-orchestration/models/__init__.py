"""
Multi-Agent Orchestration Models

This package contains trained models, configurations, and model artifacts
for the multi-agent orchestration platform.

Structure:
- llm_configs/: LLM provider configurations and presets
- agent_models/: Trained agent performance and routing models  
- routing_models/: Task routing and agent assignment models
"""

__version__ = "1.0.0"

# Import key model loading utilities
from .llm_configs.config_loader import load_llm_config, get_available_configs
from .routing_models.model_loader import load_routing_model, get_available_routing_models

__all__ = [
    "load_llm_config",
    "get_available_configs", 
    "load_routing_model",
    "get_available_routing_models"
]