"""
LLM Configuration Loader

Utilities for loading, managing, and validating LLM configurations
for different providers and agent types.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Get the directory containing this file
CONFIG_DIR = Path(__file__).parent

# Available configuration files
CONFIG_FILES = {
    "openai": "openai_configs.json",
    "anthropic": "anthropic_configs.json", 
    "bedrock": "bedrock_configs.json"
}

def load_llm_config(provider: str, config_name: str = "default") -> Dict[str, Any]:
    """
    Load LLM configuration for specified provider and configuration name.
    
    Args:
        provider: LLM provider (openai, anthropic, bedrock)
        config_name: Configuration name (default, researcher, analyst, etc.)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If provider or config_name not found
        FileNotFoundError: If configuration file doesn't exist
    """
    if provider not in CONFIG_FILES:
        available = ", ".join(CONFIG_FILES.keys())
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
    
    config_file = CONFIG_DIR / CONFIG_FILES[provider]
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_file}: {e}")
    
    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Configuration '{config_name}' not found for {provider}. Available: {available}")
    
    config = configs[config_name].copy()
    
    # Add metadata
    config["_metadata"] = {
        "provider": provider,
        "config_name": config_name,
        "loaded_at": str(Path(__file__).stat().st_mtime),
        "source_file": str(config_file)
    }
    
    logger.info(f"Loaded {provider} config '{config_name}' from {config_file}")
    return config

def get_available_configs(provider: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get list of available configurations for all or specific provider.
    
    Args:
        provider: Optional provider to filter by
        
    Returns:
        Dictionary mapping provider to list of available config names
    """
    result = {}
    
    providers_to_check = [provider] if provider else CONFIG_FILES.keys()
    
    for prov in providers_to_check:
        if prov not in CONFIG_FILES:
            continue
            
        config_file = CONFIG_DIR / CONFIG_FILES[prov]
        
        if not config_file.exists():
            result[prov] = []
            continue
        
        try:
            with open(config_file, 'r') as f:
                configs = json.load(f)
            result[prov] = list(configs.keys())
        except (json.JSONDecodeError, FileNotFoundError):
            result[prov] = []
    
    return result

def save_llm_config(provider: str, config_name: str, config: Dict[str, Any], 
                   overwrite: bool = False) -> bool:
    """
    Save a new LLM configuration.
    
    Args:
        provider: LLM provider
        config_name: Name for the configuration
        config: Configuration dictionary
        overwrite: Whether to overwrite existing configuration
        
    Returns:
        True if saved successfully
        
    Raises:
        ValueError: If configuration already exists and overwrite=False
    """
    if provider not in CONFIG_FILES:
        available = ", ".join(CONFIG_FILES.keys())
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
    
    config_file = CONFIG_DIR / CONFIG_FILES[provider]
    
    # Load existing configurations
    existing_configs = {}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                existing_configs = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in {config_file}, creating new file")
    
    # Check if configuration already exists
    if config_name in existing_configs and not overwrite:
        raise ValueError(f"Configuration '{config_name}' already exists for {provider}. Use overwrite=True to replace.")
    
    # Add the new configuration
    existing_configs[config_name] = config
    
    # Save back to file
    try:
        with open(config_file, 'w') as f:
            json.dump(existing_configs, f, indent=2)
        
        logger.info(f"Saved {provider} config '{config_name}' to {config_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate LLM configuration for required fields and reasonable values.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields
    required_fields = ["provider", "model_name", "temperature", "max_tokens"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate temperature range
    if "temperature" in config:
        temp = config["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            errors.append("Temperature must be a number between 0 and 2")
    
    # Validate max_tokens
    if "max_tokens" in config:
        max_tokens = config["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            errors.append("max_tokens must be a positive integer")
    
    # Validate timeout
    if "timeout" in config:
        timeout = config["timeout"]
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append("timeout must be a positive number")
    
    # Validate provider-specific fields
    provider = config.get("provider", "")
    if provider == "aws_bedrock" and "region" not in config:
        errors.append("AWS Bedrock configurations require 'region' field")
    
    return errors

def get_config_for_agent_type(agent_type: str, provider: str = "anthropic") -> Dict[str, Any]:
    """
    Get optimized configuration for a specific agent type.
    
    Args:
        agent_type: Type of agent (researcher, analyst, synthesizer, critic, supervisor)
        provider: LLM provider to use
        
    Returns:
        Optimized configuration for the agent type
    """
    # Map agent types to configuration names
    agent_config_map = {
        "researcher": "researcher",
        "analyst": "analyst", 
        "synthesizer": "synthesizer",
        "critic": "critic",
        "supervisor": "supervisor",
        "base": "default"
    }
    
    config_name = agent_config_map.get(agent_type.lower(), "default")
    
    try:
        return load_llm_config(provider, config_name)
    except (ValueError, FileNotFoundError):
        logger.warning(f"Could not load {config_name} config for {provider}, falling back to default")
        return load_llm_config(provider, "default")

def estimate_cost(config: Dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost for a request based on configuration and token counts.
    
    Args:
        config: LLM configuration containing cost information
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    cost_info = config.get("cost_per_1k_tokens", {})
    
    input_cost_per_1k = cost_info.get("input", 0.0)
    output_cost_per_1k = cost_info.get("output", 0.0)
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    
    return input_cost + output_cost