"""
Provider Configuration Constants

Pre-defined configuration sets for different LLM providers.
Used for quick access and programmatic configuration selection.
"""

from typing import Dict, Any

# OpenAI Configuration Presets
OPENAI_CONFIGS = {
    "FAST": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout": 15
    },
    "BALANCED": {
        "model_name": "gpt-4",
        "temperature": 0.5,
        "max_tokens": 2048,
        "timeout": 30
    },
    "QUALITY": {
        "model_name": "gpt-4-turbo",
        "temperature": 0.3,
        "max_tokens": 4096,
        "timeout": 60
    }
}

# Anthropic Configuration Presets
ANTHROPIC_CONFIGS = {
    "FAST": {
        "model_name": "claude-3-haiku-20240307",
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout": 15
    },
    "BALANCED": {
        "model_name": "claude-3-sonnet-20240229",
        "temperature": 0.5,
        "max_tokens": 2048,
        "timeout": 30
    },
    "QUALITY": {
        "model_name": "claude-3-opus-20240229",
        "temperature": 0.3,
        "max_tokens": 4096,
        "timeout": 60
    }
}

# AWS Bedrock Configuration Presets
BEDROCK_CONFIGS = {
    "FAST": {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout": 15,
        "region": "us-east-1"
    },
    "BALANCED": {
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "temperature": 0.5,
        "max_tokens": 2048,
        "timeout": 30,
        "region": "us-east-1"
    },
    "QUALITY": {
        "model_id": "anthropic.claude-3-opus-20240229-v1:0",
        "temperature": 0.3,
        "max_tokens": 4096,
        "timeout": 60,
        "region": "us-east-1"
    },
    "TITAN": {
        "model_id": "amazon.titan-text-express-v1",
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 30,
        "region": "us-east-1"
    }
}

def get_provider_config(provider: str, preset: str) -> Dict[str, Any]:
    """
    Get configuration for a specific provider and preset.
    
    Args:
        provider: Provider name (openai, anthropic, bedrock)
        preset: Preset name (FAST, BALANCED, QUALITY)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If provider or preset not found
    """
    provider_maps = {
        "openai": OPENAI_CONFIGS,
        "anthropic": ANTHROPIC_CONFIGS,
        "bedrock": BEDROCK_CONFIGS
    }
    
    if provider not in provider_maps:
        available = ", ".join(provider_maps.keys())
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
    
    configs = provider_maps[provider]
    
    if preset not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown preset '{preset}' for {provider}. Available: {available}")
    
    config = configs[preset].copy()
    config["provider"] = provider
    
    return config

def get_cost_optimized_config(provider: str, budget_tier: str = "medium") -> Dict[str, Any]:
    """
    Get cost-optimized configuration based on budget tier.
    
    Args:
        provider: Provider name
        budget_tier: Budget tier (low, medium, high)
        
    Returns:
        Cost-optimized configuration
    """
    budget_to_preset = {
        "low": "FAST",
        "medium": "BALANCED", 
        "high": "QUALITY"
    }
    
    preset = budget_to_preset.get(budget_tier, "BALANCED")
    return get_provider_config(provider, preset)

def get_latency_optimized_config(provider: str) -> Dict[str, Any]:
    """
    Get latency-optimized configuration for fastest response.
    
    Args:
        provider: Provider name
        
    Returns:
        Latency-optimized configuration
    """
    config = get_provider_config(provider, "FAST")
    
    # Further optimize for latency
    config.update({
        "max_tokens": 512,
        "timeout": 10,
        "temperature": 0.8  # Higher temperature for faster generation
    })
    
    return config

def get_quality_optimized_config(provider: str) -> Dict[str, Any]:
    """
    Get quality-optimized configuration for best results.
    
    Args:
        provider: Provider name
        
    Returns:
        Quality-optimized configuration
    """
    config = get_provider_config(provider, "QUALITY")
    
    # Further optimize for quality
    config.update({
        "temperature": 0.2,  # Lower temperature for more consistent results
        "top_p": 0.9,
        "max_retries": 3
    })
    
    return config