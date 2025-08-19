"""
LLM Provider Management

Unified interface for multiple LLM providers including OpenAI, Anthropic, AWS Bedrock, etc.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Protocol, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Import providers (with fallback for missing dependencies)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .aws_bedrock import BedrockIntegration, BedrockConfig


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws_bedrock"
    SIMULATION = "simulation"


@dataclass
class LLMConfig:
    """Base configuration for LLM providers."""
    provider: LLMProvider
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text response."""
        pass
    
    @abstractmethod
    async def generate_streaming(self, prompt: str, **kwargs):
        """Generate streaming text response."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get('model', self.config.model_name),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', 1.0)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[OPENAI] Error: {str(e)}")
            return await self._fallback_response(prompt)
    
    async def generate_streaming(self, prompt: str, **kwargs):
        """Generate streaming response using OpenAI."""
        try:
            stream = await self.client.chat.completions.create(
                model=kwargs.get('model', self.config.model_name),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"[OPENAI ERROR] {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
    
    async def _fallback_response(self, prompt: str) -> str:
        """Fallback response for errors."""
        await asyncio.sleep(0.5)
        return f"[OPENAI SIMULATION] Response to: {prompt[:50]}..."


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        self.client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            timeout=config.timeout
        )
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic."""
        try:
            response = await self.client.messages.create(
                model=kwargs.get('model', self.config.model_name),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"[ANTHROPIC] Error: {str(e)}")
            return await self._fallback_response(prompt)
    
    async def generate_streaming(self, prompt: str, **kwargs):
        """Generate streaming response using Anthropic."""
        try:
            async with self.client.messages.stream(
                model=kwargs.get('model', self.config.model_name),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                async for delta in stream:
                    if delta.type == "content_block_delta":
                        yield delta.delta.text
                        
        except Exception as e:
            yield f"[ANTHROPIC ERROR] {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get available Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
    
    async def _fallback_response(self, prompt: str) -> str:
        """Fallback response for errors."""
        await asyncio.sleep(0.5)
        return f"[ANTHROPIC SIMULATION] Response to: {prompt[:50]}..."


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock LLM provider."""
    
    def __init__(self, config: LLMConfig, bedrock_config: BedrockConfig = None):
        super().__init__(config)
        self.bedrock_config = bedrock_config or BedrockConfig(
            model_id=config.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        self.bedrock = BedrockIntegration(self.bedrock_config)
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using AWS Bedrock."""
        return await self.bedrock.generate_text(prompt, **kwargs)
    
    async def generate_streaming(self, prompt: str, **kwargs):
        """Generate streaming response using Bedrock."""
        async for chunk in self.bedrock.generate_streaming(prompt, **kwargs):
            yield chunk
    
    def get_available_models(self) -> List[str]:
        """Get available Bedrock models."""
        models = self.bedrock.list_available_models()
        return [model['modelId'] for model in models]


class SimulationProvider(BaseLLMProvider):
    """Simulation provider for development/testing."""
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate simulated text response."""
        await asyncio.sleep(0.5)  # Simulate API delay
        
        # Simple response generation based on prompt
        prompt_lower = prompt.lower()
        
        if "analyze" in prompt_lower:
            return f"Analysis: Based on the input '{prompt[:30]}...', I have conducted a thorough analysis. The key findings indicate several important patterns and insights that require consideration for decision-making."
        
        elif "summarize" in prompt_lower:
            return f"Summary: The main points from '{prompt[:30]}...' can be condensed into three key areas: 1) Primary insights, 2) Supporting evidence, 3) Actionable recommendations."
        
        elif "research" in prompt_lower:
            return f"Research findings: My investigation into '{prompt[:30]}...' reveals multiple sources and perspectives. The evidence suggests several important considerations for further exploration."
        
        else:
            return f"Response: Thank you for your query about '{prompt[:30]}...'. I have processed this request and can provide insights based on the available information and analysis."
    
    async def generate_streaming(self, prompt: str, **kwargs):
        """Generate simulated streaming response."""
        response = await self.generate_text(prompt, **kwargs)
        words = response.split()
        
        for word in words:
            yield word + " "
            await asyncio.sleep(0.1)
    
    def get_available_models(self) -> List[str]:
        """Get simulated models."""
        return ["simulation-model-1", "simulation-model-2"]


class LLMProviderManager:
    """
    Manages multiple LLM providers and provides unified interface.
    
    Allows switching between providers, load balancing, and fallback handling.
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider: Optional[str] = None
        self.fallback_provider: str = "simulation"
    
    def register_provider(self, name: str, provider: BaseLLMProvider, 
                         set_as_default: bool = False):
        """Register an LLM provider."""
        self.providers[name] = provider
        
        if set_as_default or not self.default_provider:
            self.default_provider = name
    
    def setup_openai(self, api_key: str, model: str = "gpt-3.5-turbo", 
                    set_as_default: bool = False) -> bool:
        """Setup OpenAI provider."""
        try:
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name=model,
                api_key=api_key
            )
            provider = OpenAIProvider(config)
            self.register_provider("openai", provider, set_as_default)
            return True
        except Exception as e:
            print(f"[LLM_MANAGER] Failed to setup OpenAI: {str(e)}")
            return False
    
    def setup_anthropic(self, api_key: str, model: str = "claude-3-haiku-20240307",
                       set_as_default: bool = False) -> bool:
        """Setup Anthropic provider."""
        try:
            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name=model,
                api_key=api_key
            )
            provider = AnthropicProvider(config)
            self.register_provider("anthropic", provider, set_as_default)
            return True
        except Exception as e:
            print(f"[LLM_MANAGER] Failed to setup Anthropic: {str(e)}")
            return False
    
    def setup_bedrock(self, region: str = "us-east-1", 
                     model: str = "anthropic.claude-3-haiku-20240307-v1:0",
                     aws_access_key_id: str = None, aws_secret_access_key: str = None,
                     set_as_default: bool = False) -> bool:
        """Setup AWS Bedrock provider."""
        try:
            config = LLMConfig(
                provider=LLMProvider.AWS_BEDROCK,
                model_name=model
            )
            bedrock_config = BedrockConfig(
                region_name=region,
                model_id=model,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            provider = BedrockProvider(config, bedrock_config)
            self.register_provider("bedrock", provider, set_as_default)
            return True
        except Exception as e:
            print(f"[LLM_MANAGER] Failed to setup Bedrock: {str(e)}")
            return False
    
    def setup_simulation(self, set_as_default: bool = True) -> bool:
        """Setup simulation provider."""
        config = LLMConfig(
            provider=LLMProvider.SIMULATION,
            model_name="simulation"
        )
        provider = SimulationProvider(config)
        self.register_provider("simulation", provider, set_as_default)
        return True
    
    async def generate_text(self, prompt: str, provider: str = None, **kwargs) -> str:
        """Generate text using specified or default provider."""
        target_provider = provider or self.default_provider
        
        if target_provider and target_provider in self.providers:
            try:
                return await self.providers[target_provider].generate_text(prompt, **kwargs)
            except Exception as e:
                print(f"[LLM_MANAGER] Provider {target_provider} failed: {str(e)}")
                # Try fallback
                if self.fallback_provider in self.providers:
                    return await self.providers[self.fallback_provider].generate_text(prompt, **kwargs)
        
        # Ultimate fallback
        if "simulation" not in self.providers:
            self.setup_simulation()
        
        return await self.providers["simulation"].generate_text(prompt, **kwargs)
    
    async def generate_streaming(self, prompt: str, provider: str = None, **kwargs):
        """Generate streaming text using specified or default provider."""
        target_provider = provider or self.default_provider
        
        if target_provider and target_provider in self.providers:
            try:
                async for chunk in self.providers[target_provider].generate_streaming(prompt, **kwargs):
                    yield chunk
                return
            except Exception as e:
                print(f"[LLM_MANAGER] Streaming failed for {target_provider}: {str(e)}")
        
        # Fallback to simulation
        if "simulation" not in self.providers:
            self.setup_simulation()
        
        async for chunk in self.providers["simulation"].generate_streaming(prompt, **kwargs):
            yield chunk
    
    def get_available_providers(self) -> List[str]:
        """Get list of registered providers."""
        return list(self.providers.keys())
    
    def get_provider_models(self, provider: str) -> List[str]:
        """Get available models for a provider."""
        if provider in self.providers:
            return self.providers[provider].get_available_models()
        return []
    
    def set_default_provider(self, provider: str):
        """Set default provider."""
        if provider in self.providers:
            self.default_provider = provider
        else:
            raise ValueError(f"Provider {provider} not registered")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all providers."""
        return {
            "providers": list(self.providers.keys()),
            "default_provider": self.default_provider,
            "fallback_provider": self.fallback_provider,
            "provider_models": {
                name: provider.get_available_models()
                for name, provider in self.providers.items()
            }
        }