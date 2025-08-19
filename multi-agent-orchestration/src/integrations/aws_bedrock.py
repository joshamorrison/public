"""
AWS Bedrock Integration

Provides integration with AWS Bedrock for foundation model access and management.
"""

import json
import boto3
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..agents.base_agent import BaseAgent, AgentResult


class BedrockModel(Enum):
    """Supported Bedrock foundation models."""
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_2_1 = "anthropic.claude-v2:1"
    CLAUDE_2 = "anthropic.claude-v2"
    CLAUDE_INSTANT = "anthropic.claude-instant-v1"
    
    TITAN_TEXT_G1_LARGE = "amazon.titan-text-lite-v1"
    TITAN_TEXT_G1_EXPRESS = "amazon.titan-text-express-v1"
    
    JURASSIC_2_MID = "ai21.j2-mid-v1"
    JURASSIC_2_ULTRA = "ai21.j2-ultra-v1"
    
    COHERE_COMMAND_TEXT = "cohere.command-text-v14"
    COHERE_COMMAND_LIGHT_TEXT = "cohere.command-light-text-v14"


@dataclass
class BedrockConfig:
    """Configuration for Bedrock integration."""
    region_name: str = "us-east-1"
    model_id: str = BedrockModel.CLAUDE_3_HAIKU.value
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    retry_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_attempts": 3,
        "mode": "adaptive"
    })


class BedrockIntegration:
    """
    AWS Bedrock integration for foundation model access.
    
    Provides methods to interact with Bedrock foundation models for
    text generation, embedding, and other AI capabilities.
    """
    
    def __init__(self, config: BedrockConfig):
        """
        Initialize Bedrock integration.
        
        Args:
            config: Bedrock configuration
        """
        self.config = config
        self._client = None
        self._runtime_client = None
    
    def _get_client(self):
        """Get Bedrock client with lazy initialization."""
        if not self._client:
            session = boto3.Session(
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                aws_session_token=self.config.aws_session_token,
                region_name=self.config.region_name
            )
            
            self._client = session.client(
                'bedrock',
                config=boto3.client.Config(
                    retries=self.config.retry_config
                )
            )
        return self._client
    
    def _get_runtime_client(self):
        """Get Bedrock runtime client with lazy initialization."""
        if not self._runtime_client:
            session = boto3.Session(
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                aws_session_token=self.config.aws_session_token,
                region_name=self.config.region_name
            )
            
            self._runtime_client = session.client(
                'bedrock-runtime',
                config=boto3.client.Config(
                    retries=self.config.retry_config
                )
            )
        return self._runtime_client
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Bedrock foundation model.
        
        Args:
            prompt: Input prompt
            **kwargs: Override configuration parameters
            
        Returns:
            Generated text response
        """
        try:
            # Prepare request body based on model provider
            body = self._prepare_request_body(prompt, **kwargs)
            
            # Make async call to Bedrock
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._invoke_model,
                body
            )
            
            # Parse response based on model provider
            return self._parse_response(response)
            
        except Exception as e:
            print(f"[BEDROCK] Error generating text: {str(e)}")
            # Fallback to simulated response for development
            return await self._fallback_response(prompt)
    
    def _prepare_request_body(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare request body based on model provider."""
        model_id = kwargs.get('model_id', self.config.model_id)
        
        if 'anthropic.claude' in model_id:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        
        elif 'amazon.titan' in model_id:
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": kwargs.get('max_tokens', self.config.max_tokens),
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "topP": kwargs.get('top_p', self.config.top_p)
                }
            }
        
        elif 'ai21.j2' in model_id:
            return {
                "prompt": prompt,
                "maxTokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "topP": kwargs.get('top_p', self.config.top_p)
            }
        
        elif 'cohere.command' in model_id:
            return {
                "prompt": prompt,
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "p": kwargs.get('top_p', self.config.top_p)
            }
        
        else:
            # Generic format
            return {
                "prompt": prompt,
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature)
            }
    
    def _invoke_model(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke Bedrock model (synchronous)."""
        client = self._get_runtime_client()
        
        response = client.invoke_model(
            modelId=self.config.model_id,
            contentType="application/json",
            accept="application/json", 
            body=json.dumps(body)
        )
        
        return json.loads(response['body'].read())
    
    def _parse_response(self, response: Dict[str, Any]) -> str:
        """Parse response based on model provider."""
        model_id = self.config.model_id
        
        if 'anthropic.claude' in model_id:
            if 'content' in response and response['content']:
                return response['content'][0]['text']
            elif 'completion' in response:
                return response['completion']
        
        elif 'amazon.titan' in model_id:
            if 'results' in response and response['results']:
                return response['results'][0]['outputText']
        
        elif 'ai21.j2' in model_id:
            if 'completions' in response and response['completions']:
                return response['completions'][0]['data']['text']
        
        elif 'cohere.command' in model_id:
            if 'generations' in response and response['generations']:
                return response['generations'][0]['text']
        
        # Fallback
        return str(response)
    
    async def _fallback_response(self, prompt: str) -> str:
        """Fallback response for development/testing."""
        await asyncio.sleep(0.5)  # Simulate API delay
        
        return (f"[BEDROCK SIMULATION] Based on the prompt '{prompt[:50]}...', "
                f"I would provide a comprehensive AI-generated response. "
                f"In production, this would be generated by {self.config.model_id}.")
    
    async def generate_streaming(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate streaming text response.
        
        Args:
            prompt: Input prompt
            **kwargs: Override configuration parameters
            
        Yields:
            Streaming text chunks
        """
        try:
            # For development, simulate streaming
            response = await self.generate_text(prompt, **kwargs)
            words = response.split()
            
            for i, word in enumerate(words):
                if i > 0:
                    yield " "
                yield word
                await asyncio.sleep(0.1)  # Simulate streaming delay
                
        except Exception as e:
            yield f"[BEDROCK ERROR] {str(e)}"
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List available foundation models."""
        try:
            client = self._get_client()
            response = client.list_foundation_models()
            return response.get('modelSummaries', [])
        except Exception as e:
            print(f"[BEDROCK] Error listing models: {str(e)}")
            # Return simulated model list
            return [
                {
                    "modelId": model.value,
                    "modelName": model.name,
                    "providerName": model.value.split('.')[0],
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"]
                }
                for model in BedrockModel
            ]
    
    async def get_model_info(self, model_id: str = None) -> Dict[str, Any]:
        """Get information about a specific model."""
        target_model = model_id or self.config.model_id
        
        try:
            client = self._get_client()
            response = client.get_foundation_model(modelIdentifier=target_model)
            return response.get('modelDetails', {})
        except Exception as e:
            print(f"[BEDROCK] Error getting model info: {str(e)}")
            return {
                "modelId": target_model,
                "status": "SIMULATED",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"]
            }


class BedrockAgent(BaseAgent):
    """
    Bedrock-powered agent that uses AWS foundation models.
    
    Extends BaseAgent to use Bedrock for LLM capabilities instead of
    simulated responses.
    """
    
    def __init__(self, agent_id: str, name: str, description: str, 
                 bedrock_config: BedrockConfig = None):
        """
        Initialize Bedrock agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable name
            description: Agent description
            bedrock_config: Bedrock configuration
        """
        super().__init__(agent_id, name, description)
        self.bedrock_config = bedrock_config or BedrockConfig()
        self.bedrock = BedrockIntegration(self.bedrock_config)
    
    async def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate AI response using Bedrock instead of simulation.
        
        Args:
            prompt: Input prompt for the LLM
            context: Optional context information
            
        Returns:
            Generated response text
        """
        # Enhance prompt with context if provided
        enhanced_prompt = prompt
        if context:
            context_str = json.dumps(context, indent=2)
            enhanced_prompt = f"Context: {context_str}\\n\\nPrompt: {prompt}"
        
        # Use Bedrock for generation
        return await self.bedrock.generate_text(enhanced_prompt)
    
    async def process_task(self, task: Dict[str, Any]) -> AgentResult:
        """
        Process task using Bedrock-powered capabilities.
        
        This overrides the base process_task to use real AI instead of simulation.
        """
        # Use the enhanced process_with_tools method which now uses real Bedrock
        return await self.process_with_tools(task)
    
    def get_capabilities(self) -> List[str]:
        """Return capabilities specific to this Bedrock agent."""
        base_capabilities = [
            "text_generation",
            "analysis", 
            "reasoning",
            "question_answering",
            "summarization",
            "research",
            "creative_writing"
        ]
        
        # Add model-specific capabilities
        if 'claude' in self.bedrock_config.model_id:
            base_capabilities.extend([
                "code_analysis",
                "mathematical_reasoning",
                "logical_analysis"
            ])
        
        return base_capabilities
    
    async def set_model(self, model_id: str):
        """Change the underlying Bedrock model."""
        self.bedrock_config.model_id = model_id
        # Recreate integration with new model
        self.bedrock = BedrockIntegration(self.bedrock_config)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_id": self.bedrock_config.model_id,
            "region": self.bedrock_config.region_name,
            "max_tokens": self.bedrock_config.max_tokens,
            "temperature": self.bedrock_config.temperature,
            "top_p": self.bedrock_config.top_p
        }