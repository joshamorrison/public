"""
Base Agent Interface

Abstract base class defining the interface for all agents in the 
multi-agent orchestration platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import asyncio
import json

# Import tools
from ..tools.web_search import search_web
from ..tools.document_processor import analyze_document, summarize_document, extract_keywords
from ..tools.calculation_engine import calculate_stats, evaluate_math, compound_interest


@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    id: str
    sender: str
    recipient: str
    content: str
    message_type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class AgentResult:
    """Result structure for agent outputs"""
    agent_id: str
    task_id: str
    content: str
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Defines the core interface that all specialized agents must implement.
    """

    def __init__(self, agent_id: str, name: str, description: str):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of the agent's capabilities
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.message_history: List[AgentMessage] = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "success_rate": 0.0,
        }

    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> AgentResult:
        """
        Process a task and return results.
        
        Args:
            task: Task specification with requirements and context
            
        Returns:
            AgentResult: Processed result with confidence and metadata
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of capabilities this agent can handle.
        
        Returns:
            List of capability strings
        """
        pass

    def can_handle_task(self, task_type: str) -> bool:
        """
        Check if this agent can handle a specific task type.
        
        Args:
            task_type: Type of task to evaluate
            
        Returns:
            True if agent can handle the task, False otherwise
        """
        return task_type in self.get_capabilities()

    async def send_message(self, recipient: str, content: str, 
                          message_type: str = "info") -> AgentMessage:
        """
        Send a message to another agent.
        
        Args:
            recipient: ID of the receiving agent
            content: Message content
            message_type: Type of message (info, request, response, etc.)
            
        Returns:
            AgentMessage: The sent message
        """
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=message_type,
            timestamp=datetime.now()
        )
        
        self.message_history.append(message)
        return message

    def update_performance_metrics(self, result: AgentResult, 
                                 processing_time: float):
        """
        Update agent performance metrics.
        
        Args:
            result: The result of task processing
            processing_time: Time taken to process the task
        """
        self.performance_metrics["tasks_completed"] += 1
        self.performance_metrics["total_processing_time"] += processing_time
        
        # Update average confidence
        total_tasks = self.performance_metrics["tasks_completed"]
        current_avg = self.performance_metrics["average_confidence"]
        new_avg = ((current_avg * (total_tasks - 1)) + result.confidence) / total_tasks
        self.performance_metrics["average_confidence"] = new_avg
        
        # Update success rate
        if result.success:
            success_count = self.performance_metrics["success_rate"] * (total_tasks - 1) + 1
        else:
            success_count = self.performance_metrics["success_rate"] * (total_tasks - 1)
        self.performance_metrics["success_rate"] = success_count / total_tasks
    
    # LLM Integration Methods
    async def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate AI response using LLM (simulated for now).
        In production, integrate with OpenAI, Anthropic, or AWS Bedrock.
        
        Args:
            prompt: Input prompt for the LLM
            context: Optional context information
            
        Returns:
            Generated response text
        """
        # Simulate LLM processing delay
        await asyncio.sleep(0.5)
        
        # For now, return a simulated intelligent response
        # In production, replace with actual LLM API calls
        return await self._simulate_llm_response(prompt, context)
    
    async def _simulate_llm_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Simulate LLM response for development/testing.
        Replace with actual LLM integration in production.
        """
        # Basic response templates based on prompt content
        prompt_lower = prompt.lower()
        
        if "analyze" in prompt_lower or "analysis" in prompt_lower:
            return f"Based on my analysis of the provided information, I've identified several key insights and patterns. The data suggests important trends that warrant further investigation. My assessment indicates that {self._extract_key_terms(prompt)} are particularly significant factors to consider."
        
        elif "research" in prompt_lower or "investigate" in prompt_lower:
            return f"My research findings indicate that {self._extract_key_terms(prompt)} represents an important area of study. I've gathered comprehensive information from multiple sources to provide a thorough understanding of the topic. The evidence supports several key conclusions about the current state and future implications."
        
        elif "summarize" in prompt_lower or "summary" in prompt_lower:
            return f"In summary, the key points regarding {self._extract_key_terms(prompt)} are as follows: 1) Primary findings show significant patterns, 2) Secondary analysis reveals important correlations, 3) The overall assessment suggests actionable recommendations for implementation."
        
        elif "recommend" in prompt_lower or "suggestion" in prompt_lower:
            return f"Based on my analysis, I recommend the following approach for {self._extract_key_terms(prompt)}: First, establish clear objectives and success metrics. Second, implement a phased approach with regular checkpoints. Third, monitor progress and adjust strategies as needed."
        
        else:
            return f"I've processed your request regarding {self._extract_key_terms(prompt)}. After careful consideration of the available information and relevant factors, I can provide insights that address your specific needs. The analysis suggests several important considerations for your decision-making process."
    
    def _extract_key_terms(self, text: str) -> str:
        """Extract key terms from text for response generation."""
        # Simple keyword extraction
        words = text.split()
        important_words = [word for word in words if len(word) > 4 and word.isalpha()]
        return " and ".join(important_words[:3]) if important_words else "the topic"
    
    # Tool Integration Methods
    async def use_web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Use web search tool to gather information."""
        return await search_web(query, max_results)
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Use document processor to analyze text."""
        return await analyze_document(text)
    
    async def summarize_text(self, text: str) -> Dict[str, Any]:
        """Use document processor to summarize text."""
        return await summarize_document(text)
    
    async def extract_text_keywords(self, text: str) -> Dict[str, Any]:
        """Use document processor to extract keywords."""
        return await extract_keywords(text)
    
    async def calculate_statistics(self, numbers: List[float]) -> Dict[str, Any]:
        """Use calculation engine for statistical analysis."""
        return await calculate_stats(numbers)
    
    async def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """Use calculation engine to evaluate mathematical expressions."""
        return await evaluate_math(expression)
    
    # Enhanced Task Processing
    async def process_with_tools(self, task: Dict[str, Any]) -> AgentResult:
        """
        Process task using available tools and LLM capabilities.
        This provides a default implementation that specialized agents can override.
        """
        start_time = datetime.now()
        
        try:
            task_type = task.get("type", "general")
            description = task.get("description", "")
            
            # Determine which tools to use based on task type
            if task_type in ["research", "information_gathering"]:
                # Use web search for research tasks
                search_results = await self.use_web_search(description)
                analysis = await self.analyze_text(str(search_results))
                
                # Generate AI response based on findings
                prompt = f"Based on research about '{description}', provide comprehensive analysis and insights."
                ai_response = await self.generate_response(prompt, {"search_results": search_results})
                
                content = f"Research Analysis: {ai_response}\n\nDetailed Findings: {json.dumps(search_results, indent=2)}"
                confidence = 0.85
                
            elif task_type in ["analysis", "data_analysis"]:
                # Use document processing and analysis tools
                if "data" in task:
                    stats = await self.calculate_statistics(task["data"])
                    prompt = f"Analyze the statistical results for '{description}': {stats}"
                else:
                    prompt = f"Provide detailed analysis of '{description}'"
                
                ai_response = await self.generate_response(prompt, task)
                content = ai_response
                confidence = 0.80
                
            elif task_type in ["summarization", "summary"]:
                # Use summarization tools
                if "text" in task:
                    summary_result = await self.summarize_text(task["text"])
                    content = summary_result.get("summary", "No summary available")
                    confidence = 0.90
                else:
                    prompt = f"Provide a summary of '{description}'"
                    content = await self.generate_response(prompt, task)
                    confidence = 0.75
                    
            else:
                # General task processing with AI
                prompt = f"Process this {task_type} task: {description}"
                content = await self.generate_response(prompt, task)
                confidence = 0.70
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_id=self.agent_id,
                content=content,
                confidence=confidence,
                success=True,
                timestamp=datetime.now(),
                metadata={
                    "task_type": task_type,
                    "processing_time": processing_time,
                    "tools_used": ["llm", "search", "analysis"],
                    "agent_name": self.name
                }
            )
            
            self.update_performance_metrics(result, processing_time)
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_id=self.agent_id,
                content=f"Task processing failed: {str(e)}",
                confidence=0.0,
                success=False,
                timestamp=datetime.now(),
                error_message=str(e),
                metadata={
                    "task_type": task.get("type", "unknown"),
                    "processing_time": processing_time,
                    "error": str(e)
                }
            )
            
            self.update_performance_metrics(result, processing_time)
            return result

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and performance metrics.
        
        Returns:
            Dictionary with agent status and metrics
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.get_capabilities(),
            "created_at": self.created_at.isoformat(),
            "performance_metrics": self.performance_metrics,
            "message_history_length": len(self.message_history),
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            "tasks_completed": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "success_rate": 0.0,
        }

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} ({self.agent_id}): {self.description}"

    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"BaseAgent(agent_id='{self.agent_id}', name='{self.name}', "
                f"capabilities={self.get_capabilities()})")