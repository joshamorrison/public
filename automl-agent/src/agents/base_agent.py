"""
Base Agent Class for AutoML Platform

Provides the foundation for all specialized agents in the AutoML system.
Includes common functionality for logging, configuration, and agent interactions.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

try:
    from langchain.agents import AgentExecutor
    from langchain.agents.agent import BaseAgent as LangChainBaseAgent
    from langchain.schema import AgentAction, AgentFinish
    from langchain.tools import BaseTool
except ImportError:
    # Fallback classes if LangChain not available
    class AgentExecutor: pass
    class LangChainBaseAgent: pass
    class AgentAction: pass
    class AgentFinish: pass
    class BaseTool: pass


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class AgentResult:
    """Result of agent execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None


@dataclass
class TaskContext:
    """Context information for agent tasks."""
    task_id: str
    user_input: str
    dataset_info: Optional[Dict[str, Any]] = None
    previous_results: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """
    Base class for all AutoML agents.
    
    Provides common functionality including:
    - Task execution framework
    - Logging and monitoring
    - Error handling
    - Result formatting
    - Agent communication
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        specialization: str,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[Dict[str, Any]] = None,
        communication_hub: Optional['AgentCommunicationHub'] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name (e.g., "EDA Agent")
            description: Brief description of agent capabilities
            specialization: Agent's area of expertise
            tools: List of tools available to the agent
            config: Agent-specific configuration
        """
        self.name = name
        self.description = description
        self.specialization = specialization
        self.tools = tools or []
        self.config = config or {}
        self.communication_hub = communication_hub
        
        # Agent state
        self.status = AgentStatus.IDLE
        self.current_task: Optional[TaskContext] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        # Communication and collaboration
        self.quality_thresholds: Dict[str, float] = self.config.get("quality_thresholds", {})
        self.refinement_strategies: List[str] = self.config.get("refinement_strategies", [])
        self.collaboration_preferences: Dict[str, float] = self.config.get("collaboration_preferences", {})
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Performance tracking
        self.total_tasks = 0
        self.successful_tasks = 0
        self.total_execution_time = 0.0
        self.current_performance: Dict[str, float] = {}
        
        # Register with communication hub if provided
        if self.communication_hub:
            self.communication_hub.register_agent(
                self, 
                capabilities=self.get_capabilities_list()
            )
        
        self.logger.info(f"Initialized {self.name} - {self.specialization}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup agent-specific logger."""
        logger = logging.getLogger(f"automl.agents.{self.name.lower().replace(' ', '_')}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[{self.name}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute the main task for this agent.
        
        Args:
            context: Task context with input data and constraints
            
        Returns:
            AgentResult with execution results
        """
        pass
    
    @abstractmethod
    def can_handle_task(self, context: TaskContext) -> bool:
        """
        Determine if this agent can handle the given task.
        
        Args:
            context: Task context to evaluate
            
        Returns:
            True if agent can handle the task, False otherwise
        """
        pass
    
    @abstractmethod
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """
        Estimate the complexity of the given task.
        
        Args:
            context: Task context to evaluate
            
        Returns:
            TaskComplexity level
        """
        pass
    
    def run(self, context: TaskContext) -> AgentResult:
        """
        Main execution method with error handling and monitoring.
        
        Args:
            context: Task context
            
        Returns:
            AgentResult with execution results
        """
        self.logger.info(f"Starting task: {context.task_id}")
        start_time = time.time()
        
        try:
            # Update status and current task
            self.status = AgentStatus.THINKING
            self.current_task = context
            self.total_tasks += 1
            
            # Check if agent can handle this task
            if not self.can_handle_task(context):
                raise ValueError(f"Agent {self.name} cannot handle this task type")
            
            # Execute the main task
            self.status = AgentStatus.EXECUTING
            result = self.execute_task(context)
            
            # Update status and metrics
            self.status = AgentStatus.COMPLETED
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            self.total_execution_time += execution_time
            
            if result.success:
                self.successful_tasks += 1
            
            # Log execution
            self._log_execution(context, result)
            
            self.logger.info(f"Completed task: {context.task_id} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            execution_time = time.time() - start_time
            
            error_result = AgentResult(
                success=False,
                message=f"Agent execution failed: {str(e)}",
                execution_time=execution_time,
                metadata={"error_type": type(e).__name__}
            )
            
            self.logger.error(f"Task failed: {context.task_id} - {str(e)}")
            return error_result
        
        finally:
            self.current_task = None
            if self.status != AgentStatus.ERROR:
                self.status = AgentStatus.IDLE
    
    def _log_execution(self, context: TaskContext, result: AgentResult) -> None:
        """Log execution details for monitoring and debugging."""
        execution_log = {
            "timestamp": time.time(),
            "task_id": context.task_id,
            "success": result.success,
            "execution_time": result.execution_time,
            "message": result.message,
            "complexity": self.estimate_complexity(context).value
        }
        
        self.execution_history.append(execution_log)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        success_rate = (
            self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0
        )
        avg_execution_time = (
            self.total_execution_time / self.total_tasks if self.total_tasks > 0 else 0.0
        )
        
        return {
            "agent_name": self.name,
            "specialization": self.specialization,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "current_status": self.status.value,
            "tools_available": len(self.tools)
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities and metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "specialization": self.specialization,
            "tools": [tool.name for tool in self.tools],
            "config": self.config,
            "status": self.status.value
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update agent configuration."""
        self.config.update(new_config)
        self.logger.info(f"Updated configuration: {new_config}")
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a new tool to the agent."""
        self.tools.append(tool)
        self.logger.info(f"Added tool: {tool.name}")
    
    def get_capabilities_list(self) -> List[str]:
        """Get list of agent capabilities for communication hub registration."""
        base_capabilities = [self.specialization.lower().replace(" ", "_")]
        
        # Add tool-based capabilities
        if self.tools:
            base_capabilities.extend([tool.name.lower() for tool in self.tools if hasattr(tool, 'name')])
        
        return base_capabilities
    
    def process_messages(self) -> List[Dict[str, Any]]:
        """Process pending messages from communication hub."""
        if not self.communication_hub:
            return []
        
        messages = self.communication_hub.get_messages(self.name)
        processed = []
        
        for message in messages:
            try:
                response = self._handle_message(message)
                if response:
                    processed.append(response)
            except Exception as e:
                self.logger.error(f"Failed to process message {message.message_id}: {str(e)}")
        
        return processed
    
    def _handle_message(self, message: 'AgentMessage') -> Optional[Dict[str, Any]]:
        """Handle individual message based on type."""
        from .communication import MessageType, AgentMessage, Priority
        
        if message.message_type == MessageType.QUALITY_FEEDBACK:
            return self._handle_quality_feedback(message)
        elif message.message_type == MessageType.REFINEMENT_REQUEST:
            return self._handle_refinement_request(message)
        elif message.message_type == MessageType.KNOWLEDGE_SHARE:
            return self._handle_knowledge_share(message)
        elif message.message_type == MessageType.PERFORMANCE_UPDATE:
            return self._handle_performance_update(message)
        elif message.message_type == MessageType.COLLABORATION_OFFER:
            return self._handle_collaboration_offer(message)
        
        return None
    
    def _handle_quality_feedback(self, message: 'AgentMessage') -> Dict[str, Any]:
        """Handle quality feedback message."""
        content = message.content
        current_perf = content.get("current_performance", {})
        thresholds = content.get("required_thresholds", [])
        
        # Analyze which thresholds are not met
        improvements_needed = []
        for threshold in thresholds:
            metric = threshold["metric"]
            required = threshold["threshold"]
            actual = current_perf.get(metric, 0)
            
            if actual < required:  # Simplified comparison
                improvements_needed.append({
                    "metric": metric,
                    "current": actual,
                    "target": required,
                    "gap": required - actual
                })
        
        return {
            "message_type": "quality_feedback_response",
            "improvements_needed": improvements_needed,
            "estimated_refinement_time": len(improvements_needed) * 2.0,  # minutes
            "confidence": 0.8
        }
    
    def _handle_refinement_request(self, message: 'AgentMessage') -> Dict[str, Any]:
        """Handle refinement request message."""
        content = message.content
        strategy = content.get("strategy", {})
        
        # Plan refinement approach
        refinement_plan = self._create_refinement_plan(strategy)
        
        return {
            "message_type": "refinement_plan",
            "plan": refinement_plan,
            "estimated_improvement": refinement_plan.get("estimated_improvement", 0.1),
            "execution_time": refinement_plan.get("execution_time", 5.0)
        }
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create agent-specific refinement plan. Override in subclasses."""
        return {
            "strategy_name": "generic_refinement",
            "steps": ["analyze_current_performance", "identify_improvements", "implement_changes"],
            "estimated_improvement": 0.05,
            "execution_time": 3.0
        }
    
    def _handle_knowledge_share(self, message: 'AgentMessage') -> Dict[str, Any]:
        """Handle knowledge sharing message."""
        content = message.content
        knowledge_type = content.get("knowledge_type")
        knowledge_data = content.get("data", {})
        
        # Process knowledge based on relevance to this agent
        relevance_score = self._assess_knowledge_relevance(knowledge_type, knowledge_data)
        
        if relevance_score > 0.5:
            self._incorporate_shared_knowledge(knowledge_type, knowledge_data)
        
        return {
            "message_type": "knowledge_received",
            "relevance_score": relevance_score,
            "incorporated": relevance_score > 0.5
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to this agent. Override in subclasses."""
        # Base implementation - can be overridden
        relevant_types = {
            "data_quality_issues": 0.3,
            "feature_importance": 0.2,
            "algorithm_performance": 0.4
        }
        return relevant_types.get(knowledge_type, 0.1)
    
    def _incorporate_shared_knowledge(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> None:
        """Incorporate shared knowledge into agent's knowledge base."""
        # Store in agent's configuration for future use
        if "shared_knowledge" not in self.config:
            self.config["shared_knowledge"] = {}
        
        if knowledge_type not in self.config["shared_knowledge"]:
            self.config["shared_knowledge"][knowledge_type] = []
        
        self.config["shared_knowledge"][knowledge_type].append({
            "data": knowledge_data,
            "timestamp": time.time(),
            "source": knowledge_data.get("source", "unknown")
        })
        
        self.logger.info(f"Incorporated shared knowledge: {knowledge_type}")
    
    def send_quality_feedback(self, target_agent: str, performance_metrics: Dict[str, float], thresholds: List[Dict[str, Any]]) -> None:
        """Send quality feedback to another agent."""
        if not self.communication_hub:
            return
        
        from .communication import AgentMessage, MessageType, Priority
        
        message = AgentMessage(
            from_agent=self.name,
            to_agent=target_agent,
            message_type=MessageType.QUALITY_FEEDBACK,
            priority=Priority.HIGH,
            content={
                "current_performance": performance_metrics,
                "required_thresholds": thresholds,
                "feedback_context": "performance_improvement_request"
            },
            requires_response=True
        )
        
        self.communication_hub.send_message(message)
    
    def request_collaboration(self, target_agent: str, collaboration_type: str, context: Dict[str, Any]) -> None:
        """Request collaboration with another agent."""
        if not self.communication_hub:
            return
        
        from .communication import AgentMessage, MessageType, Priority
        
        message = AgentMessage(
            from_agent=self.name,
            to_agent=target_agent,
            message_type=MessageType.COLLABORATION_OFFER,
            priority=Priority.NORMAL,
            content={
                "collaboration_type": collaboration_type,
                "context": context,
                "estimated_benefit": context.get("estimated_benefit", 0.1)
            },
            requires_response=True
        )
        
        self.communication_hub.send_message(message)
    
    def share_knowledge(self, knowledge_type: str, knowledge_data: Dict[str, Any], target_agents: Optional[List[str]] = None) -> None:
        """Share knowledge with other agents."""
        if not self.communication_hub:
            return
        
        knowledge_data["source"] = self.name
        knowledge_data["timestamp"] = time.time()
        
        self.communication_hub.share_knowledge(
            from_agent=self.name,
            knowledge_type=knowledge_type,
            knowledge_data=knowledge_data,
            target_agents=target_agents
        )
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics and notify communication hub."""
        self.current_performance.update(metrics)
        
        if self.communication_hub:
            self.communication_hub.update_performance(self.name, self.current_performance.copy())
    
    def check_quality_thresholds(self, current_metrics: Dict[str, float]) -> List[str]:
        """Check if current performance meets quality thresholds."""
        violations = []
        
        for metric_name, threshold_value in self.quality_thresholds.items():
            current_value = current_metrics.get(metric_name, 0.0)
            if current_value < threshold_value:
                violations.append(f"{metric_name}: {current_value:.3f} < {threshold_value:.3f}")
        
        return violations
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', specialization='{self.specialization}')"
    
    def __str__(self) -> str:
        return f"{self.name} ({self.specialization}) - Status: {self.status.value}"