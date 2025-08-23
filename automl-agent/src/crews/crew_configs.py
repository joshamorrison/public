"""
CrewAI Configuration Management

Centralized configuration for CrewAI crews, agents, and tasks.
Provides standardized settings for different types of ML workflows.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class WorkflowType(Enum):
    """Different types of ML workflows supported."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"


class ProcessType(Enum):
    """CrewAI process execution types."""
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    role: str
    goal: str
    backstory: str
    verbose: bool = True
    allow_delegation: bool = False
    max_iter: int = 10
    memory: bool = True
    tools: Optional[List[str]] = None


@dataclass
class TaskConfig:
    """Configuration for individual tasks."""
    description: str
    expected_output: str
    agent_role: str
    tools: Optional[List[str]] = None
    async_execution: bool = False
    context: Optional[List[str]] = None


@dataclass
class CrewConfig:
    """Configuration for entire crew."""
    name: str
    description: str
    agents: List[AgentConfig]
    tasks: List[TaskConfig]
    process: ProcessType = ProcessType.SEQUENTIAL
    verbose: bool = True
    memory: bool = True
    cache: bool = True
    max_rpm: int = 10
    share_crew: bool = False


class CrewConfigurations:
    """
    Centralized CrewAI configuration management.
    
    Provides pre-configured crews for different ML workflows.
    """
    
    def __init__(self):
        """Initialize with default configurations."""
        self._configs = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default crew configurations."""
        
        # Classification Workflow
        self._configs[WorkflowType.CLASSIFICATION] = CrewConfig(
            name="Classification Crew",
            description="Specialized crew for classification tasks",
            agents=[
                AgentConfig(
                    role="Data Analyst",
                    goal="Analyze and understand the dataset structure, quality, and patterns",
                    backstory="Expert data analyst with deep understanding of data quality assessment and exploratory analysis",
                    tools=["data_profiler", "visualization_generator", "statistical_analyzer"]
                ),
                AgentConfig(
                    role="Feature Engineer",
                    goal="Create and select optimal features for classification",
                    backstory="Specialist in feature engineering with expertise in transforming raw data into predictive features",
                    tools=["feature_creator", "feature_selector", "encoder", "scaler"]
                ),
                AgentConfig(
                    role="Classification Specialist",
                    goal="Build and optimize classification models",
                    backstory="ML engineer specialized in classification algorithms and model optimization",
                    tools=["model_trainer", "hyperparameter_tuner", "cross_validator", "metric_calculator"]
                ),
                AgentConfig(
                    role="Quality Assurance",
                    goal="Validate model performance and ensure production readiness",
                    backstory="Senior ML engineer focused on model validation and deployment standards",
                    tools=["model_validator", "performance_analyzer", "bias_detector", "explainer"]
                )
            ],
            tasks=[
                TaskConfig(
                    description="Perform comprehensive exploratory data analysis",
                    expected_output="Detailed data analysis report with insights and recommendations",
                    agent_role="Data Analyst"
                ),
                TaskConfig(
                    description="Engineer features optimized for classification",
                    expected_output="Transformed dataset with engineered features and selection rationale",
                    agent_role="Feature Engineer",
                    context=["data_analysis"]
                ),
                TaskConfig(
                    description="Train and optimize classification models",
                    expected_output="Trained models with performance metrics and hyperparameter settings",
                    agent_role="Classification Specialist",
                    context=["data_analysis", "feature_engineering"]
                ),
                TaskConfig(
                    description="Validate model quality and prepare for deployment",
                    expected_output="Model validation report with deployment recommendations",
                    agent_role="Quality Assurance",
                    context=["data_analysis", "feature_engineering", "model_training"]
                )
            ]
        )
        
        # Regression Workflow
        self._configs[WorkflowType.REGRESSION] = CrewConfig(
            name="Regression Crew",
            description="Specialized crew for regression tasks",
            agents=[
                AgentConfig(
                    role="Data Analyst",
                    goal="Analyze data patterns and relationships for regression modeling",
                    backstory="Statistical analyst expert in regression data analysis and correlation studies"
                ),
                AgentConfig(
                    role="Regression Engineer",
                    goal="Build and optimize regression models",
                    backstory="ML engineer specialized in regression algorithms and predictive modeling"
                )
            ],
            tasks=[
                TaskConfig(
                    description="Analyze data for regression modeling",
                    expected_output="Regression-focused data analysis with correlation insights",
                    agent_role="Data Analyst"
                ),
                TaskConfig(
                    description="Train and optimize regression models",
                    expected_output="Optimized regression models with performance metrics",
                    agent_role="Regression Engineer"
                )
            ]
        )
        
        # NLP Workflow
        self._configs[WorkflowType.NLP] = CrewConfig(
            name="NLP Crew",
            description="Specialized crew for natural language processing tasks",
            agents=[
                AgentConfig(
                    role="Text Analyst",
                    goal="Analyze and preprocess text data",
                    backstory="NLP expert specialized in text analysis and preprocessing"
                ),
                AgentConfig(
                    role="NLP Engineer",
                    goal="Build NLP models and extract insights from text",
                    backstory="ML engineer specialized in natural language processing and text modeling"
                )
            ],
            tasks=[
                TaskConfig(
                    description="Analyze and preprocess text data",
                    expected_output="Cleaned and analyzed text data with preprocessing insights",
                    agent_role="Text Analyst"
                ),
                TaskConfig(
                    description="Build and optimize NLP models",
                    expected_output="Trained NLP models with text-specific performance metrics",
                    agent_role="NLP Engineer"
                )
            ]
        )
    
    def get_config(self, workflow_type: WorkflowType) -> CrewConfig:
        """Get configuration for specific workflow type."""
        if workflow_type not in self._configs:
            raise ValueError(f"No configuration available for workflow type: {workflow_type}")
        return self._configs[workflow_type]
    
    def register_config(self, workflow_type: WorkflowType, config: CrewConfig):
        """Register a new or updated configuration."""
        self._configs[workflow_type] = config
    
    def list_available_workflows(self) -> List[WorkflowType]:
        """List all available workflow types."""
        return list(self._configs.keys())
    
    def get_agent_config(self, workflow_type: WorkflowType, agent_role: str) -> Optional[AgentConfig]:
        """Get specific agent configuration from a workflow."""
        config = self.get_config(workflow_type)
        for agent in config.agents:
            if agent.role == agent_role:
                return agent
        return None
    
    def get_task_config(self, workflow_type: WorkflowType, agent_role: str) -> Optional[TaskConfig]:
        """Get specific task configuration for an agent."""
        config = self.get_config(workflow_type)
        for task in config.tasks:
            if task.agent_role == agent_role:
                return task
        return None
    
    def create_custom_config(
        self,
        name: str,
        description: str,
        agent_configs: List[Dict[str, Any]],
        task_configs: List[Dict[str, Any]],
        process_type: ProcessType = ProcessType.SEQUENTIAL
    ) -> CrewConfig:
        """Create a custom crew configuration."""
        
        agents = [AgentConfig(**config) for config in agent_configs]
        tasks = [TaskConfig(**config) for config in task_configs]
        
        return CrewConfig(
            name=name,
            description=description,
            agents=agents,
            tasks=tasks,
            process=process_type
        )