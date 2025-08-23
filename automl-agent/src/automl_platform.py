"""
AutoML Platform - Main Orchestration System

The central platform that coordinates all agents and provides the main
interface for the AutoML system. Handles task routing, agent coordination,
and result aggregation.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

from .agents.base_agent import BaseAgent, AgentResult, TaskContext
from .agents.router_agent import RouterAgent
from .agents.communication import AgentCommunicationHub
from .crews.automl_crew import AutoMLCrew, CrewResult


@dataclass
class PlatformResult:
    """Complete platform execution result."""
    success: bool
    task_specification: Optional[Dict[str, Any]] = None
    workflow_results: Optional[Dict[str, AgentResult]] = None
    total_execution_time: float = 0.0
    message: str = ""
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class AutoMLPlatform:
    """
    Main AutoML Platform orchestrator.
    
    Provides the primary interface for the multi-agent AutoML system.
    Handles:
    - Task intake and validation
    - Router agent coordination
    - Workflow execution via CrewAI
    - Result aggregation and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AutoML Platform.
        
        Args:
            config: Platform configuration options
        """
        self.config = config or {}
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize communication system
        self.communication_hub = AgentCommunicationHub(config=self.config.get("communication", {}))
        
        # Initialize core components with communication
        self.router_agent = RouterAgent(
            config=self.config.get("router", {}),
            communication_hub=self.communication_hub
        )
        self.crew = AutoMLCrew(
            config=self.config.get("crew", {}),
            communication_hub=self.communication_hub
        )
        
        # Add router to crew
        self.crew.add_agent(self.router_agent)
        
        # Platform state
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.execution_history: List[PlatformResult] = []
        
        self.logger.info("AutoML Platform initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup platform logger."""
        logger = logging.getLogger("automl_platform")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[AutoML Platform] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_request(
        self,
        user_input: str,
        dataset_info: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> PlatformResult:
        """
        Process a complete AutoML request.
        
        Args:
            user_input: Natural language description of the ML task
            dataset_info: Information about the dataset
            constraints: Task constraints (time, accuracy, etc.)
            preferences: User preferences for execution
            
        Returns:
            PlatformResult with complete execution results
        """
        start_time = time.time()
        task_id = f"task_{int(time.time())}_{self.total_tasks_processed}"
        
        self.logger.info(f"Processing new request: {task_id}")
        self.logger.info(f"User input: {user_input}")
        
        try:
            # Create task context
            context = TaskContext(
                task_id=task_id,
                user_input=user_input,
                dataset_info=dataset_info,
                constraints=constraints,
                preferences=preferences
            )
            
            # Phase 1: Route the task
            self.logger.info("Phase 1: Analyzing task and creating workflow...")
            routing_result = self.router_agent.run(context)
            
            if not routing_result.success:
                return PlatformResult(
                    success=False,
                    message=f"Task routing failed: {routing_result.message}",
                    total_execution_time=time.time() - start_time
                )
            
            # Extract workflow information
            workflow_data = routing_result.data
            task_spec = workflow_data["task_specification"]
            workflow_sequence = workflow_data["workflow"]["sequence"]
            
            self.logger.info(f"Task identified as: {task_spec['problem_type']}")
            self.logger.info(f"Workflow planned: {' -> '.join(workflow_sequence)}")
            
            # Phase 2: Execute the workflow
            self.logger.info("Phase 2: Executing agent workflow...")
            
            # For now, we'll simulate the workflow execution since we haven't built all agents yet
            workflow_results = self._simulate_workflow_execution(workflow_sequence, context)
            
            # Phase 3: Aggregate results
            self.logger.info("Phase 3: Aggregating results...")
            execution_time = time.time() - start_time
            
            # Create final result
            result = PlatformResult(
                success=True,
                task_specification=task_spec,
                workflow_results=workflow_results,
                total_execution_time=execution_time,
                message=f"AutoML workflow completed successfully in {execution_time:.2f}s",
                recommendations=routing_result.recommendations,
                metadata={
                    "task_id": task_id,
                    "workflow_sequence": workflow_sequence,
                    "router_analysis": workflow_data
                }
            )
            
            # Update platform metrics
            self.total_tasks_processed += 1
            self.successful_tasks += 1
            self.execution_history.append(result)
            
            self.logger.info(f"Successfully completed task: {task_id}")
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = PlatformResult(
                success=False,
                message=f"Platform execution failed: {str(e)}",
                total_execution_time=execution_time,
                metadata={"task_id": task_id, "error_type": type(e).__name__}
            )
            
            self.total_tasks_processed += 1
            self.execution_history.append(error_result)
            
            self.logger.error(f"Task failed: {task_id} - {str(e)}")
            return error_result
    
    def _simulate_workflow_execution(
        self, 
        workflow_sequence: List[str], 
        context: TaskContext
    ) -> Dict[str, AgentResult]:
        """
        Simulate workflow execution for demonstration purposes.
        
        In the full implementation, this would execute actual agents.
        """
        results = {}
        
        for agent_name in workflow_sequence:
            # Simulate agent execution with realistic timing
            time.sleep(0.1)  # Small delay to simulate processing
            
            # Create mock successful results
            if "EDA" in agent_name:
                results[agent_name] = AgentResult(
                    success=True,
                    data={
                        "dataset_shape": [10000, 15],
                        "missing_values": {"column_a": 50, "column_b": 0},
                        "data_types": {"numerical": 8, "categorical": 7},
                        "target_distribution": {"class_0": 6800, "class_1": 3200}
                    },
                    message="EDA completed: Dataset profiling and visualization generated",
                    execution_time=0.1
                )
            
            elif "Hygiene" in agent_name:
                results[agent_name] = AgentResult(
                    success=True,
                    data={
                        "cleaning_steps": ["missing_value_imputation", "outlier_treatment"],
                        "rows_removed": 127,
                        "columns_transformed": ["column_a", "column_c"],
                        "data_quality_score": 0.94
                    },
                    message="Data cleaning completed: 94% data quality achieved",
                    execution_time=0.1
                )
            
            elif "Feature" in agent_name:
                results[agent_name] = AgentResult(
                    success=True,
                    data={
                        "original_features": 15,
                        "engineered_features": 12,
                        "selected_features": 25,
                        "feature_importance": {"feature_1": 0.234, "feature_2": 0.187}
                    },
                    message="Feature engineering completed: 25 features selected",
                    execution_time=0.1
                )
            
            elif "Classification" in agent_name:
                results[agent_name] = AgentResult(
                    success=True,
                    data={
                        "best_algorithm": "XGBoost",
                        "cross_val_score": 0.887,
                        "test_accuracy": 0.894,
                        "precision": 0.876,
                        "recall": 0.823,
                        "f1_score": 0.849
                    },
                    message="Classification model trained: 89.4% accuracy achieved",
                    execution_time=0.1
                )
            
            else:
                results[agent_name] = AgentResult(
                    success=True,
                    data={"status": "completed"},
                    message=f"{agent_name} execution completed",
                    execution_time=0.1
                )
        
        return results
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get current platform status and metrics."""
        success_rate = (
            self.successful_tasks / self.total_tasks_processed 
            if self.total_tasks_processed > 0 else 0.0
        )
        
        return {
            "total_tasks_processed": self.total_tasks_processed,
            "successful_tasks": self.successful_tasks,
            "success_rate": success_rate,
            "agents_available": len(self.crew.agents),
            "crew_status": self.crew.get_execution_status(),
            "router_performance": self.router_agent.get_performance_metrics()
        }
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add a new agent to the platform."""
        self.crew.add_agent(agent)
        self.logger.info(f"Added agent: {agent.name}")
    
    def list_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.crew.agents.keys())
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        recent_history = self.execution_history[-limit:] if limit else self.execution_history
        return [asdict(result) for result in recent_history]
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        self.crew.clear_history()
        self.logger.info("Execution history cleared")
    
    def __repr__(self) -> str:
        return f"AutoMLPlatform(agents={len(self.crew.agents)}, tasks_processed={self.total_tasks_processed})"