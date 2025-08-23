"""
AutoML Crew - CrewAI Integration

Main crew orchestration for the AutoML platform. Coordinates multiple
specialized agents to execute complex ML workflows.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    from crewai import Agent, Crew, Task, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    # Fallback classes if CrewAI is not installed
    class Agent:
        def __init__(self, **kwargs): pass
    class Crew:
        def __init__(self, **kwargs): pass
    class Task:
        def __init__(self, **kwargs): pass
    class Process:
        sequential = "sequential"
        hierarchical = "hierarchical"
    CREWAI_AVAILABLE = False

from ..agents.base_agent import BaseAgent, AgentResult, TaskContext


@dataclass
class CrewResult:
    """Result of crew execution."""
    success: bool
    results: Dict[str, AgentResult]
    total_execution_time: float
    workflow_sequence: List[str]
    metadata: Optional[Dict[str, Any]] = None


class AutoMLCrew:
    """
    Main CrewAI orchestration for AutoML workflows.
    
    Manages the execution of multiple agents in sequence or parallel,
    handling dependencies, error recovery, and result aggregation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub: Optional[Any] = None):
        """
        Initialize the AutoML Crew.
        
        Args:
            config: Crew configuration options
            communication_hub: Communication hub for agent coordination
        """
        self.config = config or {}
        self.communication_hub = communication_hub
        self.agents: Dict[str, BaseAgent] = {}
        self.crew_ai_agents: Dict[str, Agent] = {}
        self.tasks: List[Task] = []
        self.crew: Optional[Crew] = None
        
        self.execution_history: List[CrewResult] = []
        self.is_executing = False
        
        # Setup based on CrewAI availability
        self.crewai_available = CREWAI_AVAILABLE
        if not self.crewai_available:
            print("[WARNING] CrewAI not installed. Running in fallback mode.")
    
    def add_agent(self, agent: BaseAgent) -> None:
        """
        Add an agent to the crew.
        
        Args:
            agent: BaseAgent instance to add
        """
        self.agents[agent.name] = agent
        
        # Create corresponding CrewAI agent if available
        if self.crewai_available:
            crewai_agent = Agent(
                role=agent.specialization,
                goal=f"Execute {agent.specialization.lower()} tasks efficiently",
                backstory=agent.description,
                verbose=True,
                allow_delegation=False,
                tools=[tool for tool in agent.tools if hasattr(tool, 'name')]
            )
            self.crew_ai_agents[agent.name] = crewai_agent
    
    def create_workflow_tasks(self, workflow_sequence: List[str], context: TaskContext) -> List[Task]:
        """
        Create CrewAI tasks for the workflow sequence.
        
        Args:
            workflow_sequence: List of agent names in execution order
            context: Task context
            
        Returns:
            List of CrewAI Task objects
        """
        tasks = []
        
        if not self.crewai_available:
            return tasks
        
        for i, agent_name in enumerate(workflow_sequence):
            if agent_name not in self.crew_ai_agents:
                continue
                
            # Create task description based on agent specialization
            agent = self.agents[agent_name]
            task_description = self._generate_task_description(agent, context, i == 0)
            
            task = Task(
                description=task_description,
                agent=self.crew_ai_agents[agent_name],
                expected_output=f"Structured output from {agent_name} execution"
            )
            
            tasks.append(task)
        
        return tasks
    
    def _generate_task_description(self, agent: BaseAgent, context: TaskContext, is_first: bool) -> str:
        """Generate appropriate task description for each agent."""
        base_description = f"Execute {agent.specialization.lower()} on the provided dataset."
        
        if is_first:
            base_description += f" User request: '{context.user_input}'"
        
        if context.dataset_info:
            base_description += f" Dataset has {context.dataset_info.get('shape', ['unknown', 'unknown'])[0]} rows."
        
        if context.constraints:
            constraints_str = ", ".join([f"{k}: {v}" for k, v in context.constraints.items()])
            base_description += f" Constraints: {constraints_str}"
        
        return base_description
    
    async def execute_workflow_async(
        self, 
        workflow_sequence: List[str], 
        context: TaskContext,
        process_type: str = "sequential"
    ) -> CrewResult:
        """
        Execute workflow asynchronously.
        
        Args:
            workflow_sequence: List of agent names in execution order
            context: Task context
            process_type: "sequential" or "hierarchical"
            
        Returns:
            CrewResult with execution results
        """
        if self.crewai_available:
            return await self._execute_with_crewai(workflow_sequence, context, process_type)
        else:
            return await self._execute_fallback(workflow_sequence, context)
    
    def execute_workflow(
        self, 
        workflow_sequence: List[str], 
        context: TaskContext,
        process_type: str = "sequential"
    ) -> CrewResult:
        """
        Execute workflow synchronously.
        
        Args:
            workflow_sequence: List of agent names in execution order
            context: Task context
            process_type: "sequential" or "hierarchical"
            
        Returns:
            CrewResult with execution results
        """
        return asyncio.run(self.execute_workflow_async(workflow_sequence, context, process_type))
    
    async def _execute_with_crewai(
        self, 
        workflow_sequence: List[str], 
        context: TaskContext,
        process_type: str
    ) -> CrewResult:
        """Execute workflow using CrewAI."""
        start_time = time.time()
        results = {}
        
        try:
            # Create tasks for the workflow
            tasks = self.create_workflow_tasks(workflow_sequence, context)
            
            if not tasks:
                return CrewResult(
                    success=False,
                    results={},
                    total_execution_time=time.time() - start_time,
                    workflow_sequence=workflow_sequence,
                    metadata={"error": "No valid tasks created"}
                )
            
            # Create and configure crew
            crew_agents = [self.crew_ai_agents[name] for name in workflow_sequence if name in self.crew_ai_agents]
            
            crew = Crew(
                agents=crew_agents,
                tasks=tasks,
                process=Process.sequential if process_type == "sequential" else Process.hierarchical,
                verbose=True
            )
            
            # Execute the crew
            self.is_executing = True
            crew_result = crew.kickoff()
            
            # Process results (CrewAI returns different formats)
            for i, agent_name in enumerate(workflow_sequence):
                if agent_name in self.agents:
                    # Create a mock result for now - in real implementation,
                    # we would extract actual results from crew execution
                    results[agent_name] = AgentResult(
                        success=True,
                        data={"crew_output": str(crew_result)},
                        message=f"CrewAI execution completed for {agent_name}",
                        execution_time=0.0  # Would be extracted from actual execution
                    )
            
            execution_time = time.time() - start_time
            
            return CrewResult(
                success=True,
                results=results,
                total_execution_time=execution_time,
                workflow_sequence=workflow_sequence,
                metadata={"crew_result": str(crew_result)}
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return CrewResult(
                success=False,
                results=results,
                total_execution_time=execution_time,
                workflow_sequence=workflow_sequence,
                metadata={"error": str(e)}
            )
        
        finally:
            self.is_executing = False
    
    async def _execute_fallback(
        self, 
        workflow_sequence: List[str], 
        context: TaskContext
    ) -> CrewResult:
        """Execute workflow using fallback sequential execution."""
        start_time = time.time()
        results = {}
        current_context = context
        
        try:
            for agent_name in workflow_sequence:
                if agent_name not in self.agents:
                    results[agent_name] = AgentResult(
                        success=False,
                        message=f"Agent {agent_name} not found"
                    )
                    continue
                
                agent = self.agents[agent_name]
                
                # Update context with previous results
                if results:
                    current_context.previous_results = {
                        name: result.data for name, result in results.items()
                        if result.success and result.data
                    }
                
                # Execute agent
                result = agent.run(current_context)
                results[agent_name] = result
                
                # Stop execution if agent fails and it's critical
                if not result.success and self._is_critical_agent(agent_name):
                    break
            
            execution_time = time.time() - start_time
            overall_success = all(result.success for result in results.values())
            
            return CrewResult(
                success=overall_success,
                results=results,
                total_execution_time=execution_time,
                workflow_sequence=workflow_sequence,
                metadata={"execution_mode": "fallback"}
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return CrewResult(
                success=False,
                results=results,
                total_execution_time=execution_time,
                workflow_sequence=workflow_sequence,
                metadata={"error": str(e), "execution_mode": "fallback"}
            )
    
    def _is_critical_agent(self, agent_name: str) -> bool:
        """Determine if an agent is critical for workflow continuation."""
        critical_agents = [
            "Router Agent",
            "Data Hygiene Agent",
            "Classification Agent", 
            "Regression Agent"
        ]
        return agent_name in critical_agents
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "is_executing": self.is_executing,
            "total_agents": len(self.agents),
            "crewai_available": self.crewai_available,
            "execution_history_count": len(self.execution_history)
        }
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents."""
        summary = {}
        for name, agent in self.agents.items():
            summary[name] = agent.get_performance_metrics()
        return summary
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        for agent in self.agents.values():
            agent.execution_history.clear()
    
    def __len__(self) -> int:
        """Return number of agents in crew."""
        return len(self.agents)
    
    def __contains__(self, agent_name: str) -> bool:
        """Check if agent is in crew."""
        return agent_name in self.agents