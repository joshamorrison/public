"""
Supervisor Agent

Hierarchical coordination agent that manages task delegation, 
orchestrates specialist agents, and synthesizes results.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentResult


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that coordinates and delegates tasks to specialist agents.
    
    The supervisor agent:
    - Analyzes complex tasks and breaks them into subtasks
    - Delegates subtasks to appropriate specialist agents
    - Monitors progress and manages dependencies
    - Synthesizes results from multiple agents
    - Makes high-level decisions about workflow routing
    """

    def __init__(self, agent_id: str = "supervisor-001"):
        super().__init__(
            agent_id=agent_id,
            name="Supervisor Agent",
            description="Hierarchical coordinator that delegates tasks and synthesizes results"
        )
        self.specialist_agents: Dict[str, BaseAgent] = {}
        self.delegation_history: List[Dict[str, Any]] = []

    def register_specialist(self, agent: BaseAgent):
        """
        Register a specialist agent for task delegation.
        
        Args:
            agent: Specialist agent to register
        """
        self.specialist_agents[agent.agent_id] = agent
        print(f"[SUPERVISOR] Registered specialist: {agent.name} ({agent.agent_id})")

    async def process_task(self, task: Dict[str, Any]) -> AgentResult:
        """
        Process a complex task by delegating to specialist agents.
        
        Args:
            task: Task specification with requirements and context
            
        Returns:
            AgentResult: Synthesized result from specialist agents
        """
        start_time = datetime.now()
        
        try:
            # Analyze the task and decompose into subtasks
            subtasks = await self._decompose_task(task)
            
            # Delegate subtasks to appropriate specialists
            delegation_plan = await self._create_delegation_plan(subtasks)
            
            # Execute delegated tasks
            specialist_results = await self._execute_delegated_tasks(delegation_plan)
            
            # Synthesize results
            final_result = await self._synthesize_results(specialist_results, task)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.get("task_id", "unknown"),
                content=final_result,
                confidence=0.9,  # High confidence in coordinated results
                metadata={
                    "subtasks_count": len(subtasks),
                    "specialists_used": list(delegation_plan.keys()),
                    "processing_time": processing_time,
                    "delegation_strategy": "hierarchical"
                },
                timestamp=datetime.now()
            )
            
            self.update_performance_metrics(result, processing_time)
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.get("task_id", "unknown"),
                content="Failed to coordinate task delegation",
                confidence=0.0,
                metadata={"error": str(e), "processing_time": processing_time},
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            self.update_performance_metrics(result, processing_time)
            return result

    async def _decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose a complex task into manageable subtasks.
        
        Args:
            task: Original task specification
            
        Returns:
            List of subtask specifications
        """
        task_type = task.get("type", "general")
        task_description = task.get("description", "")
        
        # Simple task decomposition logic
        subtasks = []
        
        if task_type == "research":
            subtasks = [
                {"type": "information_gathering", "description": f"Gather information about: {task_description}"},
                {"type": "analysis", "description": f"Analyze information for: {task_description}"},
                {"type": "synthesis", "description": f"Synthesize findings for: {task_description}"},
            ]
        elif task_type == "analysis":
            subtasks = [
                {"type": "data_analysis", "description": f"Analyze data for: {task_description}"},
                {"type": "quality_review", "description": f"Review analysis quality for: {task_description}"},
            ]
        else:
            # Generic decomposition
            subtasks = [
                {"type": "information_gathering", "description": task_description},
                {"type": "analysis", "description": task_description},
            ]
        
        return subtasks

    async def _create_delegation_plan(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create a plan for delegating subtasks to specialist agents.
        
        Args:
            subtasks: List of subtasks to delegate
            
        Returns:
            Dictionary mapping agent IDs to their assigned subtasks
        """
        delegation_plan = {}
        
        for subtask in subtasks:
            task_type = subtask.get("type", "general")
            
            # Find the best specialist for this task type
            best_specialist = None
            for specialist in self.specialist_agents.values():
                if specialist.can_handle_task(task_type):
                    best_specialist = specialist
                    break
            
            if best_specialist:
                if best_specialist.agent_id not in delegation_plan:
                    delegation_plan[best_specialist.agent_id] = []
                delegation_plan[best_specialist.agent_id].append(subtask)
            else:
                # Fallback to any available specialist
                if self.specialist_agents:
                    fallback_agent = list(self.specialist_agents.values())[0]
                    if fallback_agent.agent_id not in delegation_plan:
                        delegation_plan[fallback_agent.agent_id] = []
                    delegation_plan[fallback_agent.agent_id].append(subtask)
        
        return delegation_plan

    async def _execute_delegated_tasks(self, delegation_plan: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[AgentResult]]:
        """
        Execute delegated tasks using specialist agents.
        
        Args:
            delegation_plan: Plan mapping agents to their tasks
            
        Returns:
            Dictionary mapping agent IDs to their results
        """
        specialist_results = {}
        
        # Execute tasks for each specialist
        for agent_id, tasks in delegation_plan.items():
            specialist = self.specialist_agents.get(agent_id)
            if not specialist:
                continue
                
            agent_results = []
            for task in tasks:
                try:
                    result = await specialist.process_task(task)
                    agent_results.append(result)
                except Exception as e:
                    # Create error result
                    error_result = AgentResult(
                        agent_id=agent_id,
                        task_id=task.get("task_id", "unknown"),
                        content=f"Failed to process task: {str(e)}",
                        confidence=0.0,
                        metadata={"error": str(e)},
                        timestamp=datetime.now(),
                        success=False,
                        error_message=str(e)
                    )
                    agent_results.append(error_result)
            
            specialist_results[agent_id] = agent_results
        
        return specialist_results

    async def _synthesize_results(self, specialist_results: Dict[str, List[AgentResult]], 
                                original_task: Dict[str, Any]) -> str:
        """
        Synthesize results from multiple specialist agents.
        
        Args:
            specialist_results: Results from specialist agents
            original_task: The original task specification
            
        Returns:
            Synthesized result content
        """
        synthesis_content = []
        synthesis_content.append(f"SUPERVISOR COORDINATION REPORT")
        synthesis_content.append(f"Task: {original_task.get('description', 'Unknown task')}")
        synthesis_content.append(f"Coordination completed at: {datetime.now()}")
        synthesis_content.append("")
        
        total_specialists = len(specialist_results)
        successful_specialists = 0
        
        for agent_id, results in specialist_results.items():
            agent = self.specialist_agents.get(agent_id)
            agent_name = agent.name if agent else agent_id
            
            synthesis_content.append(f"SPECIALIST: {agent_name}")
            synthesis_content.append(f"Tasks completed: {len(results)}")
            
            successful_results = [r for r in results if r.success]
            if successful_results:
                successful_specialists += 1
                avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
                synthesis_content.append(f"Success rate: {len(successful_results)}/{len(results)}")
                synthesis_content.append(f"Average confidence: {avg_confidence:.2f}")
                
                # Include key findings
                for i, result in enumerate(successful_results):
                    synthesis_content.append(f"  Result {i+1}: {result.content[:100]}...")
            else:
                synthesis_content.append("No successful results")
            
            synthesis_content.append("")
        
        # Overall assessment
        success_rate = successful_specialists / total_specialists if total_specialists > 0 else 0
        synthesis_content.append(f"OVERALL COORDINATION ASSESSMENT:")
        synthesis_content.append(f"Specialists engaged: {total_specialists}")
        synthesis_content.append(f"Success rate: {success_rate:.1%}")
        synthesis_content.append(f"Coordination quality: {'Excellent' if success_rate > 0.8 else 'Good' if success_rate > 0.6 else 'Needs improvement'}")
        
        return "\n".join(synthesis_content)

    def get_capabilities(self) -> List[str]:
        """Return supervisor agent capabilities."""
        return [
            "task_coordination",
            "task_delegation", 
            "result_synthesis",
            "workflow_management",
            "specialist_orchestration"
        ]

    def get_delegation_history(self) -> List[Dict[str, Any]]:
        """Return history of task delegations."""
        return self.delegation_history

    def get_registered_specialists(self) -> Dict[str, str]:
        """Return information about registered specialists."""
        return {
            agent_id: agent.name 
            for agent_id, agent in self.specialist_agents.items()
        }