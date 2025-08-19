"""
Supervisor Pattern

Hierarchical coordination with centralized decision-making.
A supervisor agent coordinates and delegates to specialist agents.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..agents.base_agent import BaseAgent, AgentResult
from ..agents.supervisor_agent import SupervisorAgent


class SupervisorPattern:
    """
    Supervisor orchestration pattern for hierarchical agent coordination.
    
    The supervisor pattern:
    - Uses a supervisor agent to coordinate workflow
    - Delegates tasks to appropriate specialist agents
    - Manages task decomposition and result synthesis
    - Provides centralized decision-making and oversight
    - Supports dynamic agent selection and routing
    """

    def __init__(self, supervisor_agent: SupervisorAgent, pattern_id: str = "supervisor-001"):
        """
        Initialize the supervisor pattern.
        
        Args:
            supervisor_agent: The supervisor agent to coordinate workflow
            pattern_id: Unique identifier for this pattern instance
        """
        self.pattern_id = pattern_id
        self.name = "Supervisor Pattern"
        self.description = "Hierarchical coordination with task delegation"
        self.supervisor = supervisor_agent
        self.execution_history: List[Dict[str, Any]] = []

    def register_specialist(self, agent: BaseAgent, capabilities: Optional[List[str]] = None):
        """
        Register a specialist agent with the supervisor.
        
        Args:
            agent: Specialist agent to register
            capabilities: Optional list of specific capabilities (uses agent's capabilities if None)
        """
        self.supervisor.register_specialist(agent)
        
        capabilities_list = capabilities or agent.get_capabilities()
        print(f"[SUPERVISOR] Registered specialist {agent.name} with capabilities: {capabilities_list}")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task using supervisor coordination.
        
        Args:
            task: Task to be coordinated and executed
            
        Returns:
            Supervisor coordination results
        """
        execution_id = f"super_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        print(f"[SUPERVISOR] Starting coordination {execution_id}")
        print(f"[SUPERVISOR] Registered specialists: {len(self.supervisor.specialist_agents)}")
        
        execution_result = {
            "execution_id": execution_id,
            "pattern_type": "supervisor",
            "start_time": start_time,
            "supervisor_agent_id": self.supervisor.agent_id,
            "specialists_available": len(self.supervisor.specialist_agents),
            "coordination_result": None,
            "specialist_results": {},
            "success": True,
            "error_message": None
        }
        
        try:
            # Prepare task for supervisor
            supervisor_task = task.copy()
            supervisor_task.update({
                "supervision_execution_id": execution_id,
                "specialists_available": list(self.supervisor.specialist_agents.keys()),
                "coordination_mode": "hierarchical"
            })
            
            print(f"[SUPERVISOR] Delegating task to supervisor: {task.get('description', 'No description')}")
            
            # Let supervisor coordinate the work
            coordination_result = await self.supervisor.process_task(supervisor_task)
            
            execution_result["coordination_result"] = coordination_result
            execution_result["success"] = coordination_result.success
            
            if not coordination_result.success:
                execution_result["error_message"] = coordination_result.error_message
            
            # Extract specialist results from supervisor metadata
            if coordination_result.metadata:
                specialist_info = {}
                for agent_id, agent in self.supervisor.specialist_agents.items():
                    specialist_info[agent_id] = {
                        "agent_name": agent.name,
                        "capabilities": agent.get_capabilities(),
                        "performance_metrics": agent.performance_metrics
                    }
                execution_result["specialist_results"] = specialist_info
            
            execution_result["end_time"] = datetime.now()
            execution_result["total_execution_time"] = (execution_result["end_time"] - start_time).total_seconds()
            
            # Store execution history
            self.execution_history.append(execution_result)
            
            status = "SUCCESS" if execution_result["success"] else "FAILED"
            print(f"[SUPERVISOR] Coordination {execution_id} completed: {status}")
            
            return execution_result
            
        except Exception as e:
            execution_result["success"] = False
            execution_result["error_message"] = str(e)
            execution_result["end_time"] = datetime.now()
            
            self.execution_history.append(execution_result)
            print(f"[SUPERVISOR] Coordination {execution_id} failed with error: {str(e)}")
            
            return execution_result

    async def execute_with_constraints(self, task: Dict[str, Any], 
                                     constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task with specific constraints on specialist selection and coordination.
        
        Args:
            task: Task to be coordinated
            constraints: Constraints on execution (preferred agents, time limits, etc.)
            
        Returns:
            Constrained supervision results
        """
        print(f"[SUPERVISOR] Executing with constraints: {constraints}")
        
        # Apply constraints to the task
        constrained_task = task.copy()
        constrained_task.update({
            "execution_constraints": constraints,
            "constraint_mode": True
        })
        
        # Add constraint-specific metadata
        if "preferred_specialists" in constraints:
            constrained_task["preferred_specialists"] = constraints["preferred_specialists"]
        
        if "max_specialists" in constraints:
            constrained_task["max_specialists"] = constraints["max_specialists"]
        
        if "time_limit" in constraints:
            constrained_task["time_limit"] = constraints["time_limit"]
        
        # Execute with constraints
        result = await self.execute(constrained_task)
        result["constraints_applied"] = constraints
        
        return result

    def get_specialist_status(self) -> Dict[str, Any]:
        """
        Get status of all registered specialist agents.
        
        Returns:
            Status information for all specialists
        """
        specialist_status = {}
        
        for agent_id, agent in self.supervisor.specialist_agents.items():
            status = agent.get_status()
            specialist_status[agent_id] = {
                "agent_name": agent.name,
                "agent_description": agent.description,
                "capabilities": agent.get_capabilities(),
                "performance_metrics": status["performance_metrics"],
                "availability": True,  # Simplified - could be enhanced with actual availability checking
                "last_task_time": None  # Could track last task execution
            }
        
        return {
            "supervisor_agent": self.supervisor.name,
            "total_specialists": len(specialist_status),
            "specialists": specialist_status,
            "coordination_history": len(self.execution_history)
        }

    def get_delegation_patterns(self) -> Dict[str, Any]:
        """
        Analyze delegation patterns from execution history.
        
        Returns:
            Analysis of how tasks are typically delegated
        """
        if not self.execution_history:
            return {"no_history": "No executions to analyze"}
        
        # Analyze specialist usage
        specialist_usage = {}
        successful_delegations = 0
        total_delegations = 0
        
        for execution in self.execution_history:
            total_delegations += 1
            
            if execution["success"]:
                successful_delegations += 1
            
            # Extract specialist usage from coordination results
            if execution.get("coordination_result") and execution["coordination_result"].metadata:
                specialists_used = execution["coordination_result"].metadata.get("specialists_used", [])
                for specialist in specialists_used:
                    if specialist not in specialist_usage:
                        specialist_usage[specialist] = 0
                    specialist_usage[specialist] += 1
        
        # Calculate delegation patterns
        most_used_specialist = max(specialist_usage.items(), key=lambda x: x[1]) if specialist_usage else None
        
        return {
            "total_coordinations": total_delegations,
            "successful_coordinations": successful_delegations,
            "success_rate": successful_delegations / total_delegations if total_delegations > 0 else 0,
            "specialist_usage": specialist_usage,
            "most_used_specialist": {
                "agent_id": most_used_specialist[0] if most_used_specialist else None,
                "usage_count": most_used_specialist[1] if most_used_specialist else 0
            } if most_used_specialist else None,
            "average_specialists_per_task": sum(specialist_usage.values()) / total_delegations if total_delegations > 0 else 0
        }

    def optimize_specialist_allocation(self) -> Dict[str, Any]:
        """
        Provide recommendations for optimizing specialist allocation.
        
        Returns:
            Optimization recommendations based on execution history
        """
        delegation_patterns = self.get_delegation_patterns()
        
        if delegation_patterns.get("no_history"):
            return {"recommendation": "Need execution history to provide optimization recommendations"}
        
        recommendations = []
        
        # Check for underutilized specialists
        specialist_usage = delegation_patterns.get("specialist_usage", {})
        total_specialists = len(self.supervisor.specialist_agents)
        
        if specialist_usage:
            avg_usage = sum(specialist_usage.values()) / len(specialist_usage)
            underutilized = [agent_id for agent_id, usage in specialist_usage.items() if usage < avg_usage * 0.5]
            
            if underutilized:
                recommendations.append({
                    "type": "underutilized_specialists",
                    "description": f"{len(underutilized)} specialists are underutilized",
                    "affected_specialists": underutilized,
                    "suggestion": "Review capabilities and task matching for these specialists"
                })
        
        # Check success rate
        success_rate = delegation_patterns.get("success_rate", 0)
        if success_rate < 0.8:
            recommendations.append({
                "type": "low_success_rate",
                "description": f"Coordination success rate is {success_rate:.1%}",
                "suggestion": "Review task complexity and specialist capabilities matching"
            })
        
        # Check specialist diversity
        specialists_per_task = delegation_patterns.get("average_specialists_per_task", 0)
        if specialists_per_task < 2:
            recommendations.append({
                "type": "low_collaboration", 
                "description": f"Average {specialists_per_task:.1f} specialists per task",
                "suggestion": "Consider encouraging more cross-specialist collaboration"
            })
        
        return {
            "current_performance": delegation_patterns,
            "optimization_recommendations": recommendations,
            "next_steps": [
                "Monitor specialist performance metrics",
                "Adjust task decomposition strategies",
                "Consider specialist training or capability enhancement"
            ]
        }

    def get_pattern_configuration(self) -> Dict[str, Any]:
        """
        Get current supervisor pattern configuration.
        
        Returns:
            Pattern configuration details
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.name,
            "supervisor_agent": {
                "agent_id": self.supervisor.agent_id,
                "agent_name": self.supervisor.name,
                "capabilities": self.supervisor.get_capabilities()
            },
            "registered_specialists": len(self.supervisor.specialist_agents),
            "specialist_agents": [
                {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "capabilities": agent.get_capabilities()
                }
                for agent in self.supervisor.specialist_agents.values()
            ],
            "executions_count": len(self.execution_history),
            "coordination_mode": "hierarchical"
        }

    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        Get supervisor pattern execution metrics.
        
        Returns:
            Execution performance metrics
        """
        if not self.execution_history:
            return {"no_executions": True}
        
        successful_executions = [ex for ex in self.execution_history if ex["success"]]
        failed_executions = [ex for ex in self.execution_history if not ex["success"]]
        
        # Calculate metrics
        metrics = {
            "total_coordinations": len(self.execution_history),
            "successful_coordinations": len(successful_executions),
            "failed_coordinations": len(failed_executions),
            "success_rate": len(successful_executions) / len(self.execution_history),
            "supervisor_performance": self.supervisor.performance_metrics
        }
        
        # Calculate average coordination time for successful runs
        if successful_executions:
            total_time = sum(ex.get("total_execution_time", 0) for ex in successful_executions)
            metrics["average_coordination_time"] = total_time / len(successful_executions)
        else:
            metrics["average_coordination_time"] = 0
        
        # Add specialist performance summary
        specialist_metrics = {}
        for agent_id, agent in self.supervisor.specialist_agents.items():
            specialist_metrics[agent_id] = {
                "agent_name": agent.name,
                "tasks_completed": agent.performance_metrics["tasks_completed"],
                "success_rate": agent.performance_metrics["success_rate"],
                "average_confidence": agent.performance_metrics["average_confidence"]
            }
        
        metrics["specialist_performance"] = specialist_metrics
        
        return metrics

    def clear_history(self):
        """Clear execution history."""
        self.execution_history = []
        print(f"[SUPERVISOR] Execution history cleared")