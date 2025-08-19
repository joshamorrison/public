"""
Parallel Pattern

Concurrent agent execution with result aggregation and fusion.
Multiple agents work simultaneously on different aspects with result synthesis.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from ..agents.base_agent import BaseAgent, AgentResult


class ParallelPattern:
    """
    Parallel orchestration pattern for concurrent agent execution.
    
    The parallel pattern:
    - Executes multiple agents concurrently (fan-out)
    - Aggregates and fuses results from parallel execution (fan-in)
    - Supports load balancing and resource optimization
    - Handles partial failures gracefully
    - Provides result correlation and conflict resolution
    """

    def __init__(self, pattern_id: str = "parallel-001"):
        """
        Initialize the parallel pattern.
        
        Args:
            pattern_id: Unique identifier for this pattern instance
        """
        self.pattern_id = pattern_id
        self.name = "Parallel Pattern"
        self.description = "Concurrent execution with result aggregation"
        self.parallel_agents: List[Dict[str, Any]] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.result_fusion_strategy = "weighted_consensus"  # or "majority_vote", "best_confidence", etc.

    def add_parallel_agent(self, agent: BaseAgent, weight: float = 1.0, 
                          task_variant: Optional[str] = None,
                          timeout: Optional[float] = None) -> int:
        """
        Add an agent for parallel execution.
        
        Args:
            agent: Agent to execute in parallel
            weight: Weight for result fusion (higher weight = more influence)
            task_variant: Optional task variation identifier
            timeout: Optional timeout for this specific agent
            
        Returns:
            Agent index in parallel execution
        """
        agent_index = len(self.parallel_agents)
        
        agent_config = {
            "agent_index": agent_index,
            "agent": agent,
            "weight": weight,
            "task_variant": task_variant or f"variant_{agent_index}",
            "timeout": timeout,
            "added_at": datetime.now()
        }
        
        self.parallel_agents.append(agent_config)
        
        print(f"[PARALLEL] Added agent {agent_index}: {agent.name} (weight: {weight})")
        return agent_index

    def set_fusion_strategy(self, strategy: str):
        """
        Set the result fusion strategy.
        
        Args:
            strategy: Fusion strategy ('weighted_consensus', 'majority_vote', 'best_confidence', 'unanimous')
        """
        valid_strategies = ["weighted_consensus", "majority_vote", "best_confidence", "unanimous"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid fusion strategy. Must be one of: {valid_strategies}")
        
        self.result_fusion_strategy = strategy
        print(f"[PARALLEL] Fusion strategy set to: {strategy}")

    async def execute(self, base_task: Dict[str, Any], 
                     max_concurrent: Optional[int] = None,
                     timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute agents in parallel and fuse results.
        
        Args:
            base_task: Base task to be processed by all parallel agents
            max_concurrent: Maximum number of concurrent executions (None for unlimited)
            timeout: Global timeout for parallel execution
            
        Returns:
            Parallel execution results with fused output
        """
        execution_id = f"parallel_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        print(f"[PARALLEL] Starting parallel execution {execution_id}")
        print(f"[PARALLEL] Parallel agents: {len(self.parallel_agents)}")
        
        execution_result = {
            "execution_id": execution_id,
            "pattern_type": "parallel",
            "start_time": start_time,
            "agents_executed": [],
            "fusion_strategy": self.result_fusion_strategy,
            "fused_result": None,
            "individual_results": {},
            "success": True,
            "error_message": None,
            "partial_failure": False
        }
        
        try:
            # Prepare tasks for each agent
            agent_tasks = []
            for agent_config in self.parallel_agents:
                agent_task = await self._prepare_agent_task(base_task, agent_config, execution_id)
                agent_tasks.append((agent_config, agent_task))
            
            # Execute agents in parallel with concurrency control
            if max_concurrent:
                parallel_results = await self._execute_with_semaphore(agent_tasks, max_concurrent, timeout)
            else:
                parallel_results = await self._execute_all_parallel(agent_tasks, timeout)
            
            execution_result["individual_results"] = parallel_results
            execution_result["agents_executed"] = list(parallel_results.keys())
            
            # Check for partial failures
            successful_results = {k: v for k, v in parallel_results.items() if v.get("success", False)}
            failed_results = {k: v for k, v in parallel_results.items() if not v.get("success", False)}
            
            if failed_results:
                execution_result["partial_failure"] = True
                execution_result["failed_agents"] = list(failed_results.keys())
                print(f"[PARALLEL] Partial failure: {len(failed_results)} agents failed")
            
            # Proceed with fusion if we have at least one successful result
            if successful_results:
                fused_result = await self._fuse_results(successful_results, execution_result)
                execution_result["fused_result"] = fused_result
                execution_result["fusion_confidence"] = fused_result.get("confidence", 0)
            else:
                execution_result["success"] = False
                execution_result["error_message"] = "All parallel agents failed"
            
            execution_result["end_time"] = datetime.now()
            execution_result["total_execution_time"] = (execution_result["end_time"] - start_time).total_seconds()
            
            # Store execution history
            self.execution_history.append(execution_result)
            
            status = "SUCCESS" if execution_result["success"] else "FAILED"
            if execution_result.get("partial_failure"):
                status += " (PARTIAL)"
            
            print(f"[PARALLEL] Execution {execution_id} completed: {status}")
            return execution_result
            
        except Exception as e:
            execution_result["success"] = False
            execution_result["error_message"] = str(e)
            execution_result["end_time"] = datetime.now()
            
            self.execution_history.append(execution_result)
            print(f"[PARALLEL] Execution {execution_id} failed with error: {str(e)}")
            
            return execution_result

    async def _prepare_agent_task(self, base_task: Dict[str, Any], 
                                agent_config: Dict[str, Any], 
                                execution_id: str) -> Dict[str, Any]:
        """
        Prepare task variant for a specific parallel agent.
        
        Args:
            base_task: Base task specification
            agent_config: Agent configuration
            execution_id: Execution identifier
            
        Returns:
            Task prepared for the specific agent
        """
        agent_task = base_task.copy()
        agent_task.update({
            "parallel_execution_id": execution_id,
            "agent_index": agent_config["agent_index"],
            "task_variant": agent_config["task_variant"],
            "parallel_context": {
                "total_parallel_agents": len(self.parallel_agents),
                "agent_weight": agent_config["weight"],
                "fusion_strategy": self.result_fusion_strategy,
                "execution_mode": "parallel"
            }
        })
        
        # Add variant-specific modifications
        variant = agent_config["task_variant"]
        if variant and variant != f"variant_{agent_config['agent_index']}":
            agent_task["description"] = f"{base_task.get('description', '')}\nVariant focus: {variant}"
        
        return agent_task

    async def _execute_all_parallel(self, agent_tasks: List[tuple], 
                                  timeout: Optional[float]) -> Dict[str, Dict[str, Any]]:
        """
        Execute all agents in parallel without concurrency limits.
        
        Args:
            agent_tasks: List of (agent_config, task) tuples
            timeout: Global timeout for execution
            
        Returns:
            Dictionary of agent results
        """
        # Create execution tasks
        execution_tasks = []
        for agent_config, agent_task in agent_tasks:
            task_coro = self._execute_single_agent(agent_config, agent_task)
            execution_tasks.append(task_coro)
        
        # Execute with timeout
        try:
            if timeout:
                results = await asyncio.wait_for(asyncio.gather(*execution_tasks, return_exceptions=True), timeout=timeout)
            else:
                results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Process results
            parallel_results = {}
            for i, result in enumerate(results):
                agent_config = agent_tasks[i][0]
                agent_key = f"agent_{agent_config['agent_index']}"
                
                if isinstance(result, Exception):
                    parallel_results[agent_key] = {
                        "success": False,
                        "error": str(result),
                        "agent_name": agent_config["agent"].name,
                        "agent_index": agent_config["agent_index"]
                    }
                else:
                    parallel_results[agent_key] = result
            
            return parallel_results
            
        except asyncio.TimeoutError:
            print(f"[PARALLEL] Global timeout exceeded: {timeout}s")
            # Return partial results for any that completed
            return {}

    async def _execute_with_semaphore(self, agent_tasks: List[tuple], 
                                    max_concurrent: int, 
                                    timeout: Optional[float]) -> Dict[str, Dict[str, Any]]:
        """
        Execute agents with concurrency control using semaphore.
        
        Args:
            agent_tasks: List of (agent_config, task) tuples
            max_concurrent: Maximum concurrent executions
            timeout: Global timeout
            
        Returns:
            Dictionary of agent results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def controlled_execution(agent_config, agent_task):
            async with semaphore:
                return await self._execute_single_agent(agent_config, agent_task)
        
        # Create controlled execution tasks
        controlled_tasks = []
        for agent_config, agent_task in agent_tasks:
            task_coro = controlled_execution(agent_config, agent_task)
            controlled_tasks.append(task_coro)
        
        # Execute with controlled concurrency
        try:
            if timeout:
                results = await asyncio.wait_for(asyncio.gather(*controlled_tasks, return_exceptions=True), timeout=timeout)
            else:
                results = await asyncio.gather(*controlled_tasks, return_exceptions=True)
            
            # Process results
            parallel_results = {}
            for i, result in enumerate(results):
                agent_config = agent_tasks[i][0]
                agent_key = f"agent_{agent_config['agent_index']}"
                
                if isinstance(result, Exception):
                    parallel_results[agent_key] = {
                        "success": False,
                        "error": str(result),
                        "agent_name": agent_config["agent"].name,
                        "agent_index": agent_config["agent_index"]
                    }
                else:
                    parallel_results[agent_key] = result
            
            return parallel_results
            
        except asyncio.TimeoutError:
            print(f"[PARALLEL] Global timeout exceeded: {timeout}s")
            return {}

    async def _execute_single_agent(self, agent_config: Dict[str, Any], 
                                  agent_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single agent with its specific configuration.
        
        Args:
            agent_config: Agent configuration
            agent_task: Task for the agent
            
        Returns:
            Agent execution result
        """
        agent = agent_config["agent"]
        agent_timeout = agent_config.get("timeout")
        
        print(f"[PARALLEL] Executing {agent.name} (index: {agent_config['agent_index']})")
        start_time = datetime.now()
        
        try:
            if agent_timeout:
                agent_result = await asyncio.wait_for(agent.process_task(agent_task), timeout=agent_timeout)
            else:
                agent_result = await agent.process_task(agent_task)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": agent_result.success,
                "agent_result": agent_result,
                "agent_name": agent.name,
                "agent_index": agent_config["agent_index"],
                "weight": agent_config["weight"],
                "execution_time": execution_time,
                "task_variant": agent_config["task_variant"]
            }
            
        except asyncio.TimeoutError:
            print(f"[PARALLEL] Agent {agent.name} timed out after {agent_timeout}s")
            return {
                "success": False,
                "error": f"Agent timeout after {agent_timeout}s",
                "agent_name": agent.name,
                "agent_index": agent_config["agent_index"],
                "execution_time": agent_timeout
            }
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            print(f"[PARALLEL] Agent {agent.name} failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent_name": agent.name,
                "agent_index": agent_config["agent_index"],
                "execution_time": execution_time
            }

    async def _fuse_results(self, successful_results: Dict[str, Dict[str, Any]], 
                          execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse results from parallel agents using the configured strategy.
        
        Args:
            successful_results: Results from successful agents
            execution_context: Execution context information
            
        Returns:
            Fused result
        """
        print(f"[PARALLEL] Fusing {len(successful_results)} results using {self.result_fusion_strategy}")
        
        if self.result_fusion_strategy == "weighted_consensus":
            return await self._fuse_weighted_consensus(successful_results)
        elif self.result_fusion_strategy == "majority_vote":
            return await self._fuse_majority_vote(successful_results)
        elif self.result_fusion_strategy == "best_confidence":
            return await self._fuse_best_confidence(successful_results)
        elif self.result_fusion_strategy == "unanimous":
            return await self._fuse_unanimous(successful_results)
        else:
            # Default to weighted consensus
            return await self._fuse_weighted_consensus(successful_results)

    async def _fuse_weighted_consensus(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse results using weighted consensus."""
        total_weight = sum(result["weight"] for result in results.values())
        weighted_confidence = sum(result["agent_result"].confidence * result["weight"] for result in results.values()) / total_weight
        
        # Create comprehensive fused content
        fusion_content = []
        fusion_content.append("PARALLEL EXECUTION RESULTS - WEIGHTED CONSENSUS")
        fusion_content.append("=" * 55)
        fusion_content.append(f"Results fused from {len(results)} agents")
        fusion_content.append(f"Weighted confidence: {weighted_confidence:.1%}")
        fusion_content.append("")
        
        # Include each agent's contribution weighted by importance
        sorted_results = sorted(results.items(), key=lambda x: x[1]["weight"], reverse=True)
        
        for agent_key, result in sorted_results:
            agent_result = result["agent_result"]
            weight_percentage = (result["weight"] / total_weight) * 100
            
            fusion_content.append(f"AGENT: {result['agent_name']} (Weight: {weight_percentage:.1f}%)")
            fusion_content.append(f"Confidence: {agent_result.confidence:.1%}")
            fusion_content.append(f"Content: {agent_result.content[:200]}...")
            fusion_content.append("")
        
        return {
            "fusion_strategy": "weighted_consensus",
            "content": "\n".join(fusion_content),
            "confidence": weighted_confidence,
            "contributing_agents": len(results),
            "fusion_metadata": {
                "total_weight": total_weight,
                "agent_contributions": {
                    agent_key: {
                        "weight": result["weight"],
                        "confidence": result["agent_result"].confidence,
                        "contribution_percentage": (result["weight"] / total_weight) * 100
                    }
                    for agent_key, result in results.items()
                }
            }
        }

    async def _fuse_best_confidence(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse results by selecting the highest confidence result."""
        best_result = max(results.values(), key=lambda x: x["agent_result"].confidence)
        
        return {
            "fusion_strategy": "best_confidence",
            "content": f"BEST CONFIDENCE RESULT\n{best_result['agent_result'].content}",
            "confidence": best_result["agent_result"].confidence,
            "contributing_agents": 1,
            "selected_agent": best_result["agent_name"],
            "fusion_metadata": {
                "all_confidences": {
                    agent_key: result["agent_result"].confidence
                    for agent_key, result in results.items()
                }
            }
        }

    async def _fuse_majority_vote(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse results using majority vote (simplified implementation)."""
        # Simplified majority vote - in practice would need more sophisticated content comparison
        avg_confidence = sum(result["agent_result"].confidence for result in results.values()) / len(results)
        
        return {
            "fusion_strategy": "majority_vote",
            "content": "MAJORITY VOTE FUSION\n" + "\n".join([f"{r['agent_name']}: {r['agent_result'].content[:100]}..." for r in results.values()]),
            "confidence": avg_confidence,
            "contributing_agents": len(results),
            "fusion_metadata": {"vote_count": len(results)}
        }

    async def _fuse_unanimous(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse results requiring unanimous agreement (simplified)."""
        min_confidence = min(result["agent_result"].confidence for result in results.values())
        
        return {
            "fusion_strategy": "unanimous", 
            "content": "UNANIMOUS CONSENSUS\n" + "\n".join([r["agent_result"].content for r in results.values()]),
            "confidence": min_confidence,  # Conservative - lowest confidence
            "contributing_agents": len(results),
            "fusion_metadata": {"consensus_strength": min_confidence}
        }

    def get_pattern_configuration(self) -> Dict[str, Any]:
        """Get current parallel pattern configuration."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.name,
            "agents_count": len(self.parallel_agents),
            "fusion_strategy": self.result_fusion_strategy,
            "agents": [
                {
                    "agent_index": config["agent_index"],
                    "agent_name": config["agent"].name,
                    "agent_id": config["agent"].agent_id,
                    "weight": config["weight"],
                    "task_variant": config["task_variant"],
                    "timeout": config.get("timeout")
                }
                for config in self.parallel_agents
            ],
            "executions_count": len(self.execution_history)
        }

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get parallel pattern execution metrics."""
        if not self.execution_history:
            return {"no_executions": True}
        
        successful_executions = [ex for ex in self.execution_history if ex["success"]]
        partial_failures = [ex for ex in self.execution_history if ex.get("partial_failure", False)]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "partial_failures": len(partial_failures),
            "complete_failures": len(self.execution_history) - len(successful_executions) - len(partial_failures),
            "success_rate": len(successful_executions) / len(self.execution_history),
            "partial_failure_rate": len(partial_failures) / len(self.execution_history),
            "average_agents_per_execution": sum(len(ex.get("agents_executed", [])) for ex in self.execution_history) / len(self.execution_history),
            "fusion_strategy": self.result_fusion_strategy
        }

    def clear_history(self):
        """Clear execution history."""
        self.execution_history = []
        print(f"[PARALLEL] Execution history cleared")