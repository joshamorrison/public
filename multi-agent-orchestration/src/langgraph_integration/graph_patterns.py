"""
LangGraph Pattern Implementations

Official LangGraph implementations of the four core multi-agent patterns.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .state_management import MultiAgentState, WorkflowState, StateManager
from ..agents.base_agent import BaseAgent

# LangGraph imports with fallback
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    START = "START"


class LangGraphPipeline:
    """
    LangGraph implementation of pipeline pattern.
    
    Sequential agent execution with state passing and quality gates.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize pipeline with agents.
        
        Args:
            agents: List of agents to execute in sequence
        """
        self.agents = agents
        self.state_manager = StateManager()
        self.graph = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph state graph for pipeline."""
        # Create state graph
        self.graph = StateGraph(MultiAgentState)
        
        # Add agent nodes
        for i, agent in enumerate(self.agents):
            node_name = f"agent_{agent.agent_id}"
            self.graph.add_node(node_name, self._create_agent_node(agent))
        
        # Add start and end nodes
        self.graph.add_node("start", self._start_node)
        self.graph.add_node("end", self._end_node)
        
        # Connect nodes in sequence
        self.graph.set_entry_point("start")
        prev_node = "start"
        
        for agent in self.agents:
            node_name = f"agent_{agent.agent_id}"
            self.graph.add_edge(prev_node, node_name)
            prev_node = node_name
        
        # Connect to end
        self.graph.add_edge(prev_node, "end")
        self.graph.set_finish_point("end")
        
        # Compile graph
        if MemorySaver:
            self.compiled_graph = self.graph.compile(checkpointer=MemorySaver())
        else:
            self.compiled_graph = self.graph.compile()
    
    def _create_agent_node(self, agent: BaseAgent) -> Callable:
        """Create a graph node for an agent."""
        async def agent_node(state: MultiAgentState) -> MultiAgentState:
            # Update current agent
            new_state = state.copy()
            new_state["current_agent"] = agent.agent_id
            new_state["current_step"] = f"agent_{agent.agent_id}"
            
            try:
                # Prepare task from state
                task = {
                    "type": "pipeline_step",
                    "description": f"Pipeline step for {agent.agent_id}",
                    "input_data": state["input_data"],
                    "intermediate_data": state["intermediate_data"],
                    "previous_results": state["agent_results"]
                }
                
                # Execute agent
                result = await agent.process_with_tools(task)
                
                # Update state with result
                new_state["agent_results"][agent.agent_id] = {
                    "content": result.content,
                    "confidence": result.confidence,
                    "metadata": result.metadata,
                    "success": result.success,
                    "timestamp": datetime.now()
                }
                
                # Store confidence score
                new_state["confidence_scores"][agent.agent_id] = result.confidence
                
                # Update intermediate data for next agent
                new_state["intermediate_data"][f"output_{agent.agent_id}"] = result.content
                
                # Mark step as completed
                new_state["completed_steps"].append(f"agent_{agent.agent_id}")
                
                return new_state
                
            except Exception as e:
                # Handle errors
                error_info = {
                    "agent_id": agent.agent_id,
                    "error": str(e),
                    "timestamp": datetime.now(),
                    "step": f"agent_{agent.agent_id}"
                }
                new_state["errors"].append(error_info)
                new_state["retry_count"] += 1
                
                return new_state
        
        return agent_node
    
    def _start_node(self, state: MultiAgentState) -> MultiAgentState:
        """Start node for pipeline."""
        new_state = state.copy()
        new_state["current_step"] = "pipeline_start"
        new_state["completed_steps"].append("start")
        return new_state
    
    def _end_node(self, state: MultiAgentState) -> MultiAgentState:
        """End node for pipeline."""
        new_state = state.copy()
        new_state["current_step"] = "pipeline_complete"
        
        # Aggregate final output
        final_output = {}
        for agent_id, result in state["agent_results"].items():
            final_output[f"{agent_id}_output"] = result["content"]
        
        new_state["output_data"] = final_output
        new_state["completed_steps"].append("end")
        
        return new_state
    
    async def execute(self, input_data: Dict[str, Any], 
                     workflow_id: str = None) -> MultiAgentState:
        """Execute pipeline workflow."""
        if not workflow_id:
            workflow_id = f"pipeline_{int(datetime.now().timestamp())}"
        
        # Create initial state
        initial_state = self.state_manager.create_initial_state(
            workflow_id, "pipeline", input_data
        )
        
        if LANGGRAPH_AVAILABLE and self.compiled_graph:
            # Execute using LangGraph
            result = await self.compiled_graph.ainvoke(initial_state)
            return result
        else:
            # Fallback execution
            return await self._fallback_execute(initial_state)
    
    async def _fallback_execute(self, state: MultiAgentState) -> MultiAgentState:
        """Fallback execution without LangGraph."""
        current_state = state
        
        # Execute start
        current_state = self._start_node(current_state)
        
        # Execute agents in sequence
        for agent in self.agents:
            agent_node = self._create_agent_node(agent)
            current_state = await agent_node(current_state)
            
            # Check for errors
            if current_state["errors"] and current_state["retry_count"] >= 3:
                break
        
        # Execute end
        current_state = self._end_node(current_state)
        
        return current_state


class LangGraphSupervisor:
    """
    LangGraph implementation of supervisor pattern.
    
    Hierarchical coordination with intelligent task delegation.
    """
    
    def __init__(self, supervisor_agent: BaseAgent, worker_agents: List[BaseAgent]):
        """
        Initialize supervisor pattern.
        
        Args:
            supervisor_agent: Coordinating supervisor agent
            worker_agents: Worker agents to delegate to
        """
        self.supervisor = supervisor_agent
        self.workers = {agent.agent_id: agent for agent in worker_agents}
        self.state_manager = StateManager()
        self.graph = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph for supervisor pattern."""
        self.graph = StateGraph(WorkflowState)
        
        # Add supervisor node
        self.graph.add_node("supervisor", self._supervisor_node)
        
        # Add worker nodes
        for agent_id, agent in self.workers.items():
            self.graph.add_node(f"worker_{agent_id}", self._create_worker_node(agent))
        
        # Add coordination nodes
        self.graph.add_node("start", self._start_node)
        self.graph.add_node("aggregate", self._aggregate_node)
        self.graph.add_node("end", self._end_node)
        
        # Build routing logic
        self.graph.set_entry_point("start")
        self.graph.add_edge("start", "supervisor")
        
        # Supervisor routes to workers
        self.graph.add_conditional_edges(
            "supervisor",
            self._supervisor_router,
            {
                **{f"worker_{agent_id}": f"worker_{agent_id}" for agent_id in self.workers.keys()},
                "aggregate": "aggregate"
            }
        )
        
        # Workers return to supervisor
        for agent_id in self.workers.keys():
            self.graph.add_edge(f"worker_{agent_id}", "supervisor")
        
        # Final aggregation
        self.graph.add_edge("aggregate", "end")
        self.graph.set_finish_point("end")
        
        # Compile
        if MemorySaver:
            self.compiled_graph = self.graph.compile(checkpointer=MemorySaver())
        else:
            self.compiled_graph = self.graph.compile()
    
    def _supervisor_router(self, state: WorkflowState) -> str:
        """Route decision from supervisor."""
        # Check if all required work is completed
        if state.get("supervisor_complete", False):
            return "aggregate"
        
        # Get next assignment from supervisor decisions
        if state["supervisor_decisions"]:
            latest_decision = state["supervisor_decisions"][-1]
            return f"worker_{latest_decision['assigned_agent']}"
        
        return "aggregate"
    
    async def _supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """Supervisor coordination node."""
        new_state = state.copy()
        new_state["current_agent"] = self.supervisor.agent_id
        
        # Prepare task for supervisor
        task = {
            "type": "coordination",
            "description": "Coordinate and delegate work to specialist agents",
            "input_data": state["input_data"],
            "worker_capabilities": {
                agent_id: agent.get_capabilities() 
                for agent_id, agent in self.workers.items()
            },
            "completed_work": state["agent_results"],
            "current_assignments": state["agent_assignments"]
        }
        
        try:
            result = await self.supervisor.process_with_tools(task)
            
            # Parse supervisor decision
            decision = self._parse_supervisor_decision(result.content, state)
            new_state["supervisor_decisions"].append(decision)
            
            # Update assignments
            if decision["action"] == "assign":
                new_state["agent_assignments"][decision["assigned_agent"]] = decision["task_description"]
            elif decision["action"] == "complete":
                new_state["supervisor_complete"] = True
            
            return new_state
            
        except Exception as e:
            new_state["errors"].append({
                "agent_id": self.supervisor.agent_id,
                "error": str(e),
                "timestamp": datetime.now()
            })
            new_state["supervisor_complete"] = True  # Fail-safe
            return new_state
    
    def _parse_supervisor_decision(self, content: str, state: WorkflowState) -> Dict[str, Any]:
        """Parse supervisor's decision from response."""
        # Simple parsing logic - in production, use more sophisticated NLP
        content_lower = content.lower()
        
        # Check for completion signals
        if any(word in content_lower for word in ["complete", "finished", "done", "final"]):
            return {
                "action": "complete",
                "reasoning": content,
                "timestamp": datetime.now()
            }
        
        # Determine best agent for remaining work
        available_agents = [aid for aid in self.workers.keys() if aid not in state["agent_assignments"]]
        
        if available_agents:
            # Simple assignment logic
            selected_agent = available_agents[0]
            return {
                "action": "assign",
                "assigned_agent": selected_agent,
                "task_description": "Process assigned work based on supervisor guidance",
                "reasoning": content,
                "timestamp": datetime.now()
            }
        
        return {
            "action": "complete",
            "reasoning": "No available agents for assignment",
            "timestamp": datetime.now()
        }
    
    def _create_worker_node(self, agent: BaseAgent) -> Callable:
        """Create worker node for agent."""
        async def worker_node(state: WorkflowState) -> WorkflowState:
            new_state = state.copy()
            new_state["current_agent"] = agent.agent_id
            
            # Get assignment from supervisor
            task_description = state["agent_assignments"].get(agent.agent_id, "Process data")
            
            task = {
                "type": "specialist_work",
                "description": task_description,
                "input_data": state["input_data"],
                "supervisor_guidance": state["supervisor_decisions"][-1] if state["supervisor_decisions"] else {},
                "context": state["intermediate_data"]
            }
            
            try:
                result = await agent.process_with_tools(task)
                
                new_state["agent_results"][agent.agent_id] = {
                    "content": result.content,
                    "confidence": result.confidence,
                    "metadata": result.metadata,
                    "success": result.success,
                    "timestamp": datetime.now()
                }
                
                # Mark assignment as completed
                if agent.agent_id in new_state["agent_assignments"]:
                    del new_state["agent_assignments"][agent.agent_id]
                
                return new_state
                
            except Exception as e:
                new_state["errors"].append({
                    "agent_id": agent.agent_id,
                    "error": str(e),
                    "timestamp": datetime.now()
                })
                return new_state
        
        return worker_node
    
    def _start_node(self, state: WorkflowState) -> WorkflowState:
        """Start node for supervisor pattern."""
        new_state = state.copy()
        new_state["current_step"] = "supervisor_start"
        new_state["supervisor_complete"] = False
        return new_state
    
    def _aggregate_node(self, state: WorkflowState) -> WorkflowState:
        """Aggregate results from all workers."""
        new_state = state.copy()
        new_state["current_step"] = "aggregating_results"
        
        # Combine all worker results
        aggregated_output = {
            "supervisor_coordination": state["supervisor_decisions"],
            "worker_results": state["agent_results"],
            "final_synthesis": "Results aggregated from all specialist agents"
        }
        
        new_state["output_data"] = aggregated_output
        return new_state
    
    def _end_node(self, state: WorkflowState) -> WorkflowState:
        """End node for supervisor pattern."""
        new_state = state.copy()
        new_state["current_step"] = "supervisor_complete"
        new_state["completed_steps"].append("end")
        return new_state
    
    async def execute(self, input_data: Dict[str, Any], 
                     workflow_id: str = None, max_iterations: int = 5) -> WorkflowState:
        """Execute supervisor workflow."""
        if not workflow_id:
            workflow_id = f"supervisor_{int(datetime.now().timestamp())}"
        
        initial_state = self.state_manager.create_workflow_state(
            workflow_id, "supervisor", input_data, max_iterations
        )
        
        if LANGGRAPH_AVAILABLE and self.compiled_graph:
            result = await self.compiled_graph.ainvoke(initial_state)
            return result
        else:
            return await self._fallback_execute(initial_state)
    
    async def _fallback_execute(self, state: WorkflowState) -> WorkflowState:
        """Fallback execution without LangGraph."""
        current_state = state
        
        # Simple coordination loop
        for iteration in range(state["max_iterations"]):
            current_state = await self._supervisor_node(current_state)
            
            if current_state.get("supervisor_complete", False):
                break
            
            # Execute assigned workers
            for agent_id in list(current_state["agent_assignments"].keys()):
                if agent_id in self.workers:
                    worker_node = self._create_worker_node(self.workers[agent_id])
                    current_state = await worker_node(current_state)
        
        current_state = self._aggregate_node(current_state)
        current_state = self._end_node(current_state)
        
        return current_state


class LangGraphParallel:
    """
    LangGraph implementation of parallel pattern.
    
    Concurrent agent execution with result fusion.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        """Initialize parallel pattern with agents."""
        self.agents = agents
        self.state_manager = StateManager()
        self.graph = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph for parallel pattern."""
        self.graph = StateGraph(MultiAgentState)
        
        # Add nodes
        self.graph.add_node("start", self._start_node)
        self.graph.add_node("distribute", self._distribute_node)
        
        for agent in self.agents:
            self.graph.add_node(f"agent_{agent.agent_id}", self._create_agent_node(agent))
        
        self.graph.add_node("collect", self._collect_node)
        self.graph.add_node("fuse", self._fuse_node)
        self.graph.add_node("end", self._end_node)
        
        # Connect nodes
        self.graph.set_entry_point("start")
        self.graph.add_edge("start", "distribute")
        
        # Fan-out to all agents
        for agent in self.agents:
            self.graph.add_edge("distribute", f"agent_{agent.agent_id}")
        
        # Fan-in from all agents
        for agent in self.agents:
            self.graph.add_edge(f"agent_{agent.agent_id}", "collect")
        
        self.graph.add_edge("collect", "fuse")
        self.graph.add_edge("fuse", "end")
        self.graph.set_finish_point("end")
        
        # Compile
        if MemorySaver:
            self.compiled_graph = self.graph.compile(checkpointer=MemorySaver())
        else:
            self.compiled_graph = self.graph.compile()
    
    def _create_agent_node(self, agent: BaseAgent) -> Callable:
        """Create parallel agent node."""
        async def agent_node(state: MultiAgentState) -> MultiAgentState:
            new_state = state.copy()
            new_state["current_agent"] = agent.agent_id
            
            task = {
                "type": "parallel_processing",
                "description": f"Parallel processing by {agent.agent_id}",
                "input_data": state["input_data"],
                "agent_perspective": agent.get_capabilities()[0] if agent.get_capabilities() else "general"
            }
            
            try:
                result = await agent.process_with_tools(task)
                
                new_state["parallel_results"][agent.agent_id] = {
                    "content": result.content,
                    "confidence": result.confidence,
                    "metadata": result.metadata,
                    "success": result.success,
                    "timestamp": datetime.now()
                }
                
                return new_state
                
            except Exception as e:
                new_state["errors"].append({
                    "agent_id": agent.agent_id,
                    "error": str(e),
                    "timestamp": datetime.now()
                })
                return new_state
        
        return agent_node
    
    def _start_node(self, state: MultiAgentState) -> MultiAgentState:
        """Start parallel processing."""
        new_state = state.copy()
        new_state["current_step"] = "parallel_start"
        return new_state
    
    def _distribute_node(self, state: MultiAgentState) -> MultiAgentState:
        """Distribute work to parallel agents."""
        new_state = state.copy()
        new_state["current_step"] = "distributing_work"
        return new_state
    
    def _collect_node(self, state: MultiAgentState) -> MultiAgentState:
        """Collect results from parallel agents."""
        new_state = state.copy()
        new_state["current_step"] = "collecting_results"
        
        # Move parallel results to main agent results
        for agent_id, result in state["parallel_results"].items():
            new_state["agent_results"][agent_id] = result
        
        return new_state
    
    def _fuse_node(self, state: MultiAgentState) -> MultiAgentState:
        """Fuse parallel results into final output."""
        new_state = state.copy()
        new_state["current_step"] = "fusing_results"
        
        # Simple result fusion
        fused_content = []
        total_confidence = 0
        successful_agents = 0
        
        for agent_id, result in state["parallel_results"].items():
            if result["success"]:
                fused_content.append(f"[{agent_id}]: {result['content']}")
                total_confidence += result["confidence"]
                successful_agents += 1
        
        avg_confidence = total_confidence / max(successful_agents, 1)
        
        new_state["output_data"] = {
            "fused_results": "\n\n".join(fused_content),
            "individual_results": state["parallel_results"],
            "fusion_confidence": avg_confidence,
            "successful_agents": successful_agents,
            "total_agents": len(self.agents)
        }
        
        return new_state
    
    def _end_node(self, state: MultiAgentState) -> MultiAgentState:
        """End parallel processing."""
        new_state = state.copy()
        new_state["current_step"] = "parallel_complete"
        new_state["completed_steps"].append("end")
        return new_state
    
    async def execute(self, input_data: Dict[str, Any], 
                     workflow_id: str = None) -> MultiAgentState:
        """Execute parallel workflow."""
        if not workflow_id:
            workflow_id = f"parallel_{int(datetime.now().timestamp())}"
        
        initial_state = self.state_manager.create_initial_state(
            workflow_id, "parallel", input_data
        )
        
        if LANGGRAPH_AVAILABLE and self.compiled_graph:
            result = await self.compiled_graph.ainvoke(initial_state)
            return result
        else:
            return await self._fallback_execute(initial_state)
    
    async def _fallback_execute(self, state: MultiAgentState) -> MultiAgentState:
        """Fallback parallel execution."""
        import asyncio
        
        current_state = self._start_node(state)
        current_state = self._distribute_node(current_state)
        
        # Execute agents in parallel
        agent_tasks = []
        for agent in self.agents:
            agent_node = self._create_agent_node(agent)
            agent_tasks.append(agent_node(current_state))
        
        # Wait for all agents to complete
        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Merge results
        for result in agent_results:
            if isinstance(result, MultiAgentState):
                current_state["parallel_results"].update(result["parallel_results"])
                current_state["errors"].extend(result["errors"])
        
        current_state = self._collect_node(current_state)
        current_state = self._fuse_node(current_state)
        current_state = self._end_node(current_state)
        
        return current_state


class LangGraphReflective:
    """
    LangGraph implementation of reflective pattern.
    
    Self-improving workflows with feedback loops and iteration.
    """
    
    def __init__(self, primary_agent: BaseAgent, critic_agent: BaseAgent):
        """
        Initialize reflective pattern.
        
        Args:
            primary_agent: Main processing agent
            critic_agent: Feedback and evaluation agent
        """
        self.primary_agent = primary_agent
        self.critic_agent = critic_agent
        self.state_manager = StateManager()
        self.graph = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph for reflective pattern."""
        self.graph = StateGraph(WorkflowState)
        
        # Add nodes
        self.graph.add_node("start", self._start_node)
        self.graph.add_node("primary_work", self._primary_work_node)
        self.graph.add_node("critic_review", self._critic_review_node)
        self.graph.add_node("should_continue", self._should_continue_node)
        self.graph.add_node("refine", self._refine_node)
        self.graph.add_node("finalize", self._finalize_node)
        self.graph.add_node("end", self._end_node)
        
        # Connect nodes with conditional logic
        self.graph.set_entry_point("start")
        self.graph.add_edge("start", "primary_work")
        self.graph.add_edge("primary_work", "critic_review")
        
        # Conditional routing based on critic feedback
        self.graph.add_conditional_edges(
            "critic_review",
            self._should_continue_router,
            {
                "continue": "refine",
                "finalize": "finalize"
            }
        )
        
        self.graph.add_edge("refine", "primary_work")  # Loop back
        self.graph.add_edge("finalize", "end")
        self.graph.set_finish_point("end")
        
        # Compile
        if MemorySaver:
            self.compiled_graph = self.graph.compile(checkpointer=MemorySaver())
        else:
            self.compiled_graph = self.graph.compile()
    
    def _should_continue_router(self, state: WorkflowState) -> str:
        """Route decision for continuation."""
        # Check convergence conditions
        if state["iteration_count"] >= state["max_iterations"]:
            return "finalize"
        
        # Check quality threshold
        if state["feedback"]:
            latest_feedback = state["feedback"][-1]
            if latest_feedback.get("quality_score", 0) >= state["convergence_threshold"]:
                return "finalize"
        
        return "continue"
    
    async def _start_node(self, state: WorkflowState) -> WorkflowState:
        """Start reflective processing."""
        new_state = state.copy()
        new_state["current_step"] = "reflective_start"
        new_state["iteration_count"] = 0
        return new_state
    
    async def _primary_work_node(self, state: WorkflowState) -> WorkflowState:
        """Primary agent work."""
        new_state = state.copy()
        new_state["current_agent"] = self.primary_agent.agent_id
        new_state["iteration_count"] += 1
        
        # Prepare task with previous feedback
        task = {
            "type": "iterative_work",
            "description": "Primary work with iterative improvement",
            "input_data": state["input_data"],
            "iteration": state["iteration_count"],
            "previous_feedback": state["feedback"][-1] if state["feedback"] else None,
            "improvement_areas": [f["improvement_suggestions"] for f in state["feedback"] if "improvement_suggestions" in f]
        }
        
        try:
            result = await self.primary_agent.process_with_tools(task)
            
            new_state["agent_results"][f"primary_iteration_{state['iteration_count']}"] = {
                "content": result.content,
                "confidence": result.confidence,
                "metadata": result.metadata,
                "success": result.success,
                "iteration": state["iteration_count"],
                "timestamp": datetime.now()
            }
            
            # Store latest work for critic review
            new_state["intermediate_data"]["latest_work"] = result.content
            
            return new_state
            
        except Exception as e:
            new_state["errors"].append({
                "agent_id": self.primary_agent.agent_id,
                "error": str(e),
                "iteration": state["iteration_count"],
                "timestamp": datetime.now()
            })
            return new_state
    
    async def _critic_review_node(self, state: WorkflowState) -> WorkflowState:
        """Critic agent review and feedback."""
        new_state = state.copy()
        new_state["current_agent"] = self.critic_agent.agent_id
        
        latest_work = state["intermediate_data"].get("latest_work", "")
        
        task = {
            "type": "quality_review",
            "description": "Review and provide feedback on work quality",
            "work_to_review": latest_work,
            "original_requirements": state["input_data"],
            "iteration": state["iteration_count"],
            "previous_iterations": [
                result for key, result in state["agent_results"].items()
                if key.startswith("primary_iteration_")
            ]
        }
        
        try:
            result = await self.critic_agent.process_with_tools(task)
            
            # Parse feedback
            feedback = self._parse_critic_feedback(result.content, result.confidence)
            new_state["feedback"].append(feedback)
            
            new_state["agent_results"][f"critic_iteration_{state['iteration_count']}"] = {
                "content": result.content,
                "confidence": result.confidence,
                "metadata": result.metadata,
                "success": result.success,
                "timestamp": datetime.now()
            }
            
            return new_state
            
        except Exception as e:
            new_state["errors"].append({
                "agent_id": self.critic_agent.agent_id,
                "error": str(e),
                "iteration": state["iteration_count"],
                "timestamp": datetime.now()
            })
            # Add default feedback to continue
            new_state["feedback"].append({
                "quality_score": 0.5,
                "should_continue": True,
                "error": str(e),
                "timestamp": datetime.now()
            })
            return new_state
    
    def _parse_critic_feedback(self, content: str, confidence: float) -> Dict[str, Any]:
        """Parse critic feedback into structured format."""
        content_lower = content.lower()
        
        # Determine quality score
        quality_indicators = {
            "excellent": 0.95,
            "good": 0.80,
            "satisfactory": 0.70,
            "adequate": 0.60,
            "poor": 0.40,
            "unacceptable": 0.20
        }
        
        quality_score = confidence  # Default to confidence
        for indicator, score in quality_indicators.items():
            if indicator in content_lower:
                quality_score = score
                break
        
        # Determine if should continue
        should_continue = True
        if any(word in content_lower for word in ["complete", "finished", "ready", "final"]):
            should_continue = False
        
        # Extract improvement suggestions (simple heuristic)
        improvement_suggestions = []
        if "improve" in content_lower:
            sentences = content.split(".")
            for sentence in sentences:
                if "improve" in sentence.lower():
                    improvement_suggestions.append(sentence.strip())
        
        return {
            "quality_score": quality_score,
            "should_continue": should_continue,
            "feedback_content": content,
            "improvement_suggestions": improvement_suggestions,
            "confidence": confidence,
            "timestamp": datetime.now()
        }
    
    async def _should_continue_node(self, state: WorkflowState) -> WorkflowState:
        """Decision node for continuation."""
        return state  # Routing handled by conditional edges
    
    async def _refine_node(self, state: WorkflowState) -> WorkflowState:
        """Prepare for next iteration."""
        new_state = state.copy()
        new_state["current_step"] = "refining_for_next_iteration"
        return new_state
    
    async def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize reflective process."""
        new_state = state.copy()
        new_state["current_step"] = "finalizing_reflective_work"
        
        # Get best iteration result
        best_iteration = state["iteration_count"]
        best_result = state["agent_results"].get(f"primary_iteration_{best_iteration}", {})
        
        new_state["output_data"] = {
            "final_result": best_result.get("content", ""),
            "total_iterations": state["iteration_count"],
            "feedback_history": state["feedback"],
            "improvement_trajectory": [
                f["quality_score"] for f in state["feedback"]
                if "quality_score" in f
            ],
            "final_quality_score": state["feedback"][-1].get("quality_score", 0) if state["feedback"] else 0
        }
        
        return new_state
    
    async def _end_node(self, state: WorkflowState) -> WorkflowState:
        """End reflective processing."""
        new_state = state.copy()
        new_state["current_step"] = "reflective_complete"
        new_state["completed_steps"].append("end")
        return new_state
    
    async def execute(self, input_data: Dict[str, Any], 
                     workflow_id: str = None, max_iterations: int = 3,
                     convergence_threshold: float = 0.85) -> WorkflowState:
        """Execute reflective workflow."""
        if not workflow_id:
            workflow_id = f"reflective_{int(datetime.now().timestamp())}"
        
        initial_state = self.state_manager.create_workflow_state(
            workflow_id, "reflective", input_data, max_iterations
        )
        initial_state["convergence_threshold"] = convergence_threshold
        
        if LANGGRAPH_AVAILABLE and self.compiled_graph:
            result = await self.compiled_graph.ainvoke(initial_state)
            return result
        else:
            return await self._fallback_execute(initial_state)
    
    async def _fallback_execute(self, state: WorkflowState) -> WorkflowState:
        """Fallback reflective execution."""
        current_state = await self._start_node(state)
        
        while current_state["iteration_count"] < current_state["max_iterations"]:
            # Primary work
            current_state = await self._primary_work_node(current_state)
            
            # Critic review
            current_state = await self._critic_review_node(current_state)
            
            # Check if should continue
            if current_state["feedback"]:
                latest_feedback = current_state["feedback"][-1]
                if (latest_feedback.get("quality_score", 0) >= current_state["convergence_threshold"] or
                    not latest_feedback.get("should_continue", True)):
                    break
        
        current_state = await self._finalize_node(current_state)
        current_state = await self._end_node(current_state)
        
        return current_state