"""
Reflective Pattern

Self-improving agents with feedback loops and meta-cognition.
Agents iteratively refine their outputs through self-evaluation and peer review.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..agents.base_agent import BaseAgent, AgentResult
from ..agents.critic_agent import CriticAgent


class ReflectivePattern:
    """
    Reflective orchestration pattern for self-improving agent workflows.
    
    The reflective pattern:
    - Implements iterative improvement cycles
    - Uses critic agents for quality assessment and feedback
    - Supports self-evaluation and peer review
    - Enables meta-reasoning about reasoning processes
    - Tracks improvement over iterations
    - Implements early stopping based on convergence criteria
    """

    def __init__(self, pattern_id: str = "reflective-001"):
        """
        Initialize the reflective pattern.
        
        Args:
            pattern_id: Unique identifier for this pattern instance
        """
        self.pattern_id = pattern_id
        self.name = "Reflective Pattern"
        self.description = "Self-improving feedback loops with meta-cognition"
        self.primary_agent: Optional[BaseAgent] = None
        self.critic_agents: List[CriticAgent] = []
        self.execution_history: List[Dict[str, Any]] = []
        
        # Reflection configuration
        self.max_iterations = 5
        self.convergence_threshold = 0.95  # Confidence threshold for early stopping
        self.improvement_threshold = 0.05  # Minimum improvement required to continue
        self.enable_meta_reasoning = True
        self.enable_peer_review = True

    def set_primary_agent(self, agent: BaseAgent):
        """
        Set the primary agent that will be improved through reflection.
        
        Args:
            agent: Primary agent to be improved
        """
        self.primary_agent = agent
        print(f"[REFLECTIVE] Set primary agent: {agent.name}")

    def add_critic_agent(self, critic: CriticAgent, role: str = "general_critic"):
        """
        Add a critic agent for quality assessment and feedback.
        
        Args:
            critic: Critic agent to add
            role: Role of the critic (general_critic, domain_expert, etc.)
        """
        critic_config = {
            "critic": critic,
            "role": role,
            "added_at": datetime.now()
        }
        self.critic_agents.append(critic)
        print(f"[REFLECTIVE] Added critic: {critic.name} (role: {role})")

    def configure_reflection(self, max_iterations: int = 5, 
                           convergence_threshold: float = 0.95,
                           improvement_threshold: float = 0.05,
                           enable_meta_reasoning: bool = True,
                           enable_peer_review: bool = True):
        """
        Configure reflection parameters.
        
        Args:
            max_iterations: Maximum number of reflection iterations
            convergence_threshold: Confidence threshold for early stopping
            improvement_threshold: Minimum improvement to continue iterations
            enable_meta_reasoning: Enable meta-cognitive reasoning
            enable_peer_review: Enable peer review between critics
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.improvement_threshold = improvement_threshold
        self.enable_meta_reasoning = enable_meta_reasoning
        self.enable_peer_review = enable_peer_review
        
        print(f"[REFLECTIVE] Configuration updated: max_iter={max_iterations}, convergence={convergence_threshold}")

    async def execute(self, initial_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute reflective improvement process.
        
        Args:
            initial_task: Initial task for the primary agent
            
        Returns:
            Reflective execution results with improvement history
        """
        if not self.primary_agent:
            raise ValueError("Primary agent must be set before execution")
        
        execution_id = f"reflect_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        print(f"[REFLECTIVE] Starting reflective execution {execution_id}")
        print(f"[REFLECTIVE] Primary agent: {self.primary_agent.name}")
        print(f"[REFLECTIVE] Critics: {len(self.critic_agents)}")
        print(f"[REFLECTIVE] Max iterations: {self.max_iterations}")
        
        execution_result = {
            "execution_id": execution_id,
            "pattern_type": "reflective",
            "start_time": start_time,
            "primary_agent_id": self.primary_agent.agent_id,
            "critics_count": len(self.critic_agents),
            "iterations": [],
            "final_result": None,
            "improvement_achieved": False,
            "convergence_reached": False,
            "early_stopping_reason": None,
            "success": True,
            "error_message": None
        }
        
        try:
            current_task = initial_task.copy()
            current_task["reflection_execution_id"] = execution_id
            
            # Track improvement across iterations
            previous_confidence = 0.0
            best_result = None
            
            # Iterative improvement loop
            for iteration in range(self.max_iterations):
                print(f"[REFLECTIVE] Starting iteration {iteration + 1}/{self.max_iterations}")
                
                iteration_result = await self._execute_iteration(
                    iteration, current_task, previous_confidence, execution_result
                )
                
                execution_result["iterations"].append(iteration_result)
                
                # Check for iteration failure
                if not iteration_result["success"]:
                    execution_result["success"] = False
                    execution_result["error_message"] = iteration_result.get("error_message")
                    break
                
                current_confidence = iteration_result["primary_result"].confidence
                
                # Track best result
                if not best_result or current_confidence > best_result.confidence:
                    best_result = iteration_result["primary_result"]
                
                # Check convergence criteria
                if current_confidence >= self.convergence_threshold:
                    execution_result["convergence_reached"] = True
                    execution_result["early_stopping_reason"] = "convergence_threshold_reached"
                    print(f"[REFLECTIVE] Convergence reached at iteration {iteration + 1}")
                    break
                
                # Check improvement threshold
                improvement = current_confidence - previous_confidence
                if iteration > 0 and improvement < self.improvement_threshold:
                    execution_result["early_stopping_reason"] = "insufficient_improvement"
                    print(f"[REFLECTIVE] Insufficient improvement at iteration {iteration + 1}: {improvement:.3f}")
                    break
                
                # Update for next iteration
                previous_confidence = current_confidence
                current_task = await self._prepare_next_iteration(iteration_result, current_task)
            
            # Calculate overall improvement
            if execution_result["iterations"]:
                initial_confidence = execution_result["iterations"][0]["primary_result"].confidence
                final_confidence = execution_result["iterations"][-1]["primary_result"].confidence
                execution_result["improvement_achieved"] = final_confidence > initial_confidence
                execution_result["total_improvement"] = final_confidence - initial_confidence
            
            execution_result["final_result"] = best_result
            execution_result["end_time"] = datetime.now()
            execution_result["total_execution_time"] = (execution_result["end_time"] - start_time).total_seconds()
            
            # Store execution history
            self.execution_history.append(execution_result)
            
            status = "SUCCESS" if execution_result["success"] else "FAILED"
            if execution_result["convergence_reached"]:
                status += " (CONVERGED)"
            
            print(f"[REFLECTIVE] Execution {execution_id} completed: {status}")
            print(f"[REFLECTIVE] Total iterations: {len(execution_result['iterations'])}")
            if execution_result.get("total_improvement"):
                print(f"[REFLECTIVE] Total improvement: {execution_result['total_improvement']:.3f}")
            
            return execution_result
            
        except Exception as e:
            execution_result["success"] = False
            execution_result["error_message"] = str(e)
            execution_result["end_time"] = datetime.now()
            
            self.execution_history.append(execution_result)
            print(f"[REFLECTIVE] Execution {execution_id} failed with error: {str(e)}")
            
            return execution_result

    async def _execute_iteration(self, iteration: int, current_task: Dict[str, Any], 
                               previous_confidence: float, 
                               execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single reflection iteration.
        
        Args:
            iteration: Current iteration number (0-based)
            current_task: Task for this iteration
            previous_confidence: Confidence from previous iteration
            execution_context: Overall execution context
            
        Returns:
            Iteration execution result
        """
        iteration_start = datetime.now()
        
        iteration_result = {
            "iteration": iteration,
            "start_time": iteration_start,
            "primary_result": None,
            "critic_feedback": [],
            "meta_reasoning": None,
            "peer_review": None,
            "improvements_identified": [],
            "confidence_change": 0.0,
            "success": True,
            "error_message": None
        }
        
        try:
            # Phase 1: Primary agent execution
            print(f"[REFLECTIVE] Iteration {iteration + 1}: Primary agent execution")
            
            iteration_task = current_task.copy()
            iteration_task.update({
                "iteration": iteration,
                "previous_confidence": previous_confidence,
                "reflection_context": {
                    "is_reflection": True,
                    "iteration_number": iteration,
                    "max_iterations": self.max_iterations
                }
            })
            
            primary_result = await self.primary_agent.process_task(iteration_task)
            iteration_result["primary_result"] = primary_result
            
            if not primary_result.success:
                iteration_result["success"] = False
                iteration_result["error_message"] = primary_result.error_message
                return iteration_result
            
            # Phase 2: Critic feedback
            if self.critic_agents:
                print(f"[REFLECTIVE] Iteration {iteration + 1}: Gathering critic feedback")
                critic_feedback = await self._gather_critic_feedback(primary_result, iteration_task)
                iteration_result["critic_feedback"] = critic_feedback
                
                # Extract improvement suggestions
                improvements = []
                for feedback in critic_feedback:
                    if feedback.get("success") and feedback.get("recommendations"):
                        improvements.extend(feedback["recommendations"])
                iteration_result["improvements_identified"] = improvements
            
            # Phase 3: Meta-reasoning (if enabled)
            if self.enable_meta_reasoning:
                print(f"[REFLECTIVE] Iteration {iteration + 1}: Meta-reasoning")
                meta_reasoning = await self._perform_meta_reasoning(
                    primary_result, iteration_result["critic_feedback"], iteration, execution_context
                )
                iteration_result["meta_reasoning"] = meta_reasoning
            
            # Phase 4: Peer review (if enabled and multiple critics)
            if self.enable_peer_review and len(self.critic_agents) > 1:
                print(f"[REFLECTIVE] Iteration {iteration + 1}: Peer review")
                peer_review = await self._conduct_peer_review(iteration_result["critic_feedback"])
                iteration_result["peer_review"] = peer_review
            
            # Calculate confidence change
            iteration_result["confidence_change"] = primary_result.confidence - previous_confidence
            
            iteration_result["end_time"] = datetime.now()
            iteration_result["iteration_time"] = (iteration_result["end_time"] - iteration_start).total_seconds()
            
            print(f"[REFLECTIVE] Iteration {iteration + 1} completed: confidence {primary_result.confidence:.3f}")
            
            return iteration_result
            
        except Exception as e:
            iteration_result["success"] = False
            iteration_result["error_message"] = str(e)
            iteration_result["end_time"] = datetime.now()
            
            print(f"[REFLECTIVE] Iteration {iteration + 1} failed: {str(e)}")
            return iteration_result

    async def _gather_critic_feedback(self, primary_result: AgentResult, 
                                    task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gather feedback from all critic agents.
        
        Args:
            primary_result: Result from primary agent
            task: Current task context
            
        Returns:
            List of critic feedback
        """
        critic_feedback = []
        
        for critic in self.critic_agents:
            try:
                # Prepare critic task
                critic_task = {
                    "type": "quality_review",
                    "content": primary_result.content,
                    "context": {
                        "primary_agent": self.primary_agent.name,
                        "confidence": primary_result.confidence,
                        "metadata": primary_result.metadata,
                        "original_task": task.get("description", "")
                    },
                    "task_id": f"critic_{critic.agent_id}_{datetime.now().strftime('%H%M%S')}"
                }
                
                critic_result = await critic.process_task(critic_task)
                
                feedback_entry = {
                    "critic_id": critic.agent_id,
                    "critic_name": critic.name,
                    "success": critic_result.success,
                    "feedback_content": critic_result.content,
                    "confidence": critic_result.confidence,
                    "recommendations": [],
                    "quality_score": critic_result.metadata.get("overall_quality_score", 0)
                }
                
                # Extract recommendations from critic metadata
                if critic_result.metadata and "recommendations_count" in critic_result.metadata:
                    # In a real implementation, would parse structured feedback
                    feedback_entry["recommendations"] = [
                        "Improve clarity and structure",
                        "Add more supporting evidence", 
                        "Enhance actionable recommendations"
                    ]
                
                critic_feedback.append(feedback_entry)
                
            except Exception as e:
                critic_feedback.append({
                    "critic_id": critic.agent_id,
                    "critic_name": critic.name,
                    "success": False,
                    "error": str(e)
                })
        
        return critic_feedback

    async def _perform_meta_reasoning(self, primary_result: AgentResult, 
                                    critic_feedback: List[Dict[str, Any]],
                                    iteration: int, 
                                    execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform meta-reasoning about the reasoning process.
        
        Args:
            primary_result: Result from primary agent
            critic_feedback: Feedback from critics
            iteration: Current iteration
            execution_context: Execution context
            
        Returns:
            Meta-reasoning analysis
        """
        meta_reasoning = {
            "iteration": iteration,
            "reasoning_quality_assessment": "",
            "process_effectiveness": 0.0,
            "improvement_trajectory": "",
            "strategic_recommendations": [],
            "confidence": 0.8
        }
        
        # Analyze reasoning quality
        successful_critics = [f for f in critic_feedback if f.get("success", False)]
        if successful_critics:
            avg_critic_confidence = sum(f.get("confidence", 0) for f in successful_critics) / len(successful_critics)
            quality_scores = [f.get("quality_score", 0) for f in successful_critics if f.get("quality_score")]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            meta_reasoning["reasoning_quality_assessment"] = f"Primary reasoning shows {primary_result.confidence:.1%} confidence with {avg_quality:.1%} average quality from critics"
            meta_reasoning["process_effectiveness"] = (primary_result.confidence + avg_critic_confidence) / 2
        
        # Analyze improvement trajectory
        if iteration > 0:
            previous_iterations = execution_context.get("iterations", [])
            if previous_iterations:
                confidence_trend = [iter_result["primary_result"].confidence for iter_result in previous_iterations]
                confidence_trend.append(primary_result.confidence)
                
                if len(confidence_trend) >= 2:
                    recent_change = confidence_trend[-1] - confidence_trend[-2]
                    if recent_change > 0.05:
                        meta_reasoning["improvement_trajectory"] = "Strong positive trajectory"
                    elif recent_change > 0:
                        meta_reasoning["improvement_trajectory"] = "Moderate improvement"
                    else:
                        meta_reasoning["improvement_trajectory"] = "Plateau or decline"
        
        # Strategic recommendations
        meta_reasoning["strategic_recommendations"] = [
            "Continue iterative improvement process",
            "Focus on highest-impact critic recommendations",
            "Consider alternative reasoning approaches if plateau reached"
        ]
        
        return meta_reasoning

    async def _conduct_peer_review(self, critic_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Conduct peer review between critics.
        
        Args:
            critic_feedback: Feedback from all critics
            
        Returns:
            Peer review analysis
        """
        successful_feedback = [f for f in critic_feedback if f.get("success", False)]
        
        if len(successful_feedback) < 2:
            return {"insufficient_critics": True}
        
        # Analyze consensus among critics
        quality_scores = [f.get("quality_score", 0) for f in successful_feedback]
        confidences = [f.get("confidence", 0) for f in successful_feedback]
        
        quality_consensus = max(quality_scores) - min(quality_scores) < 0.2 if quality_scores else False
        confidence_consensus = max(confidences) - min(confidences) < 0.2 if confidences else False
        
        peer_review = {
            "critics_participating": len(successful_feedback),
            "quality_consensus": quality_consensus,
            "confidence_consensus": confidence_consensus,
            "consensus_strength": "strong" if quality_consensus and confidence_consensus else "weak",
            "conflicting_opinions": not (quality_consensus and confidence_consensus),
            "recommendation": "High consensus supports current direction" if quality_consensus else "Mixed opinions suggest further iteration needed"
        }
        
        return peer_review

    async def _prepare_next_iteration(self, iteration_result: Dict[str, Any], 
                                    current_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare task for next iteration based on feedback.
        
        Args:
            iteration_result: Result from current iteration
            current_task: Current task specification
            
        Returns:
            Task prepared for next iteration
        """
        next_task = current_task.copy()
        
        # Incorporate critic feedback
        improvements = iteration_result.get("improvements_identified", [])
        if improvements:
            improvement_guidance = "Based on critic feedback, focus on: " + "; ".join(improvements[:3])
            next_task["improvement_guidance"] = improvement_guidance
            
            # Update description with improvement guidance
            original_desc = next_task.get("description", "")
            next_task["description"] = f"{original_desc}\n\nImprovement Focus: {improvement_guidance}"
        
        # Add meta-reasoning insights
        meta_reasoning = iteration_result.get("meta_reasoning")
        if meta_reasoning:
            next_task["meta_insights"] = meta_reasoning.get("strategic_recommendations", [])
        
        # Include previous iteration context
        next_task["previous_iteration"] = {
            "confidence": iteration_result["primary_result"].confidence,
            "improvements_identified": len(improvements),
            "critics_feedback_count": len(iteration_result.get("critic_feedback", []))
        }
        
        return next_task

    def get_pattern_configuration(self) -> Dict[str, Any]:
        """Get current reflective pattern configuration."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.name,
            "primary_agent": {
                "agent_id": self.primary_agent.agent_id if self.primary_agent else None,
                "agent_name": self.primary_agent.name if self.primary_agent else None
            },
            "critics_count": len(self.critic_agents),
            "critic_agents": [
                {
                    "critic_id": critic.agent_id,
                    "critic_name": critic.name
                }
                for critic in self.critic_agents
            ],
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "improvement_threshold": self.improvement_threshold,
            "enable_meta_reasoning": self.enable_meta_reasoning,
            "enable_peer_review": self.enable_peer_review,
            "executions_count": len(self.execution_history)
        }

    def get_reflection_analytics(self) -> Dict[str, Any]:
        """Get analytics about reflection effectiveness."""
        if not self.execution_history:
            return {"no_executions": True}
        
        # Analyze improvement patterns
        total_improvements = 0
        convergence_count = 0
        avg_iterations = 0
        
        for execution in self.execution_history:
            if execution.get("improvement_achieved"):
                total_improvements += 1
            if execution.get("convergence_reached"):
                convergence_count += 1
            avg_iterations += len(execution.get("iterations", []))
        
        avg_iterations = avg_iterations / len(self.execution_history) if self.execution_history else 0
        
        return {
            "total_executions": len(self.execution_history),
            "improvement_rate": total_improvements / len(self.execution_history),
            "convergence_rate": convergence_count / len(self.execution_history),
            "average_iterations": avg_iterations,
            "reflection_effectiveness": {
                "improvement_rate": total_improvements / len(self.execution_history),
                "early_convergence": convergence_count > 0,
                "iteration_efficiency": avg_iterations / self.max_iterations
            },
            "configuration": {
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold,
                "improvement_threshold": self.improvement_threshold
            }
        }

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get reflective pattern execution metrics."""
        if not self.execution_history:
            return {"no_executions": True}
        
        successful_executions = [ex for ex in self.execution_history if ex["success"]]
        improved_executions = [ex for ex in successful_executions if ex.get("improvement_achieved", False)]
        converged_executions = [ex for ex in successful_executions if ex.get("convergence_reached", False)]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "improvement_rate": len(improved_executions) / len(self.execution_history),
            "convergence_rate": len(converged_executions) / len(self.execution_history),
            "success_rate": len(successful_executions) / len(self.execution_history),
            "average_iterations": sum(len(ex.get("iterations", [])) for ex in self.execution_history) / len(self.execution_history),
            "critics_utilization": len(self.critic_agents),
            "reflection_features": {
                "meta_reasoning_enabled": self.enable_meta_reasoning,
                "peer_review_enabled": self.enable_peer_review,
                "max_iterations": self.max_iterations
            }
        }

    def clear_history(self):
        """Clear execution history."""
        self.execution_history = []
        print(f"[REFLECTIVE] Execution history cleared")