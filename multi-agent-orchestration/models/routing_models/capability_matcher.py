"""
Capability Matcher

Advanced capability matching and task-agent scoring system.
Uses semantic similarity and capability vectors for optimal task assignment.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class CapabilityMatcher:
    """
    Matches tasks to agents based on capability similarity and requirements.
    """
    
    def __init__(self):
        """Initialize the capability matcher."""
        self.capability_vectors: Dict[str, np.ndarray] = {}
        self.task_requirements: Dict[str, List[str]] = {}
        self.capability_hierarchy = self._build_capability_hierarchy()
        self.semantic_similarity_cache: Dict[Tuple[str, str], float] = {}
    
    def _build_capability_hierarchy(self) -> Dict[str, Dict[str, float]]:
        """
        Build hierarchical capability relationships.
        Defines which capabilities are related and their similarity weights.
        """
        return {
            # Research capabilities
            "web_search": {
                "information_gathering": 0.9,
                "data_collection": 0.8,
                "source_verification": 0.7,
                "research": 0.9
            },
            "data_analysis": {
                "statistical_analysis": 0.9,
                "pattern_recognition": 0.8,
                "insight_generation": 0.7,
                "analysis": 0.9
            },
            "research": {
                "web_search": 0.9,
                "information_gathering": 0.95,
                "literature_review": 0.8,
                "fact_checking": 0.7
            },
            
            # Analysis capabilities  
            "statistical_analysis": {
                "data_analysis": 0.9,
                "quantitative_analysis": 0.95,
                "trend_analysis": 0.8,
                "forecasting": 0.7
            },
            "pattern_recognition": {
                "data_analysis": 0.8,
                "trend_identification": 0.9,
                "anomaly_detection": 0.8,
                "insight_generation": 0.7
            },
            "analysis": {
                "data_analysis": 0.9,
                "critical_thinking": 0.8,
                "problem_solving": 0.8,
                "evaluation": 0.7
            },
            
            # Synthesis capabilities
            "content_generation": {
                "writing": 0.9,
                "synthesis": 0.9,
                "summarization": 0.8,
                "report_generation": 0.8
            },
            "summarization": {
                "content_generation": 0.8,
                "synthesis": 0.9,
                "information_distillation": 0.9,
                "abstract_writing": 0.8
            },
            "synthesis": {
                "content_generation": 0.9,
                "integration": 0.9,
                "compilation": 0.8,
                "consolidation": 0.8
            },
            
            # Review capabilities
            "quality_assessment": {
                "evaluation": 0.9,
                "review": 0.9,
                "criticism": 0.8,
                "validation": 0.8
            },
            "feedback_generation": {
                "criticism": 0.8,
                "review": 0.8,
                "improvement_suggestions": 0.9,
                "evaluation": 0.7
            },
            "review": {
                "quality_assessment": 0.9,
                "criticism": 0.9,
                "evaluation": 0.8,
                "proofreading": 0.7
            },
            
            # Coordination capabilities
            "workflow_management": {
                "coordination": 0.9,
                "project_management": 0.9,
                "task_delegation": 0.8,
                "supervision": 0.8
            },
            "delegation": {
                "task_assignment": 0.9,
                "coordination": 0.8,
                "management": 0.8,
                "supervision": 0.7
            },
            "coordination": {
                "workflow_management": 0.9,
                "team_management": 0.8,
                "synchronization": 0.8,
                "orchestration": 0.9
            }
        }
    
    def register_agent_capabilities(self, agent_id: str, capabilities: List[str]):
        """
        Register agent capabilities and create capability vector.
        
        Args:
            agent_id: Unique agent identifier
            capabilities: List of agent capabilities
        """
        # Create capability vector
        vector = self._create_capability_vector(capabilities)
        self.capability_vectors[agent_id] = vector
        
        logger.info(f"Registered capabilities for agent {agent_id}: {capabilities}")
    
    def _create_capability_vector(self, capabilities: List[str]) -> np.ndarray:
        """
        Create a numerical vector representation of capabilities.
        """
        # Define all possible capabilities (extended from hierarchy)
        all_capabilities = set()
        for cap, related in self.capability_hierarchy.items():
            all_capabilities.add(cap)
            all_capabilities.update(related.keys())
        
        all_capabilities = sorted(list(all_capabilities))
        vector = np.zeros(len(all_capabilities))
        
        # Set direct capabilities
        for cap in capabilities:
            if cap in all_capabilities:
                idx = all_capabilities.index(cap)
                vector[idx] = 1.0
        
        # Add weighted related capabilities
        for cap in capabilities:
            if cap in self.capability_hierarchy:
                for related_cap, weight in self.capability_hierarchy[cap].items():
                    if related_cap in all_capabilities:
                        idx = all_capabilities.index(related_cap)
                        vector[idx] = max(vector[idx], weight)
        
        return vector
    
    def calculate_capability_match(self, task_requirements: List[str], 
                                 agent_id: str) -> float:
        """
        Calculate how well an agent's capabilities match task requirements.
        
        Args:
            task_requirements: List of required capabilities
            agent_id: Agent to evaluate
            
        Returns:
            Match score between 0 and 1
        """
        if agent_id not in self.capability_vectors:
            return 0.0
        
        # Create vector for task requirements
        task_vector = self._create_capability_vector(task_requirements)
        agent_vector = self.capability_vectors[agent_id]
        
        # Calculate cosine similarity
        dot_product = np.dot(task_vector, agent_vector)
        task_norm = np.linalg.norm(task_vector)
        agent_norm = np.linalg.norm(agent_vector)
        
        if task_norm == 0 or agent_norm == 0:
            return 0.0
        
        similarity = dot_product / (task_norm * agent_norm)
        
        # Bonus for exact capability matches
        exact_matches = sum(1 for req in task_requirements 
                          if req in self._get_agent_direct_capabilities(agent_id))
        exact_bonus = min(0.2, exact_matches * 0.1)
        
        return min(1.0, similarity + exact_bonus)
    
    def _get_agent_direct_capabilities(self, agent_id: str) -> List[str]:
        """Get agent's direct (non-derived) capabilities."""
        # This would typically be stored separately
        # For now, infer from vector
        if agent_id not in self.capability_vectors:
            return []
        
        vector = self.capability_vectors[agent_id]
        all_capabilities = sorted(list(set().union(*[
            {cap} | set(related.keys()) 
            for cap, related in self.capability_hierarchy.items()
        ])))
        
        # Return capabilities with high scores (likely direct)
        direct_caps = []
        for i, score in enumerate(vector):
            if score >= 0.9 and i < len(all_capabilities):
                direct_caps.append(all_capabilities[i])
        
        return direct_caps

class TaskAgentScorer:
    """
    Comprehensive scoring system for task-agent matching.
    """
    
    def __init__(self):
        """Initialize the scorer."""
        self.capability_matcher = CapabilityMatcher()
        self.scoring_weights = {
            "capability_match": 0.4,
            "experience": 0.2,
            "availability": 0.2,
            "performance_history": 0.15,
            "specialization": 0.05
        }
    
    def score_agent_for_task(self, task: Dict[str, Any], agent_id: str,
                           agent_metadata: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Calculate comprehensive score for agent-task pairing.
        
        Args:
            task: Task specification
            agent_id: Agent to score
            agent_metadata: Additional agent information
            
        Returns:
            Dictionary with detailed scoring breakdown
        """
        if agent_metadata is None:
            agent_metadata = {}
        
        scores = {}
        
        # 1. Capability match score
        task_requirements = task.get("required_capabilities", [])
        if task_requirements:
            scores["capability_match"] = self.capability_matcher.calculate_capability_match(
                task_requirements, agent_id
            )
        else:
            # Infer requirements from task type
            inferred_requirements = self._infer_task_requirements(task.get("type", "general"))
            scores["capability_match"] = self.capability_matcher.calculate_capability_match(
                inferred_requirements, agent_id
            )
        
        # 2. Experience score
        scores["experience"] = self._calculate_experience_score(
            agent_id, task.get("type", "general"), agent_metadata
        )
        
        # 3. Availability score
        scores["availability"] = self._calculate_availability_score(
            agent_id, agent_metadata
        )
        
        # 4. Performance history score
        scores["performance_history"] = self._calculate_performance_score(
            agent_id, task.get("type", "general"), agent_metadata
        )
        
        # 5. Specialization score
        scores["specialization"] = self._calculate_specialization_score(
            agent_id, task, agent_metadata
        )
        
        # Calculate weighted total
        total_score = sum(
            scores[component] * self.scoring_weights[component]
            for component in scores
        )
        
        scores["total"] = total_score
        
        return scores
    
    def _infer_task_requirements(self, task_type: str) -> List[str]:
        """Infer required capabilities from task type."""
        requirement_map = {
            "research": ["web_search", "information_gathering", "data_analysis"],
            "analysis": ["data_analysis", "statistical_analysis", "pattern_recognition"],
            "synthesis": ["content_generation", "summarization", "integration"],
            "review": ["quality_assessment", "feedback_generation", "evaluation"],
            "coordination": ["workflow_management", "delegation", "supervision"]
        }
        
        return requirement_map.get(task_type, ["general_intelligence"])
    
    def _calculate_experience_score(self, agent_id: str, task_type: str, 
                                  metadata: Dict[str, Any]) -> float:
        """Calculate experience score based on task history."""
        # Get task history from metadata
        task_history = metadata.get("task_history", {})
        type_count = task_history.get(task_type, 0)
        total_tasks = sum(task_history.values())
        
        if total_tasks == 0:
            return 0.5  # Neutral for new agents
        
        # Experience factors
        type_experience = min(1.0, type_count / 10)  # Max at 10 tasks of this type
        general_experience = min(1.0, total_tasks / 50)  # Max at 50 total tasks
        
        return (type_experience * 0.7) + (general_experience * 0.3)
    
    def _calculate_availability_score(self, agent_id: str, 
                                    metadata: Dict[str, Any]) -> float:
        """Calculate availability score based on current workload."""
        current_load = metadata.get("current_workload", 0)
        max_capacity = metadata.get("max_capacity", 5)
        
        if max_capacity == 0:
            return 0.0
        
        utilization = current_load / max_capacity
        
        # Score decreases as utilization increases
        if utilization <= 0.5:
            return 1.0
        elif utilization <= 0.8:
            return 0.8
        elif utilization < 1.0:
            return 0.5
        else:
            return 0.0  # Overloaded
    
    def _calculate_performance_score(self, agent_id: str, task_type: str,
                                   metadata: Dict[str, Any]) -> float:
        """Calculate performance score based on historical results."""
        performance_data = metadata.get("performance_metrics", {})
        
        # Overall metrics
        success_rate = performance_data.get("success_rate", 0.5)
        avg_confidence = performance_data.get("average_confidence", 0.5)
        
        # Task-specific metrics
        task_metrics = performance_data.get("by_task_type", {}).get(task_type, {})
        task_success_rate = task_metrics.get("success_rate", success_rate)
        task_confidence = task_metrics.get("average_confidence", avg_confidence)
        
        # Combine metrics
        performance_score = (
            task_success_rate * 0.5 +
            task_confidence * 0.3 +
            success_rate * 0.2
        )
        
        return performance_score
    
    def _calculate_specialization_score(self, agent_id: str, task: Dict[str, Any],
                                      metadata: Dict[str, Any]) -> float:
        """Calculate specialization bonus for agent type match."""
        agent_type = metadata.get("agent_type", "").lower()
        task_type = task.get("type", "").lower()
        
        # Specialization matching
        specialization_matches = {
            "researcher": ["research", "investigation", "information_gathering"],
            "analyst": ["analysis", "evaluation", "assessment"],
            "synthesizer": ["synthesis", "compilation", "integration"],
            "critic": ["review", "criticism", "quality_assessment"],
            "supervisor": ["coordination", "management", "supervision"]
        }
        
        for specialist, matching_tasks in specialization_matches.items():
            if specialist in agent_type and task_type in matching_tasks:
                return 1.0
        
        return 0.5  # Default neutral score
    
    def rank_agents_for_task(self, task: Dict[str, Any], 
                           agent_candidates: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, float]]]:
        """
        Rank multiple agents for a task.
        
        Args:
            task: Task specification
            agent_candidates: Dictionary mapping agent_id to agent metadata
            
        Returns:
            List of (agent_id, scores) tuples sorted by total score
        """
        rankings = []
        
        for agent_id, metadata in agent_candidates.items():
            scores = self.score_agent_for_task(task, agent_id, metadata)
            rankings.append((agent_id, scores))
        
        # Sort by total score (descending)
        rankings.sort(key=lambda x: x[1]["total"], reverse=True)
        
        return rankings
    
    def explain_scoring(self, agent_id: str, task: Dict[str, Any], 
                       scores: Dict[str, float]) -> str:
        """
        Generate human-readable explanation of scoring decision.
        
        Args:
            agent_id: Agent that was scored
            task: Task that was scored for
            scores: Scoring results
            
        Returns:
            Explanation string
        """
        explanation_parts = [
            f"Scoring breakdown for agent {agent_id} on {task.get('type', 'general')} task:"
        ]
        
        for component, score in scores.items():
            if component == "total":
                continue
                
            weight = self.scoring_weights.get(component, 0)
            weighted_score = score * weight
            
            explanation_parts.append(
                f"  {component.replace('_', ' ').title()}: {score:.2f} "
                f"(weight: {weight:.1%}, contribution: {weighted_score:.3f})"
            )
        
        explanation_parts.append(f"  Total Score: {scores['total']:.3f}")
        
        return "\n".join(explanation_parts)