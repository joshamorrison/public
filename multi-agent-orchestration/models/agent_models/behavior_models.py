"""
Agent Behavior Models

Models for predicting agent behavior patterns, interaction dynamics, and collaboration effectiveness.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class AgentBehaviorModel:
    """
    Models agent behavior patterns and interaction preferences.
    """
    
    def __init__(self):
        """Initialize the behavior model."""
        self.interaction_history: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.agent_profiles: Dict[str, Dict[str, Any]] = {}
        self.collaboration_scores: Dict[Tuple[str, str], float] = {}
        self.communication_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def record_interaction(self, agent1_id: str, agent2_id: str, 
                          interaction_type: str, outcome_quality: float,
                          metadata: Dict[str, Any] = None):
        """
        Record an interaction between two agents.
        
        Args:
            agent1_id: First agent ID
            agent2_id: Second agent ID
            interaction_type: Type of interaction (handoff, collaboration, review, etc.)
            outcome_quality: Quality score of the interaction outcome (0-1)
            metadata: Additional interaction context
        """
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "outcome_quality": outcome_quality,
            "metadata": metadata or {}
        }
        
        # Store bidirectional interaction
        pair_key = tuple(sorted([agent1_id, agent2_id]))
        self.interaction_history[pair_key].append(interaction_record)
        
        # Update collaboration score
        self._update_collaboration_score(agent1_id, agent2_id, outcome_quality)
        
        # Update communication patterns
        self._update_communication_patterns(agent1_id, agent2_id, interaction_type)
    
    def _update_collaboration_score(self, agent1_id: str, agent2_id: str, quality: float):
        """Update running collaboration score between two agents."""
        pair_key = tuple(sorted([agent1_id, agent2_id]))
        
        if pair_key not in self.collaboration_scores:
            self.collaboration_scores[pair_key] = quality
        else:
            # Exponential moving average with alpha=0.3
            current_score = self.collaboration_scores[pair_key]
            self.collaboration_scores[pair_key] = 0.7 * current_score + 0.3 * quality
    
    def _update_communication_patterns(self, agent1_id: str, agent2_id: str, 
                                     interaction_type: str):
        """Update communication pattern statistics."""
        for agent_id in [agent1_id, agent2_id]:
            if agent_id not in self.communication_patterns:
                self.communication_patterns[agent_id] = {
                    "total_interactions": 0,
                    "interaction_types": defaultdict(int),
                    "preferred_partners": defaultdict(int)
                }
            
            patterns = self.communication_patterns[agent_id]
            patterns["total_interactions"] += 1
            patterns["interaction_types"][interaction_type] += 1
            
            # Track preferred partners
            other_agent = agent2_id if agent_id == agent1_id else agent1_id
            patterns["preferred_partners"][other_agent] += 1
    
    def predict_collaboration_quality(self, agent1_id: str, agent2_id: str) -> float:
        """
        Predict the quality of collaboration between two agents.
        
        Args:
            agent1_id: First agent ID
            agent2_id: Second agent ID
            
        Returns:
            Predicted collaboration quality score (0-1)
        """
        pair_key = tuple(sorted([agent1_id, agent2_id]))
        
        # If we have direct collaboration history, use it
        if pair_key in self.collaboration_scores:
            return self.collaboration_scores[pair_key]
        
        # Otherwise, predict based on agent profiles and similar collaborations
        return self._predict_new_collaboration(agent1_id, agent2_id)
    
    def _predict_new_collaboration(self, agent1_id: str, agent2_id: str) -> float:
        """Predict collaboration quality for agents with no direct history."""
        # Get agent profiles
        profile1 = self.agent_profiles.get(agent1_id, {})
        profile2 = self.agent_profiles.get(agent2_id, {})
        
        # Default prediction
        base_score = 0.6
        
        # Adjust based on agent types
        type1 = profile1.get("agent_type", "")
        type2 = profile2.get("agent_type", "")
        
        # Some agent type combinations work better together
        good_combinations = {
            ("researcher", "analyst"): 0.1,
            ("analyst", "synthesizer"): 0.15,
            ("synthesizer", "critic"): 0.1,
            ("supervisor", "researcher"): 0.05,
            ("supervisor", "analyst"): 0.05,
            ("supervisor", "synthesizer"): 0.05,
            ("supervisor", "critic"): 0.05
        }
        
        combo_key = tuple(sorted([type1, type2]))
        if combo_key in good_combinations:
            base_score += good_combinations[combo_key]
        
        # Adjust based on similar collaboration patterns
        similar_score_adj = self._calculate_similar_collaboration_adjustment(agent1_id, agent2_id)
        base_score += similar_score_adj
        
        return np.clip(base_score, 0.0, 1.0)
    
    def _calculate_similar_collaboration_adjustment(self, agent1_id: str, agent2_id: str) -> float:
        """Calculate adjustment based on similar collaboration patterns."""
        # Find agents similar to each target agent
        similar_to_agent1 = self._find_similar_agents(agent1_id)
        similar_to_agent2 = self._find_similar_agents(agent2_id)
        
        # Look at how similar agents collaborated
        collaboration_qualities = []
        
        for similar1 in similar_to_agent1:
            for similar2 in similar_to_agent2:
                pair_key = tuple(sorted([similar1, similar2]))
                if pair_key in self.collaboration_scores:
                    collaboration_qualities.append(self.collaboration_scores[pair_key])
        
        if collaboration_qualities:
            avg_quality = np.mean(collaboration_qualities)
            return (avg_quality - 0.6) * 0.5  # Scale the adjustment
        
        return 0.0
    
    def _find_similar_agents(self, target_agent_id: str, max_similar: int = 3) -> List[str]:
        """Find agents similar to the target agent based on behavior patterns."""
        target_profile = self.agent_profiles.get(target_agent_id, {})
        target_patterns = self.communication_patterns.get(target_agent_id, {})
        
        similarities = []
        
        for agent_id, profile in self.agent_profiles.items():
            if agent_id == target_agent_id:
                continue
            
            similarity = self._calculate_agent_similarity(
                target_profile, target_patterns,
                profile, self.communication_patterns.get(agent_id, {})
            )
            
            similarities.append((agent_id, similarity))
        
        # Sort by similarity and return top agents
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in similarities[:max_similar]]
    
    def _calculate_agent_similarity(self, profile1: Dict[str, Any], patterns1: Dict[str, Any],
                                  profile2: Dict[str, Any], patterns2: Dict[str, Any]) -> float:
        """Calculate similarity between two agents."""
        similarity = 0.0
        
        # Type similarity
        if profile1.get("agent_type") == profile2.get("agent_type"):
            similarity += 0.3
        
        # Capability similarity
        caps1 = set(profile1.get("capabilities", []))
        caps2 = set(profile2.get("capabilities", []))
        
        if caps1 or caps2:
            cap_similarity = len(caps1 & caps2) / len(caps1 | caps2)
            similarity += cap_similarity * 0.4
        
        # Communication pattern similarity
        types1 = patterns1.get("interaction_types", {})
        types2 = patterns2.get("interaction_types", {})
        
        if types1 or types2:
            all_types = set(types1.keys()) | set(types2.keys())
            pattern_similarity = 0.0
            
            for interaction_type in all_types:
                freq1 = types1.get(interaction_type, 0) / max(1, patterns1.get("total_interactions", 1))
                freq2 = types2.get(interaction_type, 0) / max(1, patterns2.get("total_interactions", 1))
                pattern_similarity += 1.0 - abs(freq1 - freq2)
            
            pattern_similarity /= len(all_types)
            similarity += pattern_similarity * 0.3
        
        return similarity
    
    def recommend_collaboration_partners(self, agent_id: str, 
                                       available_agents: List[str],
                                       max_recommendations: int = 3) -> List[Tuple[str, float]]:
        """
        Recommend the best collaboration partners for an agent.
        
        Args:
            agent_id: Agent seeking collaboration partners
            available_agents: List of available agents
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of (agent_id, predicted_quality) tuples
        """
        recommendations = []
        
        for candidate_id in available_agents:
            if candidate_id == agent_id:
                continue
            
            predicted_quality = self.predict_collaboration_quality(agent_id, candidate_id)
            recommendations.append((candidate_id, predicted_quality))
        
        # Sort by predicted quality and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:max_recommendations]
    
    def update_agent_profile(self, agent_id: str, profile_data: Dict[str, Any]):
        """Update agent profile information."""
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = {}
        
        self.agent_profiles[agent_id].update(profile_data)

class InteractionPredictor:
    """
    Predicts optimal interaction patterns and timing for multi-agent workflows.
    """
    
    def __init__(self):
        """Initialize the interaction predictor."""
        self.interaction_sequences: List[List[Dict[str, Any]]] = []
        self.timing_patterns: Dict[str, List[float]] = defaultdict(list)
        self.success_patterns: Dict[str, List[bool]] = defaultdict(list)
    
    def record_workflow_sequence(self, workflow_id: str, 
                                interactions: List[Dict[str, Any]],
                                overall_success: bool):
        """
        Record a complete workflow interaction sequence.
        
        Args:
            workflow_id: Workflow identifier
            interactions: List of interactions in chronological order
            overall_success: Whether the workflow was successful
        """
        # Add workflow metadata to each interaction
        enhanced_interactions = []
        for i, interaction in enumerate(interactions):
            enhanced_interaction = interaction.copy()
            enhanced_interaction.update({
                "workflow_id": workflow_id,
                "sequence_position": i,
                "workflow_success": overall_success,
                "sequence_length": len(interactions)
            })
            enhanced_interactions.append(enhanced_interaction)
        
        self.interaction_sequences.append(enhanced_interactions)
        
        # Update timing patterns
        self._update_timing_patterns(enhanced_interactions)
        
        # Update success patterns
        self._update_success_patterns(enhanced_interactions, overall_success)
    
    def _update_timing_patterns(self, interactions: List[Dict[str, Any]]):
        """Update timing pattern statistics."""
        if len(interactions) < 2:
            return
        
        for i in range(len(interactions) - 1):
            current = interactions[i]
            next_interaction = interactions[i + 1]
            
            # Calculate time between interactions
            current_time = datetime.fromisoformat(current["timestamp"])
            next_time = datetime.fromisoformat(next_interaction["timestamp"])
            time_diff = (next_time - current_time).total_seconds()
            
            # Create pattern key
            pattern_key = f"{current['interaction_type']}_to_{next_interaction['interaction_type']}"
            self.timing_patterns[pattern_key].append(time_diff)
    
    def _update_success_patterns(self, interactions: List[Dict[str, Any]], success: bool):
        """Update success pattern statistics."""
        # Record success for different sequence patterns
        sequence_types = [interaction["interaction_type"] for interaction in interactions]
        sequence_key = "_".join(sequence_types)
        
        self.success_patterns[sequence_key].append(success)
        
        # Record success for subsequences
        for length in range(2, min(4, len(sequence_types) + 1)):
            for start in range(len(sequence_types) - length + 1):
                subseq = sequence_types[start:start + length]
                subseq_key = "_".join(subseq)
                self.success_patterns[subseq_key].append(success)
    
    def predict_optimal_timing(self, from_interaction: str, to_interaction: str) -> float:
        """
        Predict optimal timing between two interaction types.
        
        Args:
            from_interaction: Source interaction type
            to_interaction: Target interaction type
            
        Returns:
            Predicted optimal delay in seconds
        """
        pattern_key = f"{from_interaction}_to_{to_interaction}"
        
        if pattern_key not in self.timing_patterns:
            # Default timing based on interaction types
            return self._default_interaction_timing(from_interaction, to_interaction)
        
        timings = self.timing_patterns[pattern_key]
        
        # Use median as it's robust to outliers
        optimal_timing = np.median(timings)
        
        # Ensure reasonable bounds
        return max(5.0, min(300.0, optimal_timing))  # 5 seconds to 5 minutes
    
    def _default_interaction_timing(self, from_interaction: str, to_interaction: str) -> float:
        """Provide default timing estimates for unknown patterns."""
        default_timings = {
            "research_to_analysis": 30.0,
            "analysis_to_synthesis": 20.0,
            "synthesis_to_review": 15.0,
            "review_to_revision": 10.0,
            "handoff_to_processing": 5.0,
            "processing_to_feedback": 25.0
        }
        
        pattern_key = f"{from_interaction}_to_{to_interaction}"
        return default_timings.get(pattern_key, 20.0)  # Default 20 seconds
    
    def predict_sequence_success(self, planned_sequence: List[str]) -> float:
        """
        Predict success probability for a planned interaction sequence.
        
        Args:
            planned_sequence: List of interaction types in planned order
            
        Returns:
            Predicted success probability (0-1)
        """
        if not planned_sequence:
            return 0.5
        
        # Check for exact sequence match
        sequence_key = "_".join(planned_sequence)
        if sequence_key in self.success_patterns:
            return np.mean(self.success_patterns[sequence_key])
        
        # Check for subsequence matches
        success_scores = []
        
        for length in range(2, min(4, len(planned_sequence) + 1)):
            for start in range(len(planned_sequence) - length + 1):
                subseq = planned_sequence[start:start + length]
                subseq_key = "_".join(subseq)
                
                if subseq_key in self.success_patterns:
                    subseq_score = np.mean(self.success_patterns[subseq_key])
                    # Weight longer subsequences more heavily
                    weight = length / 3.0
                    success_scores.append(subseq_score * weight)
        
        if success_scores:
            return np.mean(success_scores)
        
        # Fallback to default based on sequence length and types
        return self._default_sequence_success(planned_sequence)
    
    def _default_sequence_success(self, sequence: List[str]) -> float:
        """Provide default success estimate for unknown sequences."""
        # Base success rate decreases with sequence length
        base_rate = max(0.3, 0.8 - (len(sequence) - 1) * 0.1)
        
        # Bonus for well-known good patterns
        good_patterns = ["research", "analysis", "synthesis", "review"]
        if sequence == good_patterns[:len(sequence)]:
            base_rate += 0.1
        
        return min(1.0, base_rate)
    
    def optimize_interaction_sequence(self, available_interactions: List[str],
                                    max_length: int = 5) -> List[str]:
        """
        Find the optimal sequence of interactions for best success probability.
        
        Args:
            available_interactions: List of available interaction types
            max_length: Maximum sequence length to consider
            
        Returns:
            Optimal interaction sequence
        """
        from itertools import permutations
        
        best_sequence = []
        best_score = 0.0
        
        # Try different sequence lengths
        for length in range(1, min(max_length + 1, len(available_interactions) + 1)):
            # Try all permutations of this length
            for sequence in permutations(available_interactions, length):
                success_prob = self.predict_sequence_success(list(sequence))
                
                # Prefer shorter sequences with similar success rates
                adjusted_score = success_prob - (length - 1) * 0.05
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_sequence = list(sequence)
        
        return best_sequence
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive interaction statistics."""
        stats = {
            "total_workflows": len(self.interaction_sequences),
            "unique_timing_patterns": len(self.timing_patterns),
            "unique_success_patterns": len(self.success_patterns),
            "common_patterns": {},
            "timing_summary": {}
        }
        
        # Find most common success patterns
        pattern_frequencies = {
            pattern: len(successes) 
            for pattern, successes in self.success_patterns.items()
        }
        
        sorted_patterns = sorted(pattern_frequencies.items(), 
                               key=lambda x: x[1], reverse=True)
        
        stats["common_patterns"] = dict(sorted_patterns[:10])
        
        # Timing summary
        for pattern, timings in self.timing_patterns.items():
            if len(timings) >= 3:  # Only include patterns with sufficient data
                stats["timing_summary"][pattern] = {
                    "median": np.median(timings),
                    "mean": np.mean(timings),
                    "std": np.std(timings),
                    "samples": len(timings)
                }
        
        return stats