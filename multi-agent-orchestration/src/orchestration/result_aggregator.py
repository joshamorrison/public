"""
Result Aggregator

System for aggregating and synthesizing results from multiple agents and patterns.
Provides intelligent result fusion, conflict resolution, and quality assessment.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from ..agents.base_agent import AgentResult


@dataclass
class AggregationResult:
    """Structured result from aggregation process."""
    aggregated_content: str
    confidence: float
    source_count: int
    aggregation_method: str
    quality_score: float
    metadata: Dict[str, Any]
    timestamp: datetime
    success: bool = True
    error_message: Optional[str] = None


class ResultAggregator:
    """
    Intelligent result aggregation system for multi-agent outputs.
    
    The result aggregator:
    - Combines results from multiple agents or patterns
    - Resolves conflicts and inconsistencies
    - Provides quality scoring and confidence assessment
    - Supports different aggregation strategies
    - Maintains provenance and traceability
    """

    def __init__(self):
        """Initialize the result aggregator."""
        self.aggregation_history: List[Dict[str, Any]] = []
        self.supported_strategies = [
            "weighted_average",
            "highest_confidence",
            "consensus_building",
            "comprehensive_synthesis",
            "majority_vote"
        ]

    async def aggregate_agent_results(self, results: List[Union[AgentResult, Dict[str, Any]]], 
                                    strategy: str = "comprehensive_synthesis",
                                    weights: Optional[Dict[str, float]] = None) -> AggregationResult:
        """
        Aggregate results from multiple agents.
        
        Args:
            results: List of agent results to aggregate
            strategy: Aggregation strategy to use
            weights: Optional weights for each result (by index or agent_id)
            
        Returns:
            Aggregated result
        """
        if not results:
            return AggregationResult(
                aggregated_content="No results to aggregate",
                confidence=0.0,
                source_count=0,
                aggregation_method=strategy,
                quality_score=0.0,
                metadata={"error": "No results provided"},
                timestamp=datetime.now(),
                success=False,
                error_message="No results provided for aggregation"
            )

        print(f"[RESULT_AGGREGATOR] Aggregating {len(results)} results using {strategy}")
        
        start_time = datetime.now()
        
        try:
            # Normalize results to consistent format
            normalized_results = await self._normalize_results(results)
            
            # Apply aggregation strategy
            if strategy == "weighted_average":
                aggregated = await self._aggregate_weighted_average(normalized_results, weights)
            elif strategy == "highest_confidence":
                aggregated = await self._aggregate_highest_confidence(normalized_results)
            elif strategy == "consensus_building":
                aggregated = await self._aggregate_consensus(normalized_results)
            elif strategy == "comprehensive_synthesis":
                aggregated = await self._aggregate_comprehensive_synthesis(normalized_results)
            elif strategy == "majority_vote":
                aggregated = await self._aggregate_majority_vote(normalized_results)
            else:
                raise ValueError(f"Unsupported aggregation strategy: {strategy}")
            
            # Calculate overall quality score
            quality_score = await self._calculate_quality_score(aggregated, normalized_results)
            
            # Create aggregation result
            result = AggregationResult(
                aggregated_content=aggregated["content"],
                confidence=aggregated["confidence"],
                source_count=len(normalized_results),
                aggregation_method=strategy,
                quality_score=quality_score,
                metadata={
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "source_agents": [r.get("agent_id", "unknown") for r in normalized_results],
                    "aggregation_details": aggregated.get("details", {}),
                    "weights_applied": weights is not None
                },
                timestamp=datetime.now()
            )
            
            # Store aggregation history
            self.aggregation_history.append({
                "timestamp": result.timestamp,
                "strategy": strategy,
                "source_count": len(results),
                "success": True,
                "quality_score": quality_score
            })
            
            print(f"[RESULT_AGGREGATOR] Aggregation completed: confidence {result.confidence:.1%}, quality {quality_score:.1%}")
            
            return result
            
        except Exception as e:
            error_result = AggregationResult(
                aggregated_content=f"Aggregation failed: {str(e)}",
                confidence=0.0,
                source_count=len(results),
                aggregation_method=strategy,
                quality_score=0.0,
                metadata={"error": str(e), "processing_time": (datetime.now() - start_time).total_seconds()},
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            # Store failed aggregation
            self.aggregation_history.append({
                "timestamp": error_result.timestamp,
                "strategy": strategy,
                "source_count": len(results),
                "success": False,
                "error": str(e)
            })
            
            print(f"[RESULT_AGGREGATOR] Aggregation failed: {str(e)}")
            return error_result

    async def _normalize_results(self, results: List[Union[AgentResult, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Normalize results to consistent format for aggregation."""
        normalized = []
        
        for i, result in enumerate(results):
            if isinstance(result, AgentResult):
                normalized.append({
                    "index": i,
                    "agent_id": result.agent_id,
                    "content": result.content,
                    "confidence": result.confidence,
                    "metadata": result.metadata,
                    "timestamp": result.timestamp,
                    "success": result.success
                })
            elif isinstance(result, dict):
                # Handle various dict formats
                content = result.get("content", result.get("result", str(result)))
                confidence = result.get("confidence", result.get("score", 0.5))
                
                normalized.append({
                    "index": i,
                    "agent_id": result.get("agent_id", f"agent_{i}"),
                    "content": str(content),
                    "confidence": float(confidence),
                    "metadata": result.get("metadata", {}),
                    "timestamp": result.get("timestamp", datetime.now()),
                    "success": result.get("success", True)
                })
            else:
                # Handle other types
                normalized.append({
                    "index": i,
                    "agent_id": f"source_{i}",
                    "content": str(result),
                    "confidence": 0.5,  # Default confidence
                    "metadata": {},
                    "timestamp": datetime.now(),
                    "success": True
                })
        
        return normalized

    async def _aggregate_weighted_average(self, results: List[Dict[str, Any]], 
                                        weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Aggregate using weighted average of confidences."""
        if weights is None:
            weights = {str(i): 1.0 for i in range(len(results))}
        
        total_weight = 0
        weighted_confidence = 0
        content_parts = []
        
        for result in results:
            weight = weights.get(str(result["index"]), weights.get(result["agent_id"], 1.0))
            total_weight += weight
            weighted_confidence += result["confidence"] * weight
            
            content_parts.append({
                "content": result["content"],
                "weight": weight,
                "confidence": result["confidence"],
                "agent_id": result["agent_id"]
            })
        
        # Calculate weighted average confidence
        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        
        # Create aggregated content
        aggregated_content = []
        aggregated_content.append("WEIGHTED AGGREGATION RESULT")
        aggregated_content.append("=" * 40)
        aggregated_content.append(f"Combined confidence: {avg_confidence:.1%}")
        aggregated_content.append("")
        
        # Sort by weight and include content
        content_parts.sort(key=lambda x: x["weight"], reverse=True)
        for part in content_parts:
            weight_pct = (part["weight"] / total_weight) * 100
            aggregated_content.append(f"SOURCE: {part['agent_id']} (Weight: {weight_pct:.1f}%)")
            aggregated_content.append(f"Confidence: {part['confidence']:.1%}")
            aggregated_content.append(f"Content: {part['content'][:150]}...")
            aggregated_content.append("")
        
        return {
            "content": "\n".join(aggregated_content),
            "confidence": avg_confidence,
            "details": {
                "total_weight": total_weight,
                "weighted_contributions": content_parts
            }
        }

    async def _aggregate_highest_confidence(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select result with highest confidence."""
        best_result = max(results, key=lambda x: x["confidence"])
        
        return {
            "content": f"HIGHEST CONFIDENCE RESULT\nSelected from {best_result['agent_id']}\n\n{best_result['content']}",
            "confidence": best_result["confidence"],
            "details": {
                "selected_agent": best_result["agent_id"],
                "all_confidences": [(r["agent_id"], r["confidence"]) for r in results]
            }
        }

    async def _aggregate_consensus(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build consensus from multiple results."""
        # Calculate average confidence
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        
        # Find common themes (simplified - could be enhanced with NLP)
        word_frequency = {}
        for result in results:
            words = result["content"].lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Get most common themes
        common_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        consensus_content = []
        consensus_content.append("CONSENSUS BUILDING RESULT")
        consensus_content.append("=" * 40)
        consensus_content.append(f"Consensus confidence: {avg_confidence:.1%}")
        consensus_content.append(f"Sources analyzed: {len(results)}")
        consensus_content.append("")
        
        if common_words:
            consensus_content.append("Common themes identified:")
            for word, frequency in common_words:
                consensus_content.append(f"- {word}: mentioned {frequency} times")
            consensus_content.append("")
        
        # Include all source perspectives
        for i, result in enumerate(results, 1):
            consensus_content.append(f"PERSPECTIVE {i}: {result['agent_id']}")
            consensus_content.append(f"Confidence: {result['confidence']:.1%}")
            consensus_content.append(f"Summary: {result['content'][:100]}...")
            consensus_content.append("")
        
        return {
            "content": "\n".join(consensus_content),
            "confidence": avg_confidence,
            "details": {
                "common_themes": common_words,
                "consensus_strength": len(common_words) / 10  # Normalized
            }
        }

    async def _aggregate_comprehensive_synthesis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive synthesis of all results."""
        total_confidence = sum(r["confidence"] for r in results)
        avg_confidence = total_confidence / len(results)
        
        # Sort results by confidence
        sorted_results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        
        synthesis_content = []
        synthesis_content.append("COMPREHENSIVE SYNTHESIS")
        synthesis_content.append("=" * 50)
        synthesis_content.append(f"Synthesized from {len(results)} sources")
        synthesis_content.append(f"Overall confidence: {avg_confidence:.1%}")
        synthesis_content.append("")
        
        # Executive summary
        synthesis_content.append("EXECUTIVE SUMMARY")
        synthesis_content.append("-" * 20)
        synthesis_content.append(f"Analysis incorporates insights from {len(results)} different agents/sources.")
        synthesis_content.append(f"Confidence levels range from {min(r['confidence'] for r in results):.1%} to {max(r['confidence'] for r in results):.1%}.")
        synthesis_content.append("")
        
        # Detailed synthesis
        synthesis_content.append("DETAILED SYNTHESIS")
        synthesis_content.append("-" * 20)
        
        for i, result in enumerate(sorted_results, 1):
            synthesis_content.append(f"{i}. ANALYSIS FROM {result['agent_id'].upper()}")
            synthesis_content.append(f"   Confidence: {result['confidence']:.1%}")
            synthesis_content.append(f"   Contribution: {result['content']}")
            synthesis_content.append("")
        
        # Cross-analysis insights
        synthesis_content.append("CROSS-ANALYSIS INSIGHTS")
        synthesis_content.append("-" * 25)
        
        high_confidence_count = sum(1 for r in results if r["confidence"] > 0.8)
        if high_confidence_count > len(results) / 2:
            synthesis_content.append("• High consensus with majority of sources showing strong confidence")
        else:
            synthesis_content.append("• Mixed confidence levels suggest need for additional validation")
        
        if len(set(r["agent_id"] for r in results)) > 1:
            synthesis_content.append("• Multiple agent perspectives provide comprehensive coverage")
        
        return {
            "content": "\n".join(synthesis_content),
            "confidence": avg_confidence,
            "details": {
                "source_confidence_distribution": [(r["agent_id"], r["confidence"]) for r in results],
                "synthesis_quality": avg_confidence,
                "coverage_completeness": len(results) / 5  # Normalized assuming 5 is ideal
            }
        }

    async def _aggregate_majority_vote(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate using majority vote (simplified implementation)."""
        # For simplicity, group by confidence ranges
        high_confidence = [r for r in results if r["confidence"] > 0.8]
        medium_confidence = [r for r in results if 0.5 < r["confidence"] <= 0.8]
        low_confidence = [r for r in results if r["confidence"] <= 0.5]
        
        # Determine majority
        if len(high_confidence) >= len(results) / 2:
            majority_group = high_confidence
            majority_level = "high"
            overall_confidence = sum(r["confidence"] for r in high_confidence) / len(high_confidence)
        elif len(medium_confidence) >= len(results) / 2:
            majority_group = medium_confidence
            majority_level = "medium"
            overall_confidence = sum(r["confidence"] for r in medium_confidence) / len(medium_confidence)
        else:
            majority_group = results  # No clear majority
            majority_level = "mixed"
            overall_confidence = sum(r["confidence"] for r in results) / len(results)
        
        vote_content = []
        vote_content.append("MAJORITY VOTE RESULT")
        vote_content.append("=" * 40)
        vote_content.append(f"Voting outcome: {majority_level.upper()} confidence majority")
        vote_content.append(f"Majority size: {len(majority_group)}/{len(results)} sources")
        vote_content.append(f"Consensus confidence: {overall_confidence:.1%}")
        vote_content.append("")
        
        vote_content.append("MAJORITY OPINION:")
        for result in majority_group:
            vote_content.append(f"• {result['agent_id']}: {result['content'][:80]}...")
        
        return {
            "content": "\n".join(vote_content),
            "confidence": overall_confidence,
            "details": {
                "majority_level": majority_level,
                "vote_distribution": {
                    "high": len(high_confidence),
                    "medium": len(medium_confidence),
                    "low": len(low_confidence)
                }
            }
        }

    async def _calculate_quality_score(self, aggregated_result: Dict[str, Any], 
                                     source_results: List[Dict[str, Any]]) -> float:
        """Calculate quality score for aggregated result."""
        # Factors affecting quality
        source_diversity = len(set(r["agent_id"] for r in source_results))
        confidence_consistency = 1 - (max(r["confidence"] for r in source_results) - min(r["confidence"] for r in source_results))
        average_confidence = aggregated_result["confidence"]
        source_count_factor = min(1.0, len(source_results) / 3)  # Normalize to 3 sources
        
        # Weighted quality score
        quality_score = (
            source_diversity * 0.2 +
            confidence_consistency * 0.3 +
            average_confidence * 0.3 +
            source_count_factor * 0.2
        )
        
        return min(quality_score, 1.0)

    def get_aggregation_metrics(self) -> Dict[str, Any]:
        """Get aggregation performance metrics."""
        if not self.aggregation_history:
            return {"no_history": True}
        
        successful_aggregations = [h for h in self.aggregation_history if h["success"]]
        strategy_usage = {}
        
        for aggregation in self.aggregation_history:
            strategy = aggregation["strategy"]
            if strategy not in strategy_usage:
                strategy_usage[strategy] = {"count": 0, "success_count": 0}
            
            strategy_usage[strategy]["count"] += 1
            if aggregation["success"]:
                strategy_usage[strategy]["success_count"] += 1
        
        return {
            "total_aggregations": len(self.aggregation_history),
            "successful_aggregations": len(successful_aggregations),
            "success_rate": len(successful_aggregations) / len(self.aggregation_history),
            "average_quality_score": sum(h.get("quality_score", 0) for h in successful_aggregations) / len(successful_aggregations) if successful_aggregations else 0,
            "strategy_usage": strategy_usage,
            "supported_strategies": self.supported_strategies
        }

    def clear_history(self):
        """Clear aggregation history."""
        self.aggregation_history.clear()
        print("[RESULT_AGGREGATOR] Aggregation history cleared")